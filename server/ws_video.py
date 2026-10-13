###############################################################################
#  WebSocket 音视频降级通道（特殊环境专用，默认不启用）
#
#  背景：部分部署平台把 WebRTC 的 UDP 媒体面拦死，浏览器始终停在
#  「ICE 断开」，看不到数字人画面。本模块提供一条纯 TCP 的降级通道：
#  浏览器经 (nginx) /zltalk-ws/ws/video 连上来，服务端复用 /offer 已创建的
#  同一会话（同一个 HumanPlayer），把视频编码成 JPEG、音频转成 PCM16LE 推给浏览器。
#
#  设计要点：
#  - 默认不生效：只有客户端主动连接 /ws/video 才会走这里，WebRTC 路径行为不变。
#  - 复用会话：带 ?sessionid=xxx 时复用 /offer 建立的会话（同一形象、同一渲染），
#    避免同一路数字人被渲染两份；未带 sessionid 时回退为最近一个有播放器的会话，
#    再不行才自建会话。
#  - 会话保护：复用期间对会话加“租约”，避免 WebRTC pc 因 ICE 失败而销毁该会话
#    （见 session_manager.lease_session）。租约释放时若 pc 早已销毁则补做清理。
###############################################################################

import asyncio
import json

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from server.session_manager import session_manager
from server.webrtc import HumanPlayer
from utils.logger import logger

# JPEG 质量：流畅优先（数字人画面 256x256，80 质量足够）
JPEG_QUALITY = 80
DEFAULT_W, DEFAULT_H, DEFAULT_FPS = 256, 256, 25
# 等首帧的超时：形象冷加载可能较久，这里只在渲染线程已能出帧时才会很快返回
FIRST_FRAME_TIMEOUT = 20.0
MAX_WS_MSG = 8 * 1024 * 1024

BIN_JPEG = 0x01
BIN_PCM = 0x02


def _player_of(sessionid):
    """取指定会话上的 HumanPlayer（/offer 建立的那个），没有则返回 None。"""
    sess = session_manager.get_session(sessionid)
    if sess is None:
        return None, None
    return sess, getattr(getattr(sess, "output", None), "_player", None)


def _latest_session_with_player():
    """没有 sessionid 时的兜底：取最近一个挂了 HumanPlayer 的会话。"""
    for sid in reversed(list(session_manager.sessions.keys())):
        sess, player = _player_of(sid)
        if sess is not None and player is not None:
            return sid, sess, player
    return None, None, None


async def ws_video(request: web.Request) -> web.WebSocketResponse:
    query_sid = (request.query.get("sessionid") or "").strip()

    sessionid, avatar_session, player = (None, None, None)
    if query_sid:
        avatar_session, player = _player_of(query_sid)
        sessionid = query_sid if avatar_session is not None else None
        if avatar_session is None:
            logger.warning("ws_video: sessionid=%s 不存在，尝试复用最近会话", query_sid)
    if avatar_session is None:
        sessionid, avatar_session, player = _latest_session_with_player()

    own_session = False
    if avatar_session is None:
        # 完全没有可用会话：自建一个（此时数字人由 WS 通道单独驱动）
        try:
            sessionid = await session_manager.create_session({})
        except Exception:
            logger.exception("ws_video: create_session failed")
            return web.Response(status=500, text="create session failed")
        avatar_session = session_manager.get_session(sessionid)
        own_session = True
        player = None

    if player is None:
        # 复用会话但没有 WebRTC 播放器（例如纯 WS 访问）时补建一个
        player = HumanPlayer(avatar_session)

    # 复用（非自建）时加租约，避免 pc 因 ICE 失败销毁会话导致流中断
    if not own_session:
        session_manager.lease_session(sessionid)

    video_track = player.video
    audio_track = player.audio

    ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=MAX_WS_MSG, autoping=True)
    await ws.prepare(request)
    logger.info("ws_video: client connected, sessionid=%s reused=%s", sessionid, not own_session)

    send_lock = asyncio.Lock()

    async def send_bytes(payload: bytes) -> None:
        async with send_lock:
            if not ws.closed:
                await ws.send_bytes(payload)

    async def send_json(obj: dict) -> None:
        async with send_lock:
            if not ws.closed:
                await ws.send_str(json.dumps(obj))

    def encode_jpeg(bgr) -> bytes:
        ok, buf = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            return b""
        return bytes((BIN_JPEG,)) + buf.tobytes()

    # ── 先取一帧确定分辨率（同时会拉起渲染线程），再发 ready ──
    first_bgr = None
    try:
        frame = await asyncio.wait_for(video_track.recv(), timeout=FIRST_FRAME_TIMEOUT)
        first_bgr = frame.to_ndarray(format="bgr24")
    except Exception as e:
        logger.warning("ws_video: 等待首帧失败/超时: %s", e)

    if first_bgr is not None:
        fh, fw = first_bgr.shape[:2]
    else:
        fh, fw = DEFAULT_H, DEFAULT_W

    fps = int(getattr(request.app.get("opt"), "fps", None) or DEFAULT_FPS)
    await send_json({
        "type": "ready",
        "sessionid": sessionid,
        "video": {"w": int(fw), "h": int(fh), "fps": fps},
        "audio": {"rate": 16000, "channels": 1, "format": "s16le"},
    })

    if first_bgr is not None:
        data = encode_jpeg(first_bgr)
        if data:
            await send_bytes(data)

    async def pump_video() -> None:
        while not ws.closed:
            try:
                frame = await video_track.recv()
            except asyncio.CancelledError:
                raise
            except Exception:
                break
            data = encode_jpeg(frame.to_ndarray(format="bgr24"))
            if data:
                await send_bytes(data)

    async def pump_audio() -> None:
        while not ws.closed:
            try:
                frame = await audio_track.recv()
            except asyncio.CancelledError:
                raise
            except Exception:
                break
            pcm = frame.to_ndarray().astype(np.int16).tobytes()
            await send_bytes(bytes((BIN_PCM,)) + pcm)

    tasks = [asyncio.ensure_future(pump_video()), asyncio.ensure_future(pump_audio())]
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("type") == "stop":
                    break
            elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                break
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("ws_video: receive loop error sessionid=%s", sessionid)
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

        if own_session:
            session_manager.stop_session(sessionid)
            session_manager.remove_session(sessionid)
        else:
            # pc 若已因 ICE 失败销毁过该会话，release 会返回 True → 这里补做清理
            if session_manager.release_session(sessionid):
                session_manager.stop_session(sessionid)
                session_manager.remove_session(sessionid)
        logger.info("ws_video: client disconnected, sessionid=%s", sessionid)

    return ws
