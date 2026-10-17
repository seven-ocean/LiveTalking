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
import os
import time

import cv2
import numpy as np
from aiohttp import WSMsgType, web

from server.session_manager import session_manager
from server.webrtc import HumanPlayer
from utils.logger import logger


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# JPEG 质量：流畅优先（分辨率保持原始帧，不做缩放；可配 WS_VIDEO_JPEG_QUALITY）
JPEG_QUALITY = _env_int("WS_VIDEO_JPEG_QUALITY", 80)
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

    # 关键：真实环境里 ICE 常连不上 → WebRTC 发送端从不取帧 → 渲染线程不会被拉起。
    # WS 通道必须自己把渲染线程启动起来，否则一帧都产不出（表现为黑屏 + 报 256×256 占位尺寸）。
    player.ensure_worker()

    # 复用（非自建）时加租约，避免 pc 因 ICE 失败销毁会话导致流中断
    if not own_session:
        session_manager.lease_session(sessionid)

    # 直接读 WS 专属帧源（push_video/push_audio 额外复制的那一份），
    # 不再消费 aiortc 的轨道 —— 从根上避免两条通道互相抢帧。
    video_slot = player.ws_video_slot()
    audio_slot = player.ws_audio_slot()

    ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=MAX_WS_MSG, autoping=True)
    await ws.prepare(request)
    logger.info("ws_video: client connected, sessionid=%s reused=%s", sessionid, not own_session)
    session_manager.touch(sessionid)

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
    _t0 = time.monotonic()
    while time.monotonic() - _t0 < FIRST_FRAME_TIMEOUT:
        _f = video_slot.pop()
        if _f is not None:
            try:
                first_bgr = _f.to_ndarray(format="bgr24")
            except Exception:
                first_bgr = None
            if first_bgr is not None:
                break
        if ws.closed:
            return ws
        await asyncio.sleep(0.01)

    if first_bgr is not None:
        fh, fw = first_bgr.shape[:2]
    else:
        # 拿不到首帧时不能用 256×256 占位（前端会按它裁画面 → 用户完全看不到内容），
        # 退回到该形象的**真实帧尺寸**。
        fh, fw = DEFAULT_H, DEFAULT_W
        try:
            _frames = getattr(avatar_session, 'frame_list_cycle', None)
            if _frames:
                fh, fw = _frames[0].shape[:2]
        except Exception:
            logger.debug("ws_video: resolve frame size from avatar failed", exc_info=True)
        logger.warning("ws_video: 未取到首帧，按形象真实尺寸上报 %dx%d（渲染线程刚启动时会这样，前端可从后续帧自校正）",
                       fw, fh)

    fps = int(getattr(request.app.get("opt"), "fps", None) or DEFAULT_FPS)
    frame_interval = 1.0 / max(1, fps)
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
        """从自己那条队列按 fps 节拍逐帧输出（彻底不碰 aiortc 的轨道）。

        两个要点：
        1. **必须用队列**：`process_frames` 以推理批为粒度突发产出（实测每约 600ms
           推 16 帧）。队列把这 16 帧摊平，正好平滑成 25fps；若只留最新帧，每个突发
           只送出 1 帧、丢掉 15 帧（实测掉到 1.6-2.7fps）。
        2. **按 fps 节拍**：原实现完全不节流，产出抖动会被原样暴露成卡顿，
           且慢客户端下帧量可失控（实测能跑到数千 fps 并把进程 OOM）。
        """
        next_tick = time.monotonic()
        n = 0
        while not ws.closed:
            frame = video_slot.pop()
            if frame is None:
                # 队列空（这一批还没来）：不积累节拍，避免下一批到达后补发式突发
                await asyncio.sleep(0.003)
                next_tick = time.monotonic()
                continue

            now = time.monotonic()
            if now < next_tick:
                await asyncio.sleep(next_tick - now)
            elif now - next_tick > 1.0:
                next_tick = now              # 落后超过 1s 重新对齐，避免越睡越慢
            next_tick += frame_interval

            try:
                bgr = await asyncio.to_thread(frame.to_ndarray, "bgr24")
                data = await asyncio.to_thread(encode_jpeg, bgr)
            except Exception:
                logger.exception("ws_video: encode failed")
                continue
            if not data:
                continue

            t_send = time.monotonic()
            await send_bytes(data)
            if time.monotonic() - t_send > frame_interval * 2:
                # 发送耗时超过两拍 → 网络已回压，重排节拍（队列本身已在丢最旧帧）
                next_tick = time.monotonic() + frame_interval

            n += 1
            if n % 250 == 0:
                logger.info("[ws] 已推送 %d 帧（队列积压 %d，目标 %d fps）",
                            n, video_slot.qsize(), fps)

    async def pump_audio() -> None:
        while not ws.closed:
            frame = audio_slot.pop()
            if frame is None:
                await asyncio.sleep(0.005)
                continue
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
        # 收尾：没有 aiortc 轨道在消费时停掉渲染线程，避免断连后线程泄漏
        try:
            player.release_worker()
        except Exception:
            logger.debug("ws_video: release_worker failed", exc_info=True)

        if own_session:
            session_manager.stop_session(sessionid)
            session_manager.remove_session(sessionid)
        else:
            # pc 若已因 ICE 失败销毁过该会话，release 会返回 True → 这里补做清理
            _pending = session_manager.release_session(sessionid)
            # 关键补漏：ICE 一直卡在 checking 时 pc **永远不会**变成 failed，
            # 原有「pc 失败即清理」路径因此永不触发 → 浏览器关掉后会话一直留着占内存
            # （实测用户关掉两路后仍有 2 个 session）。所以这里主动判定：
            # WS 已断开且该会话已无任何活着的 pc → 立即销毁。
            _rtc = request.app.get("rtc_manager")
            _no_live_pc = True
            if _rtc is not None:
                try:
                    _no_live_pc = not _rtc.session_has_live_pc(sessionid, require_connected=True)
                except Exception:
                    logger.debug("ws_video: session_has_live_pc failed", exc_info=True)
                    _no_live_pc = False
            if _pending or _no_live_pc:
                logger.info("ws_video: 回收会话 sessionid=%s（pending=%s, no_live_pc=%s）",
                            sessionid, _pending, _no_live_pc)
                session_manager.stop_session(sessionid)
                session_manager.remove_session(sessionid)
                if _rtc is not None:
                    try:
                        await _rtc.drop_session_pcs(sessionid)
                    except Exception:
                        logger.debug("ws_video: drop_session_pcs failed", exc_info=True)
        logger.info("ws_video: client disconnected, sessionid=%s", sessionid)

    return ws
