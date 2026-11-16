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
import base64
import json
import os
import time

import cv2
import numpy as np
import resampy
from aiohttp import WSMsgType, web

from server.session_manager import session_manager
from server.webrtc import HumanPlayer
from server.ws_h264 import BIN_H264, H264Encoder
from utils.logger import logger

# JPEG 编码/缩放已经很小，OpenCV 再开多线程只会和推理线程抢核
cv2.setNumThreads(1)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# 默认走 H.264（独立 x264 进程 + 客户端 WebCodecs 硬解），尺度 0.75 = 540x960；
# 不支持 WebCodecs 的浏览器回落 JPEG（以下参数用于该兜底路径）。
JPEG_QUALITY = _env_int("WS_VIDEO_JPEG_QUALITY", 60)
VIDEO_SCALE = _env_float("WS_VIDEO_SCALE", 0.75)
VIDEO_BITRATE = _env_int("WS_VIDEO_BITRATE", 2_000_000)
# 可选：WS 降级通道单独限制帧率（0=跟随数字人 25fps）。低配客户端可用 20 换取稳定。
WS_FPS = _env_int("WS_VIDEO_FPS", 0)
DEFAULT_W, DEFAULT_H, DEFAULT_FPS = 256, 256, 25
# 等首帧的超时：形象冷加载可能较久，这里只在渲染线程已能出帧时才会很快返回
FIRST_FRAME_TIMEOUT = 20.0
MAX_WS_MSG = 8 * 1024 * 1024
# 单次 WS 写入的超时。客户端掉线（尤其是半开连接：TCP 已断但没有 FIN/RST）时，
# send_bytes 会一直卡在写满的发送缓冲上，整个 handler 永远进不了 finally ——
# 表现为编码进程和渲染线程一直泄漏、后台持续吃 CPU。
WS_SEND_TIMEOUT = 5.0

BIN_JPEG = 0x01
BIN_PCM = 0x02

# 每路 WS 的对账统计（供测试页 /ws/stats 查询；只统计，不影响播放逻辑）
_WS_STATS = {}


def snapshot_stats(sessionid, player, video_slot, stats, h264, codec):
    try:
        vs = video_slot.stats() if hasattr(video_slot, 'stats') else {}
        es = h264.stats() if h264 is not None else {}
        _WS_STATS[sessionid] = {
            'codec': codec,
            'ts': time.time(),
            'produced_video': int(getattr(player, 'produced_video', 0)),
            'produced_audio': int(getattr(player, 'produced_audio', 0)),
            'ws_slot': vs,
            'downsample': int(vs.get('drops', 0)),
            'feed_pops': int(stats.get('feed_pops', 0)),
            'feed_none': int(stats.get('feed_none', 0)),
            'aus_sent': int(stats.get('aus_sent', 0)),
            'audio_sent': int(stats.get('audio_sent', 0)),
            'silence_sent': int(stats.get('silence_sent', 0)),
            'enc': es,
            'server_dropped': int(
                stats.get('feed_none', 0)
                + es.get('feed_dropped', 0)
                + es.get('async_dropped', 0)
            ),
        }
    except Exception:
        logger.debug("ws_video: snapshot_stats failed", exc_info=True)


async def ws_stats(request: web.Request) -> web.Response:
    sid = (request.query.get("sessionid") or "").strip()
    data = _WS_STATS.get(sid)
    if not data or time.time() - data.get("ts", 0) > 300:
        _WS_STATS.pop(sid, None)
        return web.json_response({"code": -1, "msg": "no stats"}, status=404)
    return web.json_response({"code": 0, "data": data})

# 静态帧跳过：待机时画面几乎不变，没必要每帧都编 JPEG。用缩小图的平均绝对差判断，
# 成本约 0.1ms（对比 ~8ms 的 JPEG 编码），待机 CPU 能降一大截。
STATIC_DIFF_THRESHOLD = 1.2


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
    want_h264 = (request.query.get("codec") or "jpeg").lower() == "h264"

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

    # 复用会话时队列里可能还压着上一轮的数据（视频最多 32 帧、音频最多 500×20ms）。
    # 不清掉的话，新连接一上来就播旧画面/旧声音 → 音画错位、追赶式卡顿。
    _stale_v = video_slot.clear() if hasattr(video_slot, 'clear') else 0
    _stale_a = audio_slot.clear() if hasattr(audio_slot, 'clear') else 0
    if _stale_v or _stale_a:
        logger.info("ws_video: 清空上一轮残留 视频%d 音频%d", _stale_v, _stale_a)

    ws = web.WebSocketResponse(heartbeat=30.0, max_msg_size=MAX_WS_MSG, autoping=True)
    await ws.prepare(request)
    logger.info("ws_video: client connected, sessionid=%s reused=%s", sessionid, not own_session)
    session_manager.touch(sessionid)

    send_lock = asyncio.Lock()

    async def abort_ws(reason: str) -> None:
        """发送失败/超时：强制关掉连接，让接收循环立刻退出并走 finally 收尾。"""
        logger.warning("ws_video: %s，强制关闭连接 sessionid=%s", reason, sessionid)
        try:
            await asyncio.wait_for(ws.close(), 1.0)
        except Exception:
            pass
        tr = request.transport
        if tr is not None:
            try:
                tr.close()
            except Exception:
                pass

    async def send_bytes(payload: bytes) -> None:
        async with send_lock:
            if ws.closed:
                return
            try:
                await asyncio.wait_for(ws.send_bytes(payload), WS_SEND_TIMEOUT)
            except asyncio.TimeoutError:
                await abort_ws("发送超时（客户端可能已掉线）")
            except (ConnectionResetError, ConnectionError, RuntimeError) as exc:
                await abort_ws("发送失败: %r" % (exc,))

    async def send_json(obj: dict) -> None:
        async with send_lock:
            if ws.closed:
                return
            try:
                await asyncio.wait_for(ws.send_str(json.dumps(obj)), WS_SEND_TIMEOUT)
            except asyncio.TimeoutError:
                await abort_ws("发送超时（客户端可能已掉线）")
            except (ConnectionResetError, ConnectionError, RuntimeError) as exc:
                await abort_ws("发送失败: %r" % (exc,))

    # ── 先取一帧确定分辨率（同时会拉起渲染线程），再发 ready ──
    first_frame = None
    _t0 = time.monotonic()
    while time.monotonic() - _t0 < FIRST_FRAME_TIMEOUT:
        _item = video_slot.pop()
        _f, _seq = _item if isinstance(_item, tuple) else (_item, -1)
        if _f is not None and _f.width and _f.height:
            first_frame = _f
            # 首帧会直接喂给编码器，视频「已发出」序号先推进到它
            if _seq >= 0 and _seq > player.ws_video_sent_seq:
                player.ws_video_sent_seq = _seq
            break
        if ws.closed:
            return ws
        await asyncio.sleep(0.01)

    if first_frame is not None:
        fh, fw = first_frame.height, first_frame.width
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

    fps = WS_FPS or int(getattr(request.app.get("opt"), "fps", None) or DEFAULT_FPS)
    frame_interval = 1.0 / max(1, fps)
    if VIDEO_SCALE and abs(VIDEO_SCALE - 1.0) > 0.01:
        fw = max(2, int(round(fw * VIDEO_SCALE)))
        fh = max(2, int(round(fh * VIDEO_SCALE)))

    # 帧本来就是从 BGR ndarray 建的（HumanPlayer.push_video），to_ndarray 是零拷贝视图；
    # 缩放用 cv2.resize（0.3ms），比 libswscale 的 reformat（约 4ms）快一个数量级。
    thumb_w = max(8, fw // 12)
    thumb_h = max(8, fh // 12)
    last_thumb = [None]

    def encode_jpeg(frame) -> bytes:
        """把一帧转成 WS 二进制消息；画面几乎没变化时返回空（跳过编码）。"""
        try:
            bgr = frame.to_ndarray(format="bgr24")
        except Exception:
            return b""
        if bgr is None or bgr.size == 0:
            return b""
        if bgr.shape[1] != fw or bgr.shape[0] != fh:
            bgr = cv2.resize(bgr, (fw, fh), interpolation=cv2.INTER_AREA)

        small = cv2.resize(bgr, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        prev = last_thumb[0]
        if prev is not None and cv2.absdiff(small, prev).mean() < STATIC_DIFF_THRESHOLD:
            return b""
        last_thumb[0] = small

        ok, buf = cv2.imencode(
            ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            return b""
        return bytes((BIN_JPEG,)) + buf.tobytes()

    h264 = None
    if want_h264:
        try:
            loop = asyncio.get_running_loop()
            h264 = H264Encoder(int(fw), int(fh), fps=fps, bitrate=VIDEO_BITRATE)
            h264.start(loop)
            h264.on_exit = lambda: loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(ws.close())
            )
            if first_frame is not None:
                h264.push_frame(first_frame)
            ok = await loop.run_in_executor(None, h264.wait_ready, 5.0)
            if not ok:
                raise RuntimeError("H.264 编码器 5s 内未输出 SPS/PPS")
        except Exception:
            logger.exception("ws_video: H.264 启动失败，回落 JPEG")
            if h264 is not None:
                try:
                    h264.stop()
                except Exception:
                    pass
            h264 = None

    ready = {
        "type": "ready",
        "sessionid": sessionid,
        "codec": "h264" if h264 is not None else "jpeg",
        "video": {"w": int(fw), "h": int(fh), "fps": fps},
        "audio": {"rate": 16000, "channels": 1, "format": "s16le"},
    }
    if h264 is not None:
        ready["video"].update({
            "codec": h264.codec_string,
            "avcC": base64.b64encode(h264.avcc).decode("ascii"),
        })
    await send_json(ready)
    if h264 is not None:
        logger.info("ws_video: ready %dx%d @%d fps，H.264 %s %d bps，音频 16000 Hz",
                    int(fw), int(fh), fps, h264.codec_string, VIDEO_BITRATE)
    else:
        logger.info("ws_video: ready %dx%d @%d fps，JPEG 质量 %d，音频 16000 Hz",
                    int(fw), int(fh), fps, JPEG_QUALITY)

    if h264 is None and first_frame is not None:
        data = await asyncio.to_thread(encode_jpeg, first_frame)
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
        # 与上游 PlayerStreamTrack.next_timestamp 同一套模型：每个槽位对应绝对墙钟时间
        # （start + slot*interval）。落后时**丢帧追时间**，绝不把时间轴拉长——否则视频
        # 会比音频慢，表现为慢动作、口型越说越对不上。
        start = time.monotonic()
        slot = 0
        n = 0
        enc_ms = send_ms = 0.0
        dropped = 0
        skipped = 0
        while not ws.closed:
            if paused[0]:
                video_slot.clear()
                await asyncio.sleep(frame_interval)
                continue
            item = video_slot.pop()
            frame, seq = item if isinstance(item, tuple) else (item, -1)
            now = time.monotonic()
            if now - (start + slot * frame_interval) > 1.0:
                slot = int((now - start) / frame_interval)

            target = start + slot * frame_interval
            if now < target:
                await asyncio.sleep(target - now)

            if frame is None:
                slot += 1
                continue

            slot += 1

            t_enc = time.monotonic()
            try:
                data = await asyncio.to_thread(encode_jpeg, frame)
            except Exception:
                logger.exception("ws_video: encode failed")
                continue
            enc_ms += (time.monotonic() - t_enc) * 1000
            if data:
                t_send = time.monotonic()
                await send_bytes(data)
                send_ms += (time.monotonic() - t_send) * 1000
            else:
                # 画面几乎没变：跳过编码和发送，客户端保留上一帧
                skipped += 1
            # 无论发没发数据（静态帧客户端保留上一帧，画面等效），这一帧的时间点都已经到了
            if seq >= 0 and seq > player.ws_video_sent_seq:
                player.ws_video_sent_seq = seq

            # CPU/事件循环抖动导致落后时，不丢帧，直接把槽位对齐到当前时间，
            # 避免“越睡越慢”或大批丢帧（丢帧会让画面一顿一顿）。
            if time.monotonic() - target > frame_interval:
                slot = int((time.monotonic() - start) / frame_interval) + 1

            n += 1
            if n % 250 == 0:
                logger.info("[ws] 处理 %d 帧 编码%.1fms 发送%.1fms 跳过%d 积压%d 目标%d fps",
                            n, enc_ms / 250, send_ms / 250, skipped,
                            video_slot.qsize(), fps)
                enc_ms = send_ms = 0.0
                skipped = 0
                dropped = 0

    async def pump_audio() -> None:
        # 与上游 AUDIO_PTIME=20ms 一致：每条 PCM 按时钟逐帧发送。
        # 不节拍的话，TTS 整段音频会瞬间灌满 socket 并长时间占住 send_lock，
        # 视频 pump 排不上队 → 大面积丢帧（实测丢 30%+）。
        ptime = 0.02
        a_start = time.monotonic()
        a_slot = 0
        pending = None  # 已取出、但对应视频还没发出去的那段音频
        silence = bytes((BIN_PCM,)) + b"\x00" * 320 * 2
        while not ws.closed:
            if paused[0]:
                pending = None
                audio_slot.clear()
                await asyncio.sleep(ptime)
                continue
            if pending is None:
                item = audio_slot.pop()
                if item is not None:
                    pending = item if isinstance(item, tuple) else (item, -1)
            if pending is None:
                await asyncio.sleep(0.002)
                continue

            now = time.monotonic()
            target = a_start + a_slot * ptime
            if now < target:
                await asyncio.sleep(target - now)
            elif now - target > 1.0:
                a_slot = int((now - a_start) / ptime)
            a_slot += 1

            frame, seq = pending
            if seq >= 0:
                # 门控：这段音频对应的视频帧必须先发出去。视频落后时最多等 200ms，
                # 超时就用静音占住当前 20ms 槽位（而不是丢掉音频）——宁可声音稍晚，
                # 也不能让声音跑到口型前面。pending 保留，下个槽位继续等。
                deadline = time.monotonic() + 0.2
                while player.ws_video_sent_seq < seq and time.monotonic() < deadline:
                    await asyncio.sleep(0.005)
                if player.ws_video_sent_seq < seq:
                    _stats['silence_sent'] += 1
                    await send_bytes(silence)
                    continue

            # 全链路统一 16k（server/webrtc.py SAMPLE_RATE=16000），直接发送，不再重采样
            arr = frame.to_ndarray()
            pcm = arr.tobytes() if arr.dtype == np.int16 else arr.astype(np.int16).tobytes()
            await send_bytes(bytes((BIN_PCM,)) + pcm)
            _stats['audio_sent'] += 1
            pending = None

    async def pump_h264_feed() -> None:
        """按 25fps 从 WS 帧源取帧喂给编码器；编码在独立 ffmpeg 进程里，不占事件循环。"""
        start = time.monotonic()
        slot = 0
        while not ws.closed:
            if paused[0]:
                video_slot.clear()
                start = time.monotonic()
                slot = 0
                await asyncio.sleep(frame_interval)
                continue
            item = video_slot.pop()
            _stats['feed_pops'] += 1
            frame, seq = item if isinstance(item, tuple) else (item, -1)
            target = start + slot * frame_interval
            now = time.monotonic()
            if now < target:
                await asyncio.sleep(target - now)
            slot += 1
            if frame is None:
                _stats['feed_none'] += 1
                continue
            h264.push_frame(frame, seq)
            if time.monotonic() - target > 1.0:  # 落后太多：重新对齐
                start = time.monotonic()
                slot = 0

    async def send_h264() -> None:
        while not ws.closed:
            item = await h264.next_au()
            if paused[0]:
                # 隐藏期间消费掉编码器输出但不发送，防止 ffmpeg 管道积压
                continue
            key, au, seq = item if len(item) == 3 else (item[0], item[1], -1)
            await send_bytes(bytes((BIN_H264, 1 if key else 0)) + au)
            _stats['aus_sent'] += 1
            # 真正写进 socket 之后才算「视频已发出」，音频门控以此为准
            if seq >= 0 and seq > player.ws_video_sent_seq:
                player.ws_video_sent_seq = seq

    player.ws_active = True
    # 客户端隐藏标签页时发 pause：暂停该路视频/音频发送，避免隐藏页在后台
    # 持续解码/绘制，也避免 socket 积压（多开时把资源让给可见的那一路）。
    paused = [False]
    _stats = {'feed_pops': 0, 'feed_none': 0, 'aus_sent': 0,
              'audio_sent': 0, 'silence_sent': 0}

    async def _stats_loop():
        while not ws.closed:
            snapshot_stats(sessionid, player, video_slot, _stats, h264,
                           'h264' if h264 is not None else 'jpeg')
            await asyncio.sleep(1.0)
    if h264 is not None:
        tasks = [
            asyncio.ensure_future(pump_h264_feed()),
            asyncio.ensure_future(send_h264()),
            asyncio.ensure_future(pump_audio()),
        ]
    else:
        tasks = [asyncio.ensure_future(pump_video()), asyncio.ensure_future(pump_audio())]
    tasks.append(asyncio.ensure_future(_stats_loop()))
    try:
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except Exception:
                    continue
                if isinstance(data, dict) and data.get("type") == "stop":
                    break
                if isinstance(data, dict) and data.get("type") == "pause":
                    paused[0] = True
                    video_slot.clear()
                    audio_slot.clear()
                    logger.info("ws_video: paused by client sessionid=%s", sessionid)
                    continue
                if isinstance(data, dict) and data.get("type") == "resume":
                    paused[0] = False
                    video_slot.clear()
                    audio_slot.clear()
                    logger.info("ws_video: resumed by client sessionid=%s", sessionid)
                    continue
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
        player.ws_active = False
        if h264 is not None:
            try:
                # stop 里要 terminate + join 线程，放线程池执行，别卡住事件循环
                await asyncio.get_running_loop().run_in_executor(None, h264.stop)
            except Exception:
                logger.debug("ws_video: h264.stop failed", exc_info=True)
        # 收尾：没有 aiortc 轨道在消费时停掉渲染线程，避免断连后线程泄漏
        try:
            player.release_worker()
        except Exception:
            logger.debug("ws_video: release_worker failed", exc_info=True)

        if own_session:
            session_manager.stop_session(sessionid)
            session_manager.remove_session(sessionid)
        else:
            # 只释放 WS 租约，**不立即销毁会话**：
            #   1. WS 模式下 pc 因 UDP 被封必然 failed，断开后重连或下一次测试应复用
            #      已加载的形象模型；立即销毁会导致每次重连都重新加载（实测约 11s，
            #      表现为“发完文字半天不回复”）。
            #   2. 泄漏由 app.py 的空闲回收任务兜底（无租约、无活 pc、空闲超时）。
            session_manager.release_session(sessionid)
            session_manager.touch(sessionid)
            logger.info("ws_video: 会话保留待复用 sessionid=%s", sessionid)
        snapshot_stats(sessionid, player, video_slot, _stats, h264,
                       'h264' if h264 is not None else 'jpeg')
        try:
            vs = video_slot.stats() if hasattr(video_slot, 'stats') else {}
            es = h264.stats() if h264 is not None else {}
            logger.info(
                "[ws-stats] sessionid=%s produced_video=%d produced_audio=%d "
                "ws_slot=%s feed_pops=%d feed_none=%d aus_sent=%d "
                "audio_sent=%d silence_sent=%d enc=%s",
                sessionid, getattr(player, 'produced_video', -1),
                getattr(player, 'produced_audio', -1), vs,
                _stats['feed_pops'], _stats['feed_none'], _stats['aus_sent'],
                _stats['audio_sent'], _stats['silence_sent'], es,
            )
        except Exception:
            logger.debug("ws_video: stats log failed", exc_info=True)
        logger.info("ws_video: client disconnected, sessionid=%s", sessionid)

    return ws
