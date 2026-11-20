###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import os
import random
import copy
import re
import time
from typing import Dict, Optional, List
import queue

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

# aiortc 默认码率是按“窄带会议”设的（H264 1Mbps / VP8 500kbps），
# 对 720p/1080p 数字人画面明显偏低，表现为静态待机时画面发糊、细节丢失。
# 这里在进程内把默认/下限/上限码率抬到适合数字人的范围（无需改第三方包文件）：
#   - 默认值决定刚建连、REMB 还没上来时的清晰度；
#   - 下限值防止浏览器早期 REMB 偏低时把画面压糊（可用环境变量下调）。
try:
    from aiortc.codecs import h264 as _h264
    from aiortc.codecs import vpx as _vpx
    _h264.DEFAULT_BITRATE = int(os.environ.get("ZL_WEBRTC_H264_BITRATE", "6000000"))
    _h264.MIN_BITRATE = int(os.environ.get("ZL_WEBRTC_H264_MIN_BITRATE", "4000000"))
    _h264.MAX_BITRATE = int(os.environ.get("ZL_WEBRTC_H264_MAX_BITRATE", "12000000"))
    _vpx.DEFAULT_BITRATE = int(os.environ.get("ZL_WEBRTC_VP8_BITRATE", "4000000"))
    _vpx.MIN_BITRATE = int(os.environ.get("ZL_WEBRTC_VP8_MIN_BITRATE", "3000000"))
    _vpx.MAX_BITRATE = int(os.environ.get("ZL_WEBRTC_VP8_MAX_BITRATE", "8000000"))
    print(
        "[webrtc] video bitrate: h264=%d/%d/%d vp8=%d/%d/%d (default/min/max)"
        % (_h264.DEFAULT_BITRATE, _h264.MIN_BITRATE, _h264.MAX_BITRATE,
           _vpx.DEFAULT_BITRATE, _vpx.MIN_BITRATE, _vpx.MAX_BITRATE)
    )
except Exception:
    pass

from utils.logger import logger, get_webrtc_logger
from utils.hardware_monitor import log_resource_snapshot


webrtc_log = get_webrtc_logger()

# 云主机/NAT 部署时，aiortc 的 host 候选是私网 IP，浏览器无法直连。
# 设置 ZL_PUBLIC_IP 后，把 answer SDP 中 host 候选和 c= 行的 IP 改写成公网 IP，
# 配合安全组放行 UDP 端口范围即可直连（EIP/1:1 NAT 场景）。
_PUBLIC_IP = os.environ.get("ZL_PUBLIC_IP", "").strip()


def _rewrite_sdp_public_ip(sdp: str) -> str:
    if not _PUBLIC_IP:
        return sdp
    sdp = re.sub(r"(?m)^c=IN IP4 \S+", f"c=IN IP4 {_PUBLIC_IP}", sdp)
    sdp = re.sub(
        r"(a=candidate:\S+ \d+ udp \d+ )(\S+)( \d+ typ host)",
        lambda m: m.group(1) + _PUBLIC_IP + m.group(3),
        sdp,
    )
    return sdp


_SETUP_LINE_RE = re.compile(r"^a=setup:(\S+)\s*$")


def _hoist_setup_to_session(sdp: str) -> str:
    """把媒体级的 a=setup 提升到会话级。

    RFC 4145 / JSEP 规定 a=setup 只能作为会话级属性；aiortc 的
    MediaDescription.to_sdp() 会把它写进每个 m= 段。旧版 Chrome 容忍这种写法，
    新版 Chrome/Edge 的 SDP 解析器会直接拒绝：
      Failed to parse SessionDescription. a=setup:active Invalid SDP line.
    这里统一提到 m= 之前（会话级），媒体段里的副本删除。
    """
    if "a=setup:" not in sdp:
        return sdp
    lines = sdp.splitlines()
    media_setups = []
    session_setup = None
    in_media = False
    out = []
    for line in lines:
        if line.startswith("m="):
            in_media = True
        m = _SETUP_LINE_RE.match(line)
        if m:
            if in_media:
                media_setups.append(m.group(1))
                continue
            if session_setup is None:
                session_setup = m.group(1)
        out.append(line)
    if not media_setups:
        return sdp
    if session_setup is None:
        first_media = next((i for i, l in enumerate(out) if l.startswith("m=")), len(out))
        t_index = next((i for i, l in enumerate(out[:first_media]) if l.startswith("t=")), None)
        insert_at = (t_index + 1) if t_index is not None else first_media
        out.insert(insert_at, f"a=setup:{media_setups[0]}")
    if len(set(media_setups)) > 1:
        webrtc_log.warning(
            "SDP has mixed media-level a=setup %s, hoisted %s to session level",
            sorted(set(media_setups)), media_setups[0],
        )
    return "\r\n".join(out) + "\r\n"


_FINGERPRINT_RE = re.compile(r"^a=fingerprint:([A-Za-z0-9-]+)\s")


def _keep_sha256_fingerprints(sdp: str) -> str:
    """只保留 sha-256 的 a=fingerprint。

    aiortc 会同时给 sha-256/sha-384/sha-512 三条；新版 Chrome 的 SDP 解析器
    只接受 sha-256（现场报 `a=fingerprint:sha-512 ... Invalid SDP line`）。
    DTLS 校验用同一张证书，保留 sha-256 足够。
    """
    if "a=fingerprint:" not in sdp:
        return sdp
    lines = sdp.splitlines()
    has_sha256 = any(
        (m := _FINGERPRINT_RE.match(line)) and m.group(1).lower() == "sha-256"
        for line in lines
    )
    if not has_sha256:
        return sdp
    out = []
    dropped = 0
    for line in lines:
        m = _FINGERPRINT_RE.match(line)
        if m and m.group(1).lower() != "sha-256":
            dropped += 1
            continue
        out.append(line)
    if dropped:
        webrtc_log.info("SDP: dropped %d non-sha256 fingerprint line(s) for Chrome parser", dropped)
    return "\r\n".join(out) + "\r\n"


def _normalize_answer_sdp(sdp: str) -> str:
    """把 aiortc 生成的 answer 归一成新版 Chrome 能解析的形式。"""
    return _keep_sha256_fingerprints(_hoist_setup_to_session(sdp))


async def _wait_ice_gathering(pc, timeout: float = 3.0) -> None:
    """等 aiortc 收集完 ICE 候选再返回 answer，避免 SDP 里没有 candidate。"""
    if pc.iceGatheringState == "complete":
        return
    loop = asyncio.get_event_loop()
    fut = loop.create_future()

    @pc.on("icegatheringstatechange")
    def _on_state_change():
        if pc.iceGatheringState == "complete" and not fut.done():
            fut.set_result(None)

    # 注册回调后立即再查一次，避免状态刚好在注册前变成 complete 导致白等超时
    if pc.iceGatheringState == "complete" and not fut.done():
        fut.set_result(None)

    try:
        await asyncio.wait_for(fut, timeout)
    except asyncio.TimeoutError:
        pass


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager
from server.model_switch import ModelSwitchRequired  # [fix-publish-webrtc]:0693f81638


def _force_target_bitrate(enc, bps: int) -> bool:
    """绕过 aiortc 的 [MIN,MAX] 夹取，直接把编码器目标码率钉在 bps。

    aiortc 的 target_bitrate setter 会把值夹到 MIN_BITRATE（本项目默认 4M），
    浏览器 REMB 又能把它顶到 MAX_BITRATE（12M）—— 于是 540P@25 的数字人实际
    跑在 10Mbps 上下。单端口 TURN/TCP 通道扛不住这种突发（TCP 队头阻塞），
    表现就是音频和视频**同时**卡住几百毫秒到数秒（客户端 conceal 与 freeze 同步上涨）。
    """
    if enc is None:
        return False
    attr = '_%s__target_bitrate' % type(enc).__name__
    if not hasattr(enc, attr):
        return False
    try:
        setattr(enc, attr, int(bps))
        return True
    except Exception:
        return False


async def _enforce_bitrate_cap(player, pc, cap_bps: int) -> None:
    """把发送码率压在 cap_bps 以内——只在编码器首次就绪时设一次。

    ⚠ 不要周期性重设：aiortc 的 H264Encoder 在 target 与 codec.bit_rate 相差
    >10% 时会**重建编码器**（新 IDR）。REMB 每秒把它写回 4~12M，我们再压回 1.2M，
    就会造成每秒级的关键帧风暴（实测每个 2s 采样收到 6~14 个关键帧），
    带宽/抖动反而更差。
    """
    sender = getattr(player, 'video_sender', None)
    if not sender or cap_bps <= 0:
        return
    for _ in range(200):          # 最多等 10s
        enc = getattr(sender, '_RTCRtpSender__encoder', None)
        if enc is not None:
            _force_target_bitrate(enc, cap_bps)
            _wrap_encode_timing(player, enc)
            return
        await asyncio.sleep(0.05)


def _wrap_encode_timing(player, enc) -> None:
    """[诊断] 记录每帧 H.264 编码耗时（判断发送端是否被编码拖成「成串发送」）。"""
    if getattr(enc, '_zl_timed', False):
        return
    orig = getattr(enc, 'encode', None)
    if orig is None:
        return

    def encode(frame, force_keyframe=False):
        t0 = time.perf_counter()
        try:
            return orig(frame, force_keyframe)
        finally:
            d = (time.perf_counter() - t0) * 1000.0
            player._enc_n = getattr(player, '_enc_n', 0) + 1
            player._enc_sum = getattr(player, '_enc_sum', 0.0) + d
            if d > getattr(player, '_enc_max', 0.0):
                player._enc_max = d

    enc.encode = encode
    enc._zl_timed = True


async def _apply_video_bitrate(player) -> None:
    """把会话画质档位的码率下发给 H264 编码器（编码器在第一帧才创建）。"""
    bitrate = int(getattr(player, 'video_bitrate', 0) or 0)
    sender = getattr(player, 'video_sender', None)
    if not sender or bitrate <= 0:
        return
    for _ in range(200):          # 最多等 10s
        enc = getattr(sender, '_RTCRtpSender__encoder', None)
        if enc is not None and hasattr(enc, 'target_bitrate'):
            try:
                enc.target_bitrate = bitrate
                logger.info('[webrtc] video target_bitrate -> %d bps', bitrate)
            except Exception:
                pass
            return
        await asyncio.sleep(0.05)


async def _sender_stats_loop(player, pc) -> None:
    """每 5s 采样一次发送侧码率，写进 [perf] 日志（判断是码率给低还是带宽到顶）。"""
    sender = getattr(player, 'video_sender', None)
    if not sender:
        return
    last_bytes = 0
    last_t = time.monotonic()
    while True:
        await asyncio.sleep(5)
        try:
            if pc.connectionState in ('closed', 'failed'):
                return
            st = await sender.getStats()
            # aiortc 的 getStats() 返回**列表**（原来按对象取 bytesSent 恒为 0 → vbps 一直 0）
            if isinstance(st, (list, tuple)):
                total = sum(int(getattr(x, 'bytesSent', 0) or 0) for x in st)
            else:
                total = int(getattr(st, 'bytesSent', 0) or 0)
            now = time.monotonic()
            player.vbps = (total - last_bytes) * 8 / max(0.1, now - last_t)
            last_bytes, last_t = total, now
            enc = getattr(sender, '_RTCRtpSender__encoder', None)
            player.vtb = int(getattr(enc, 'target_bitrate', 0) or 0) if enc is not None else 0
        except Exception:
            continue


def _requested_fps(params) -> int:
    """URL ?video_fps=24 → 24；未指定返回 0（不参与比对）。"""
    for key in ("video_fps", "videoFps"):
        try:
            v = params.get(key)
            if v:
                n = int(float(v))
                return n if 1 <= n <= 25 else 0
        except (TypeError, ValueError, AttributeError):
            pass
    return 0


def _requested_short(params) -> int:
    """从 offer 参数里取客户端要的**短边**（0 = 未指定/原画，由服务端按素材决定）。"""
    for key in ("video_short", "video_short_side", "video_height", "videoHeight"):
        try:
            v = params.get(key)
            if v:
                return int(float(v))
        except (TypeError, ValueError, AttributeError):
            pass
    q = str(params.get("video_quality") or params.get("videoQuality") or "").strip().lower()
    return {"540p": 540, "720p": 720, "1080p": 1080}.get(q, 0)


def _webrtc_video_enabled() -> bool:
    """WebRTC 是否发送视频轨。

    ZL_WEBRTC_VIDEO=0 或存在 ../runtime/conf/webrtc-video.off 时只发音频。
    后者是**热开关**：建会话时现读，不需要重启，用来验证「TCP 上视频把音频顶住」。
    """
    if os.environ.get("ZL_WEBRTC_VIDEO", "1").strip().lower() in ("0", "false", "no", "off"):
        return False
    flag = os.path.abspath(os.path.join(os.getcwd(), "..", "runtime", "conf", "webrtc-video.off"))
    return not os.path.exists(flag)


class RTCManager:
    """
    WebRTC 连接管理器。
    
    管理 PeerConnection 生命周期、音视频轨道收发、DataChannel。
    """

    def __init__(self, opt):
        """
        Args:
            opt: 全局配置
        """
        self.opt = opt
        self.pcs: set = set()
        # HTTP-FLV 降级通道的推流：sessionid -> {publisher, player, own_session, last_ping}
        self.flv_publishers: Dict[str, dict] = {}
        # 并发 publish（多标签页/重试）会各起一路 ffmpeg 抢同一个流名，必须串行化
        self._flv_publish_lock = asyncio.Lock()

    # pc 判定为「仍在尝试连接」的宽限期：超过它还没 connected，就认为是死连接。
    # 依据：ICE 连不上时 aiortc 的 connectionState 会**长期停在 connecting/checking**、
    # 永不变成 failed，若只按 failed/closed 判定，泄漏的会话与 pc 永远不会被回收（实测）。
    PC_CONNECT_GRACE_SECONDS = 60

    def session_has_live_pc(self, sessionid: str, require_connected: bool = False) -> bool:
        """该会话是否还有一个"活着"的 pc。

        默认（require_connected=False）：已 connected，或创建不满宽限期（还在尝试连）。
        require_connected=True：只有**真的连上了**才算活 —— 用于 WS 断开时的回收判定：
        前端只在 WebRTC 连不上时才启用 WS 降级通道，所以「WS 曾连上」本身就意味着
        这条 pc 是连不通的死连接，无需再等宽限期，可立即销毁会话。
        """
        now = time.time()
        for pc in list(self.pcs):
            if getattr(pc, '_zl_sessionid', None) != sessionid:
                continue
            state = getattr(pc, 'connectionState', 'closed')
            if state == 'connected':
                return True
            if require_connected or state in ('failed', 'closed'):
                continue
            created = getattr(pc, '_zl_created', 0)
            if now - created < self.PC_CONNECT_GRACE_SECONDS:
                return True
        return False

    async def drop_session_pcs(self, sessionid: str) -> None:
        """会话被回收时，把它留下的死 pc 一并关掉释放资源。"""
        for pc in list(self.pcs):
            if getattr(pc, '_zl_sessionid', None) != sessionid:
                continue
            self.pcs.discard(pc)
            try:
                await pc.close()
            except Exception:
                pass

    def _build_ice_servers(self) -> List[RTCIceServer]:
        """根据配置构建 RTCIceServer 列表，支持 STUN/TURN。

        配置为空时返回空列表：纯内网/离线场景无需公网 STUN，
        否则 setLocalDescription 收集候选时会等待公网 STUN 事务超时（约 5s）。
        """
        servers = []
        ice_configs = getattr(self.opt, 'ice_servers', None) or []

        for cfg in ice_configs:
            if not isinstance(cfg, dict):
                logger.warning("Invalid ICE server config (not a dict): %s", cfg)
                continue
            urls = cfg.get("urls")
            if not urls:
                logger.warning("Invalid ICE server config (missing urls): %s", cfg)
                continue
            kwargs = {"urls": urls}
            if cfg.get("username"):
                kwargs["username"] = cfg["username"]
            if cfg.get("credential"):
                kwargs["credential"] = cfg["credential"]
            servers.append(RTCIceServer(**kwargs))

        return servers

    def _max_sessions(self) -> int:
        """单进程会话上限；多卡模式由 gpu_lb 按每卡上限控制，worker 侧仍生效。"""
        try:
            return max(1, int(os.environ.get(
                "TALK_MAX_SESSIONS", getattr(self.opt, "max_session", 2) or 2)))
        except (TypeError, ValueError):
            return 2

    def _has_capacity(self, exclude_sid: str = "") -> bool:
        """判断能否再建新会话；满员时按 LRU 释放一个最旧会话。

        两级策略：优先释放「无活 PC、无 WS 租用」的会话；没有才动最旧的活跃会话。
        默认允许“来一个排队就踢最旧”，TALK_QUEUE_EVICT=0 可关闭。
        """
        limit = self._max_sessions()
        if len(session_manager.sessions) < limit:
            return True
        if os.environ.get("TALK_QUEUE_EVICT", "1").strip().lower() in ("0", "false", "no", "off"):
            return False

        def _free(sid):
            try:
                return (not session_manager.is_leased(sid)
                        and not self.session_has_live_pc(sid))
            except Exception:
                return False

        def _not_leased(sid):
            try:
                return not session_manager.is_leased(sid)
            except Exception:
                return False

        evicted = (session_manager.evict_oldest(exclude=exclude_sid, predicate=_free)
                   or session_manager.evict_oldest(exclude=exclude_sid, predicate=_not_leased)
                   or session_manager.evict_oldest(exclude=exclude_sid))
        if evicted:
            logger.info("[capacity] 满员（上限 %d），已释放最旧会话 sessionid=%s", limit, evicted)
        return len(session_manager.sessions) < limit

    def _capacity_response(self) -> web.Response:
        used = len([s for s in session_manager.sessions.values() if s is not None])
        limit = self._max_sessions()
        # 数值只进日志/运维指标，不返回给最终用户
        logger.warning("[capacity] 线路已满（%d/%d），拒绝新会话", used, limit)
        return web.Response(
            status=503,
            content_type="application/json",
            text=json.dumps({
                "code": 503,
                "msg": "当前线路资源紧张，正在为您排队，请稍候…",
                "queue": True,
            }),
        )

    async def _do_handle_offer(self, request):
        """实际处理 WebRTC offer 信令（内部方法，便于外层加超时）。"""
        params = await request.json()

        # 页面关闭/切走时的显式释放：复用 /offer 通道，不要求 SDP，不改 Java relay。
        if params.get("release"):
            sid = (params.get("sessionid") or "").strip()
            if sid:
                session_manager.stop_session(sid)
                session_manager.remove_session(sid)
                logger.info("[capacity] 显式释放会话 sessionid=%s", sid)
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": 0, "released": sid}),
            )

        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        # ── 阶段 1：复用或创建会话（含形象冷加载，可能远超 30s）——独立超时 ──
        # 页面在 ICE 失败后经常连续重试 offer。每次新建会话会多一套渲染/推理线程，
        # 旧会话还要保留一段时间，越用越卡。这里优先复用「没有活跃 WebRTC 连接、
        # 没有被 WS 租用、且形象相同」的会话；默认开启，可用 REUSE_SESSION_ON_OFFER=0 关闭。
        session_timeout = getattr(self.opt, 'session_timeout', 120.0)
        sessionid = ''
        params_sid = (params.get('sessionid') or '').strip() if isinstance(params, dict) else ''

        def _session_matches(sess_opt) -> bool:
            """本会话的档位/帧率/节流模式是否与本次 offer 一致。

            不一致必须重建：预热（/warmup）创建的会话不带这些参数，
            前端重连又常带 sessionid 回来，若不比对就会一直套用 fps=20 的旧会话，
            URL 上的 pacing/bwcap 永远不生效。
            """
            if _requested_short(params) != int(getattr(sess_opt, 'video_short_eff', 0) or 0):
                return False
            _wf = _requested_fps(params)
            if _wf and _wf != int(getattr(sess_opt, 'video_fps', 0) or 0):
                return False
            _wp = str(params.get('pacing') or '').strip().lower()
            if _wp not in ('legacy', 'official', 'bounded', 'sync25', 'runway'):
                _wp = 'legacy'
            if _wp != str(getattr(sess_opt, 'pacing', 'legacy') or 'legacy').lower():
                return False
            _wpc = str(params.get('frame_pace') or '').strip().lower() in ('fill', '1', 'true', 'on')
            if _wpc != bool(getattr(sess_opt, 'frame_pace', False)):
                return False
            return True

        # 只有 URL 显式带 pacing（试参）时才严格比对会话参数；默认路径保持原有行为。
        _explicit_pacing = str(params.get('pacing') or '').strip().lower() in (
            'legacy', 'official', 'bounded', 'sync25')
        if (params_sid and session_manager.has_session(params_sid)
                and not session_manager.is_leased(params_sid)):
            _so = getattr(session_manager.get_session(params_sid), 'opt', None)
            if (not _explicit_pacing) or _session_matches(_so):
                sessionid = params_sid
            else:
                logger.info("[session] 请求参数与会话不符（pacing=%s fps=%s -> pacing=%s fps=%s），重建会话 sessionid=%s",
                            getattr(_so, 'pacing', ''), getattr(_so, 'video_fps', 0),
                            str(params.get('pacing') or 'legacy').strip().lower(),
                            _requested_fps(params), params_sid)
                # 线路容量有限（现场 1/1）：旧会话不先释放，新会话会被「线路已满」拒掉，
                # 前端于是又退回旧会话（fps=20 / legacy），参数永远不生效。
                try:
                    if not self.session_has_live_pc(params_sid, require_connected=True):
                        session_manager.remove_session(params_sid)
                        logger.info("[session] 已释放旧会话（参数不符）sessionid=%s", params_sid)
                except Exception:
                    pass
        elif os.environ.get('REUSE_SESSION_ON_OFFER', '1').strip().lower() in ('1', 'true', 'yes', 'on'):
            want_avatar = params.get('avatar') or getattr(self.opt, 'avatar_id', None)
            for sid in reversed(list(session_manager.sessions.keys())):
                if not session_manager.has_session(sid):
                    continue
                # WS 降级通道正在进行时绝不能换播放器，否则那条 WS 会收不到帧
                if session_manager.is_leased(sid):
                    continue
                # 只认「真的连上了」的 pc；failed/connecting 的都算可复用
                if self.session_has_live_pc(sid, require_connected=True):
                    continue
                sess = session_manager.get_session(sid)
                sess_opt = getattr(sess, 'opt', None)
                if getattr(sess_opt, 'avatar_id', None) != want_avatar:
                    continue
                # 画质档位（目标高度）变了：不能复用旧会话，否则切档后要等回收才生效
                if _requested_short(params) != int(getattr(sess_opt, 'video_short_eff', 0) or 0):
                    logger.info("[quality] 画质档位变化（%s -> %s），重建会话 sessionid=%s",
                                getattr(sess_opt, 'video_short_eff', 0),
                                _requested_short(params), sid)
                    continue
                # 帧率 / 补齐节拍 也要参与比对：否则改 URL 参数后复用旧会话，新参数不生效
                _want_fps = _requested_fps(params)
                if _want_fps and _want_fps != int(getattr(sess_opt, 'video_fps', 0) or 0):
                    logger.info("[quality] 帧率变化（%s -> %s），重建会话 sessionid=%s",
                                getattr(sess_opt, 'video_fps', 0), _want_fps, sid)
                    continue
                # 节流模式变化也要重建会话（否则改 URL 后复用旧会话，新参数不生效）
                _want_pacing = str(params.get('pacing') or '').strip().lower()
                if _want_pacing not in ('legacy', 'official', 'bounded', 'sync25', 'runway'):
                    _want_pacing = 'legacy'
                if _want_pacing != str(getattr(sess_opt, 'pacing', 'legacy') or 'legacy').lower():
                    logger.info("[pacing] 节流模式变化（%s -> %s），重建会话 sessionid=%s",
                                getattr(sess_opt, 'pacing', 'legacy'), _want_pacing, sid)
                    continue
                _want_pace = str(params.get('frame_pace') or '').strip().lower() in ('fill', '1', 'true', 'on')
                if _want_pace != bool(getattr(sess_opt, 'frame_pace', False)):
                    logger.info("[quality] 补齐节拍变化（%s -> %s），重建会话 sessionid=%s",
                                getattr(sess_opt, 'frame_pace', False), _want_pace, sid)
                    continue
                sessionid = sid
                break

        if sessionid:
            session_manager.touch(sessionid)
            try:
                session_manager.pending_destroy.discard(sessionid)
            except Exception:
                pass
            webrtc_log.info(
                "offer reused existing session",
                extra={"sessionid": sessionid},
            )
            avatar_session = session_manager.get_session(sessionid)
        else:
            # 容量控制：满员时按 LRU 释放最旧会话再接收；仍满则短暂排队等待，
            # 避免“别的会话正在加载（占位 None）”时把新用户直接拒掉。
            if not self._has_capacity(exclude_sid=params_sid):
                try:
                    wait_s = float(os.environ.get("TALK_QUEUE_WAIT_SECONDS", "15"))
                except (TypeError, ValueError):
                    wait_s = 15.0
                waited = 0.0
                while waited < wait_s and not self._has_capacity(exclude_sid=params_sid):
                    await asyncio.sleep(0.5)
                    waited += 0.5
                if not self._has_capacity(exclude_sid=params_sid):
                    return self._capacity_response()
            sessionid = await asyncio.wait_for(
                session_manager.create_session(params),
                timeout=session_timeout,
            )
            webrtc_log.info(
                "offer received, creating session",
                extra={"sessionid": sessionid},
            )
            avatar_session = session_manager.get_session(sessionid)

        # 资源快照含 time.sleep(0.5) 的 CPU 采样，放到线程池避免阻塞事件循环
        await asyncio.get_event_loop().run_in_executor(
            None, log_resource_snapshot, "before_offer", sessionid
        )

        # 创建 PeerConnection
        ice_servers = self._build_ice_servers()
        webrtc_log.debug(
            "using ICE servers: %s",
            [s.urls for s in ice_servers],
            extra={"sessionid": sessionid},
        )
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )
        # 标记归属会话，供会话回收判定（见 session_has_live_pc）
        try:
            pc._zl_sessionid = sessionid
            pc._zl_created = time.time()
        except Exception:
            pass
        self.pcs.add(pc)

        # 防止状态变更时重复触发清理
        _cleanup_started = False

        async def _cleanup_pc(pc, sessionid):
            nonlocal _cleanup_started
            if _cleanup_started:
                return
            _cleanup_started = True
            webrtc_log.info(
                "starting pc cleanup",
                extra={"sessionid": sessionid},
            )
            try:
                # pc.close() 内部可能触发同步的 thread.join，用 wait_for 限制在事件循环中的时间；
                # 即使超时也不阻塞循环，后续丢弃引用让 Python 垃圾回收兜底。
                await asyncio.wait_for(pc.close(), timeout=3.0)
            except asyncio.TimeoutError:
                webrtc_log.warning(
                    "pc.close() timed out after 3s, discarding pc",
                    extra={"sessionid": sessionid},
                )
            except Exception as e:
                webrtc_log.warning(
                    "pc.close error: %s",
                    e,
                    extra={"sessionid": sessionid},
                )
            self.pcs.discard(pc)
            # 若会话正被 WS 降级通道租用（复用同一会话），则推迟销毁，
            # 避免 pc 因 ICE 失败关闭时把正在推送的会话一起杀掉。
            if session_manager.is_leased(sessionid):
                session_manager.mark_pending_destroy(sessionid)
                webrtc_log.info(
                    "session leased by ws fallback, defer destroy",
                    extra={"sessionid": sessionid},
                )
            else:
                # 不在这里销毁会话：WS 模式/页面重连会在几秒内复用同一会话，
                # 立即销毁会导致每次 pc 失败都重新加载形象（实测约 11s）。
                # 统一交给 app.py 的空闲回收任务（无租约、无活 pc、空闲超 5 分钟）。
                session_manager.mark_pending_destroy(sessionid)
                webrtc_log.info(
                    "pc closed, session kept for reuse (reaper will clean when idle)",
                    extra={"sessionid": sessionid},
                )
            webrtc_log.info(
                "pc cleanup completed",
                extra={"sessionid": sessionid},
            )

        @pc.on("icecandidate")
        async def on_icecandidate(candidate):
            webrtc_log.debug(
                "local ice candidate: %s",
                candidate,
                extra={"sessionid": sessionid},
            )

        @pc.on("iceconnectionstatechange")
        async def on_iceconnectionstatechange():
            webrtc_log.info(
                "ICE connection state is %s",
                pc.iceConnectionState,
                extra={"sessionid": sessionid},
            )

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            webrtc_log.info(
                "Connection state is %s",
                pc.connectionState,
                extra={"sessionid": sessionid},
            )
            if pc.connectionState in ("failed", "closed"):
                asyncio.create_task(_cleanup_pc(pc, sessionid))

        try:
            # 添加发送轨道
            from server.webrtc import HumanPlayer
            player = HumanPlayer(avatar_session)
            pc.addTrack(player.audio)
            # 音频优先诊断/降载开关：ZL_WEBRTC_VIDEO=0 时不发视频轨，
            # 用来验证「TCP 上视频把音频顶住」这类 HOL 阻塞问题。
            if _webrtc_video_enabled():
                video_sender = pc.addTrack(player.video)
                player.video_sender = video_sender
                try:
                    player.video_bitrate = int(
                        getattr(avatar_session.opt, 'video_bitrate', 0) or 0)
                except Exception:
                    player.video_bitrate = 0
                asyncio.create_task(_apply_video_bitrate(player))
                asyncio.create_task(_sender_stats_loop(player, pc))
                # URL ?bwcap=1200（kbps）或 sync25 会话：把发送码率钉在档位值以内。
                # 不钉的话 REMB 会把它顶到 12M，单端口 TURN/TCP 上音视频会一起卡。
                try:
                    _cap = int(getattr(avatar_session.opt, 'bwcap_kbps', 0) or 0) * 1000
                except Exception:
                    _cap = 0
                if _cap > 0:
                    logger.info('[webrtc] bitrate cap -> %d bps (session %s)',
                                _cap, sessionid)
                    asyncio.create_task(_enforce_bitrate_cap(player, pc, _cap))

            # 设置编解码器偏好
            capabilities = RTCRtpSender.getCapabilities("video")
            preferences = list(filter(lambda x: x.name == "H264", capabilities.codecs))
            preferences += list(filter(lambda x: x.name == "VP8", capabilities.codecs))
            preferences += list(filter(lambda x: x.name == "rtx", capabilities.codecs))
            transceiver = pc.getTransceivers()[1]
            transceiver.setCodecPreferences(preferences)

            # ── 阶段 2：SDP/ICE 协商——独立超时（不覆盖形象加载时间）──
            offer_timeout = getattr(self.opt, 'offer_timeout', 30.0)
            return await asyncio.wait_for(
                self._negotiate(pc, offer, sessionid),
                timeout=offer_timeout,
            )
        except Exception:
            webrtc_log.exception(
                "offer processing failed, cleaning up",
                extra={"sessionid": sessionid},
            )
            await _cleanup_pc(pc, sessionid)
            raise

    async def _negotiate(self, pc, offer, sessionid):
        """SDP/ICE 协商阶段（setRemoteDescription → createAnswer → setLocalDescription）。

        该阶段的耗时主要由 ICE 候选收集（setLocalDescription）决定，与外层
        session_timeout 隔离，由 offer_timeout 单独限时。
        """
        await pc.setRemoteDescription(offer)
        webrtc_log.info("remote description set", extra={"sessionid": sessionid})

        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_ice_gathering(pc, 3.0)
        webrtc_log.info("local description set", extra={"sessionid": sessionid})

        sdp = _rewrite_sdp_public_ip(pc.localDescription.sdp)
        sdp = _normalize_answer_sdp(sdp)
        if not sdp or "v=0" not in sdp:
            raise RuntimeError("answer SDP is empty")  # [fix-publish-webrtc]:04c5862d2e
        if "a=candidate:" not in sdp:
            webrtc_log.warning(
                "answer SDP has no ICE candidate, browser cannot connect",
                extra={"sessionid": sessionid},
            )
        if _PUBLIC_IP:
            webrtc_log.info(
                "rewrote SDP host candidates to public IP %s", _PUBLIC_IP,
                extra={"sessionid": sessionid},
            )

        await asyncio.get_event_loop().run_in_executor(
            None, log_resource_snapshot, "after_answer", sessionid
        )

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": sdp,
                "type": pc.localDescription.type,
                "sessionid": sessionid,
            }),
        )

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令。

        两阶段各自限时（会话创建见 session_timeout，SDP/ICE 协商见 offer_timeout），
        此处仅作薄壳，负责把阶段超时/异常映射为 HTTP 响应。
        """
        try:
            return await self._do_handle_offer(request)
        except ModelSwitchRequired as e:
            # [fix-publish-webrtc]:bc7f213ba3 先回 503，再按原逻辑重启，避免 relay 拿到空 answer
            logger.warning(
                "handle_offer requires model switch: avatar=%s; restart after response",
                e.avatar_id,
            )
            try:
                asyncio.get_running_loop().call_later(0.5, os._exit, 42)
            except RuntimeError:
                pass
            return web.Response(
                status=503,
                content_type="application/json",
                text=json.dumps({
                    "code": 503,
                    "msg": f"model switching to {e.avatar_id}, please retry",
                }),
            )
        except asyncio.TimeoutError as e:
            logger.exception("handle_offer timeout: %s", e)
            return web.Response(
                status=504,
                content_type="application/json",
                text=json.dumps({
                    "code": -1,
                    "msg": f"Server offer processing timeout: {e}",
                }),
            )
        except Exception as e:
            logger.exception("handle_offer exception")
            return web.Response(
                status=500,
                content_type="application/json",
                text=json.dumps({
                    "code": -1,
                    "msg": f"Server offer processing failed: {e}",
                }),
            )

    async def handle_rtcpush(self, push_url, sessionid: str):
        """RTCPush 模式：主动推流"""
        import aiohttp
        await session_manager.create_session({}, sessionid)
        avatar_session = session_manager.get_session(sessionid)

        ice_servers = self._build_ice_servers()
        pc = RTCPeerConnection(
            configuration=RTCConfiguration(iceServers=ice_servers)
        )
        self.pcs.add(pc)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            logger.info("Connection state is %s", pc.connectionState)
            if pc.connectionState == "failed":
                await pc.close()
                self.pcs.discard(pc)

        from server.webrtc import HumanPlayer
        player = HumanPlayer(avatar_session)
        pc.addTrack(player.audio)
        if _webrtc_video_enabled():
            pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    # ─── HTTP-FLV 降级通道 ──────────────────────────────────────────────
    # 按需起一路独立会话，用 ffmpeg 把帧编码成 RTMP 推到本机 SRS，SRS 的
    # http_remux 再输出 /live/livestream.flv。不走 SRS 的 rtc_to_rtmp：
    # 那条链路要做异步音频转码，音视频消息偶尔乱序，SRS 的观众端时间戳纠偏
    # 会把视频时间戳拧来拧去（实测 200~350ms 跳变），播放端一顿一顿。
    # 详见 server/flv_publisher.py。
    async def handle_flv_publish(self, push_url: str = "", sessionid: str = None) -> str:
        async with self._flv_publish_lock:
            return await self._handle_flv_publish_locked(push_url, sessionid)

    async def _handle_flv_publish_locked(self, push_url: str, sessionid: str) -> str:
        from server.flv_publisher import FlvPublisher
        from server.webrtc import HumanPlayer

        # 多标签页/失败重试时复用同一路推流：否则多个 ffmpeg 抢同一个 RTMP 流名，
        # 失败的一方反复重建会话，表现为“老是掉线”。
        for sid, entry in list(self.flv_publishers.items()):
            if entry["publisher"].is_alive():
                entry["last_ping"] = time.time()
                session_manager.touch(sid)
                logger.info("[flv] 复用已有推流 sessionid=%s", sid)
                return sid

        own_session = False
        if not (sessionid and session_manager.has_session(sessionid)):
            sessionid = await session_manager.create_session({})
            own_session = True
        avatar_session = session_manager.get_session(sessionid)

        player = HumanPlayer(avatar_session)
        # 与 WS 通道同理：没有消费者取帧时渲染线程不会自己跑起来
        player.ensure_worker()

        loop = asyncio.get_running_loop()

        async def _try_start():
            publisher = FlvPublisher(player, sessionid, push_url)
            try:
                await loop.run_in_executor(None, publisher.start)
            except Exception:
                await loop.run_in_executor(None, publisher.stop)
                raise
            return publisher

        try:
            publisher = await _try_start()
        except Exception:
            # SRS 上同名流刚释放时 RTMP 会短暂报 I/O error，等 1.5s 再试一次
            await asyncio.sleep(1.5)
            try:
                publisher = await _try_start()
            except Exception:
                try:
                    player.release_worker()
                except Exception:
                    pass
                if own_session:
                    session_manager.stop_session(sessionid)
                    session_manager.remove_session(sessionid)
                raise

        self.flv_publishers[sessionid] = {
            "publisher": publisher,
            "player": player,
            "own_session": own_session,
            "last_ping": time.time(),
        }

        def _on_ffmpeg_exit():
            loop.call_soon_threadsafe(
                lambda: asyncio.ensure_future(self.stop_flv_publish(sessionid))
            )

        publisher.on_exit = _on_ffmpeg_exit

        session_manager.lease_session(sessionid)
        session_manager.touch(sessionid)
        logger.info("[flv] 推流已启动 sessionid=%s url=%s", sessionid, publisher.push_url)
        return sessionid

    def touch_flv_publish(self, sessionid: str) -> bool:
        """浏览器心跳：返回 False 表示推流已不存在。"""
        entry = self.flv_publishers.get(sessionid)
        if entry is None or not entry["publisher"].is_alive():
            return False
        entry["last_ping"] = time.time()
        session_manager.touch(sessionid)
        return True

    def flv_status(self, sessionid: str) -> dict:
        entry = self.flv_publishers.get(sessionid)
        if entry is None:
            return {}
        publisher = entry["publisher"]
        return {
            "video_frames": int(publisher.video_frames),
            "alive": publisher.is_alive(),
        }

    async def stop_flv_publish(self, sessionid: str) -> bool:
        entry = self.flv_publishers.pop(sessionid, None)
        if entry is None:
            return False
        try:
            await asyncio.get_running_loop().run_in_executor(
                None, entry["publisher"].stop
            )
        except Exception:
            logger.debug("[flv] publisher.stop failed", exc_info=True)
        try:
            entry["player"].release_worker()
        except Exception:
            logger.debug("[flv] release_worker failed", exc_info=True)
        session_manager.release_session(sessionid)
        if entry.get("own_session"):
            session_manager.stop_session(sessionid)
            session_manager.remove_session(sessionid)
        else:
            session_manager.touch(sessionid)
        logger.info("[flv] 推流已停止 sessionid=%s", sessionid)
        return True

    async def reap_flv_publishers(self, idle_seconds: float = 45.0) -> None:
        """浏览器异常退出时收掉没人要的推流（页面每 5s 心跳一次）。"""
        now = time.time()
        for sid, entry in list(self.flv_publishers.items()):
            if now - entry.get("last_ping", 0) <= idle_seconds:
                continue
            logger.info("[flv] 推流心跳超时，自动停止 sessionid=%s", sid)
            try:
                await self.stop_flv_publish(sid)
            except Exception:
                logger.exception("[flv] 自动停止失败 sessionid=%s", sid)

    def metrics(self) -> dict:
        """运行指标：会话、PC、每会话播放器计数（/metrics 使用）。"""
        sessions = []
        for item in session_manager.stats():
            sid = item.get("sessionid")
            sess = session_manager.get_session(sid)
            player = None
            try:
                player = getattr(getattr(sess, "output", None), "_player", None)
            except Exception:
                pass
            item = dict(item)
            item["has_pc"] = self.session_has_live_pc(sid)
            if player is not None and hasattr(player, "stats"):
                item["player"] = player.stats()
            sessions.append(item)
        return {
            "used": session_manager.active_count(),
            "max": self._max_sessions(),
            "sessions": sessions,
            "pcs": len(self.pcs),
            "queue_evict": os.environ.get("TALK_QUEUE_EVICT", "1"),
            "reuse_on_offer": os.environ.get("REUSE_SESSION_ON_OFFER", "1"),
        }

    async def shutdown(self):
        """关闭全部 PeerConnection"""
        for sid in list(self.flv_publishers.keys()):
            await self.stop_flv_publish(sid)
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()
