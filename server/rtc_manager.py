###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import random
import copy
import time
from typing import Dict, Optional, List
import queue

from aiohttp import web
from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer, RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender

from utils.logger import logger, get_webrtc_logger
from utils.hardware_monitor import log_resource_snapshot


webrtc_log = get_webrtc_logger()


# def _rand_session_id(n: int = 6) -> int:
#     """生成 N 位随机 session ID"""
#     return random.randint(10 ** (n - 1), 10 ** n - 1)


from server.session_manager import session_manager


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

    async def _do_handle_offer(self, request):
        """实际处理 WebRTC offer 信令（内部方法，便于外层加超时）。"""
        params = await request.json()
        offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

        if False: # 不再由 RTCManager 控制 max_session，让业务逻辑或SessionManager 控制
            logger.info('reach max session')
            return web.Response(
                content_type="application/json",
                text=json.dumps({"code": -1, "msg": "reach max session"}),
            )

        # ── 阶段 1：创建会话（含形象冷加载，可能远超 30s）——独立超时 ──
        session_timeout = getattr(self.opt, 'session_timeout', 120.0)
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
                session_manager.stop_session(sessionid)
                session_manager.remove_session(sessionid)
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
            pc.addTrack(player.video)

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
        webrtc_log.info("local description set", extra={"sessionid": sessionid})

        await asyncio.get_event_loop().run_in_executor(
            None, log_resource_snapshot, "after_answer", sessionid
        )

        return web.Response(
            content_type="application/json",
            text=json.dumps({
                "sdp": pc.localDescription.sdp,
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
        pc.addTrack(player.video)

        await pc.setLocalDescription(await pc.createOffer())

        async with aiohttp.ClientSession() as session:
            async with session.post(push_url, data=pc.localDescription.sdp) as response:
                answer_sdp = await response.text()

        await pc.setRemoteDescription(
            RTCSessionDescription(sdp=answer_sdp, type='answer')
        )

    async def shutdown(self):
        """关闭所有 PeerConnection"""
        coros = [pc.close() for pc in self.pcs]
        await asyncio.gather(*coros)
        self.pcs.clear()
