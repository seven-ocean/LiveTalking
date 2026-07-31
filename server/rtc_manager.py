###############################################################################
#  WebRTC 连接管理 + RTC 音频/视频接收
###############################################################################

import json
import asyncio
import random
import copy
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

    def _build_ice_servers(self) -> List[RTCIceServer]:
        """根据配置构建 RTCIceServer 列表，支持 STUN/TURN。"""
        servers = []
        ice_configs = getattr(self.opt, 'ice_servers', None)
        if not ice_configs:
            ice_configs = [{"urls": "stun:stun.freeswitch.org:3478"}]

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

        if not servers:
            logger.warning("No valid ICE servers configured, fallback to default STUN")
            servers = [RTCIceServer(urls="stun:stun.freeswitch.org:3478")]

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

        # 通过 SessionManager 构建
        sessionid = await session_manager.create_session(params)
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
        except Exception:
            webrtc_log.exception(
                "offer processing failed, cleaning up",
                extra={"sessionid": sessionid},
            )
            await _cleanup_pc(pc, sessionid)
            raise

    async def handle_offer(self, request):
        """处理 WebRTC offer 信令，带超时保护。"""
        timeout = getattr(self.opt, 'offer_timeout', 10.0)
        try:
            return await asyncio.wait_for(
                self._do_handle_offer(request),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.exception("handle_offer timeout after %ss", timeout)
            return web.Response(
                status=504,
                content_type="application/json",
                text=json.dumps({
                    "code": -1,
                    "msg": f"Server offer processing timeout ({timeout}s)",
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
