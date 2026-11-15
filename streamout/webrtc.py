###############################################################################
#  Output — WebRTC 输出
###############################################################################

from streamout.base_output import BaseOutput
from registry import register
from utils.logger import logger
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from avatars.base_avatar import BaseAvatar


@register("streamout", "webrtc")
@register("streamout", "rtcpush")
class WebRTCOutput(BaseOutput):
    """WebRTC 输出模式 — 通过 aiortc 推送音视频"""

    def __init__(self, opt=None, parent: Optional['BaseAvatar'] = None, **kwargs):
        super().__init__(opt, parent)
        self._player = None

    def start(self) -> None:
        """WebRTC 输出由 rtc_manager 管理，此处无需额外启动"""
        pass

    def push_video_frame(self, frame) -> None:
        if self._player:
            self._player.push_video(frame)

    def push_audio_frame(self, frame, eventpoint=None) -> None:
        if self._player:
            self._player.push_audio(frame, eventpoint)



    def get_buffer_size(self) -> int:
        # 优先用「活跃消费者」的积压：只有真正在取帧的通道（WebRTC 轨 / WS / FLV）才算，
        # 避免 ICE 未连通时空跑的 RTP 队列把渲染线程误节流。
        if self._player and hasattr(self._player, 'get_active_backlog'):
            return self._player.get_active_backlog()
        if self._player and hasattr(self._player, 'get_buffer_size'):
            return self._player.get_buffer_size()
        return 0

    def stop(self) -> None:
        pass
