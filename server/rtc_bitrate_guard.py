"""WebRTC H.264 码率护栏（aiortc 1.14.0）。

背景：REMB 会把 ``H264Encoder.target_bitrate`` 一路顶到 ``MAX_BITRATE``，x264 又只有
``bit_rate``、没有 ``maxrate/bufsize``，720p 数字人实际能跑到配置值的 4~5 倍，弱网下
单帧/I 帧脉冲直接打爆 UDP 队列。

这里在进程内包装 aiortc 的编码器，不复制编码逻辑：

- ``target_bitrate`` setter：下限 ``_zl_min_bps``，上限 ``_zl_cap_bps``（REMB 无法越界）；
- ``_encode_frame``：调用原实现前预建/重建 ``libx264``，补上 ``maxrate/bufsize``，
  并把私有目标码率同步为夹取后的值，避免原实现因为偏差 >10% 再重建一次；
- ``_zl_kf_count``：关键帧/编码器重建计数，供 ``[perf] kf=`` 观察关键帧风暴。

参数由 ``rtc_manager._apply_video_bitrate`` 按会话写入编码器实例；没有写入时用环境变量
``ZL_WEBRTC_RC_*`` 兜底。``ZL_WEBRTC_RC_GUARD=0`` 可整体关闭。
"""

from __future__ import annotations

import fractions
import logging
import os

import av

logger = logging.getLogger(__name__)

_installed = False


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_enabled(name: str) -> bool:
    return str(os.environ.get(name, "1")).strip().lower() not in ("0", "false", "no", "off")


def _bump_keyframe(enc) -> None:
    enc._zl_kf_count = int(getattr(enc, "_zl_kf_count", 0)) + 1
    player = getattr(enc, "_zl_player", None)
    if player is not None:
        player.kf_n = int(getattr(player, "kf_n", 0)) + 1


def _sync_private_target(enc, value: int) -> None:
    """绕过 property setter 直接写 aiortc 的私有字段，避免再次夹取/重建判定。"""
    for name in ("_H264Encoder__target_bitrate", "_VP8Encoder__target_bitrate"):
        if hasattr(enc, name):
            try:
                setattr(enc, name, int(value))
            except Exception:
                pass
            return


def _install_h264() -> None:
    from aiortc.codecs import h264 as _h264

    cls = _h264.H264Encoder
    if getattr(cls, "_zl_guard_installed", False):
        return

    original_prop = cls.target_bitrate
    original_encode_frame = cls._encode_frame

    def _target_setter(self, bitrate):
        try:
            value = int(bitrate)
        except (TypeError, ValueError):
            return original_prop.fset(self, bitrate)
        floor = int(getattr(self, "_zl_min_bps", 0) or _h264.MIN_BITRATE)
        ceiling = int(getattr(self, "_zl_cap_bps", 0) or _h264.MAX_BITRATE)
        floor = max(1, floor)
        ceiling = max(floor, ceiling)
        clamped = max(floor, min(value, ceiling))
        # 记住 REMB 的累计目标；下面按重建阈值决定何时真正改编码器
        self._zl_desired_bps = clamped
        original_prop.fset(self, clamped)

    def _encode_frame(self, frame, force_keyframe):
        try:
            target = int(getattr(self, "_zl_desired_bps", 0) or self.target_bitrate)
            cap = int(getattr(self, "_zl_cap_bps", 0) or 0)
            if cap <= 0:
                cap = int(target * max(1.0, _env_float("ZL_WEBRTC_RC_CAP_FACTOR", 1.5)))
            cap = max(_h264.MIN_BITRATE, min(cap, _h264.MAX_BITRATE))
            maxrate_factor = float(
                getattr(self, "_zl_maxrate_factor", 0)
                or _env_float("ZL_WEBRTC_RC_MAXRATE_FACTOR", 1.2)
            )
            bufsize_ms = float(
                getattr(self, "_zl_bufsize_ms", 0)
                or _env_float("ZL_WEBRTC_RC_BUFSIZE_MS", 200)
            )
            maxrate = max(1, int(cap * max(1.0, maxrate_factor)))
            bufsize = max(1, int(cap * max(0.05, bufsize_ms / 1000.0)))
            rebuild_ratio = max(
                0.1, _env_float("ZL_WEBRTC_RC_REBUILD_RATIO", 0.25))

            need_rebuild = (
                self.codec is None
                or frame.width != self.codec.width
                or frame.height != self.codec.height
                or cap != int(getattr(self, "_zl_effective_cap", 0) or cap)
                or (
                    self.codec.bit_rate
                    and abs(target - self.codec.bit_rate) / self.codec.bit_rate
                    > rebuild_ratio
                )
            )
            if need_rebuild:
                codec = av.CodecContext.create("libx264", "w")
                codec.width = frame.width
                codec.height = frame.height
                codec.bit_rate = min(target, cap)
                codec.pix_fmt = "yuv420p"
                codec.framerate = fractions.Fraction(_h264.MAX_FRAME_RATE, 1)
                codec.time_base = fractions.Fraction(1, _h264.MAX_FRAME_RATE)
                codec.options = {
                    "level": "31",
                    "tune": "zerolatency",
                    "maxrate": str(maxrate),
                    "bufsize": str(bufsize),
                }
                codec.profile = "Baseline"
                previous = self.codec
                self.codec = codec
                self.buffer_data = b""
                self.buffer_pts = None
                self._zl_effective_maxrate = maxrate
                self._zl_effective_bufsize = bufsize
                self._zl_effective_cap = cap
                _sync_private_target(self, codec.bit_rate)
                try:
                    from utils.logger import logger as mylogger
                    mylogger.info(
                        "[webrtc] h264 encoder ready: bit_rate=%d maxrate=%d bufsize=%d "
                        "cap=%d rebuild=%s",
                        codec.bit_rate, maxrate, bufsize, cap, previous is not None)
                except Exception:
                    pass
                if previous is not None:
                    _bump_keyframe(self)
                elif not getattr(self, "_zl_kf_count", 0):
                    _bump_keyframe(self)
            elif self.codec is not None and target != self.codec.bit_rate:
                # 小于重建阈值：把私有 target 同步为当前编码码率，
                # 避免 aiortc 原实现按 >10% 再次重建、白白插入 IDR。
                _sync_private_target(self, self.codec.bit_rate)
            if force_keyframe:
                _bump_keyframe(self)
        except Exception:
            logger.exception("[webrtc] bitrate guard failed, fallback to aiortc encode_frame")
        return original_encode_frame(self, frame, force_keyframe)

    cls.target_bitrate = property(original_prop.fget, _target_setter)
    cls._encode_frame = _encode_frame
    cls._zl_guard_installed = True


def install_bitrate_guard() -> None:
    global _installed
    if _installed:
        return
    _installed = True
    if not _env_enabled("ZL_WEBRTC_RC_GUARD"):
        logger.info("[webrtc] bitrate guard disabled by ZL_WEBRTC_RC_GUARD")
        return
    try:
        _install_h264()
        default_factor = _env_float("ZL_WEBRTC_RC_CAP_FACTOR", 1.5)
        maxrate_factor = _env_float("ZL_WEBRTC_RC_MAXRATE_FACTOR", 1.2)
        bufsize_ms = _env_float("ZL_WEBRTC_RC_BUFSIZE_MS", 200)
        logger.info(
            "[webrtc] bitrate guard installed: default_cap_factor=%.2f "
            "maxrate_factor=%.2f bufsize=%.0fms",
            default_factor, maxrate_factor, bufsize_ms,
        )
        try:
            from utils.logger import logger as mylogger
            mylogger.info(
                "[webrtc] bitrate guard installed: default_cap_factor=%.2f "
                "maxrate_factor=%.2f bufsize=%.0fms",
                default_factor, maxrate_factor, bufsize_ms,
            )
        except Exception:
            pass
    except Exception:
        logger.exception("[webrtc] bitrate guard install failed, keep aiortc defaults")
