###############################################################################
#  HTTP-FLV 降级通道：用 ffmpeg 把数字人帧编码后直接 RTMP 推到本机 SRS
#
#  为什么不用 SRS 的 rtc_to_rtmp：那条链路要把 WebRTC 的 Opus 转成 AAC，音频转码是
#  异步的，音视频消息偶尔乱序入队；SRS 给 FLV 观众做时间戳纠偏（±250ms）时会被这种
#  乱序带偏，实测视频时间戳出现 200~350ms 的前跳和 ~200ms 的回退，播放端一顿一顿。
#
#  这里改成 RTMP 发布：时间戳由 ffmpeg 按帧数/采样数单调生成，音视频天然在同一条
#  时间线上，SRS 只做转发（http_remux 输出 /live/livestream.flv）。
###############################################################################

import os
import shutil
import subprocess
import threading
import time

import numpy as np

from utils.logger import logger

VIDEO_PTIME = 0.04
AUDIO_PTIME = 0.02
SAMPLE_RATE = 48000  # 与 WebRTC/WS 播放口径一致（48k），避免再转一次
AUDIO_CHUNK_BYTES = int(SAMPLE_RATE * AUDIO_PTIME) * 2  # 960 samples s16le = 1920B
FIRST_FRAME_TIMEOUT = 15.0


def default_push_url() -> str:
    return os.environ.get("FLV_RTMP_URL", "rtmp://127.0.0.1/live/livestream")


def video_encoder() -> str:
    """视频编码器：默认 libx264（软件）；FLV_ENCODER=h264_mxenc 可切到 MACA GPU 编码。"""
    return (os.environ.get("FLV_ENCODER", "libx264") or "libx264").strip()


def ffmpeg_bin(encoder: str = "") -> str:
    if encoder and encoder != "libx264":
        # 只有 MACA 自带的 ffmpeg 才有 h264_mxenc（VPUE 硬件编码器）
        maca_ffmpeg = "/opt/maca/ffmpeg/bin/ffmpeg"
        if os.path.isfile(maca_ffmpeg):
            return maca_ffmpeg
        raise RuntimeError(
            "FLV_ENCODER=%s 需要 MACA ffmpeg（/opt/maca/ffmpeg/bin/ffmpeg），未找到"
            % encoder
        )
    explicit = os.environ.get("FLV_FFMPEG_BIN", "")
    if explicit and os.path.isfile(explicit):
        return explicit
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bundled = os.path.join(root, "runtime", "bin", "ffmpeg")
    if os.path.isfile(bundled):
        return bundled
    return shutil.which("ffmpeg") or "ffmpeg"


def video_scale() -> float:
    try:
        v = float(os.environ.get("FLV_VIDEO_SCALE", "0.75") or 0.75)
    except (TypeError, ValueError):
        return 0.75
    return v if 0.2 <= v <= 1.0 else 0.75


def video_bitrate() -> int:
    try:
        return max(300_000, int(os.environ.get("FLV_VIDEO_BITRATE", "2000000")))
    except (TypeError, ValueError):
        return 2_000_000


class FlvPublisher:
    """从一个 HumanPlayer 的降级帧源取帧，经 ffmpeg 匀速推 RTMP。"""

    def __init__(self, player, sessionid: str, push_url: str = ""):
        self.player = player
        self.sessionid = sessionid
        self.push_url = push_url or default_push_url()
        self.width = 0
        self.height = 0
        self.video_frames = 0
        self.on_exit = None

        self._stop = threading.Event()
        self._proc = None
        self._vpipe = None
        self._apipe = None
        self._threads = []
        self._first_frame = None

    def start(self) -> None:
        slot = self.player.ws_video_slot()
        deadline = time.monotonic() + FIRST_FRAME_TIMEOUT
        while time.monotonic() < deadline and not self._stop.is_set():
            item = slot.pop()
            frame = item[0] if isinstance(item, tuple) else item
            if frame is not None and frame.width and frame.height:
                self._first_frame = frame
                break
            time.sleep(0.02)

        if self._first_frame is not None:
            self.width, self.height = self._first_frame.width, self._first_frame.height
        else:
            # 拿不到首帧时退回形象真实尺寸，避免 ffmpeg 参数和实际帧不一致
            self.width, self.height = 720, 1280
            try:
                frames = getattr(self.player._HumanPlayer__container, "frame_list_cycle", None)
                if frames:
                    self.height, self.width = frames[0].shape[:2]
            except Exception:
                logger.debug("[flv] 解析形象帧尺寸失败", exc_info=True)
            logger.warning("[flv] 未取到首帧，按 %dx%d 启动推流", self.width, self.height)

        self.player.flv_active = True
        self._spawn()
        self._threads = [
            threading.Thread(target=self._video_loop, name="flv-video", daemon=True),
            threading.Thread(target=self._audio_loop, name="flv-audio", daemon=True),
        ]
        for t in self._threads:
            t.start()
        # ffmpeg 连不上 SRS（同名流被占用 / 1935 未监听）会很快退出，尽早失败让页面重试
        time.sleep(0.8)
        if self._proc.poll() is not None:
            raise RuntimeError(
                "ffmpeg 推流启动失败：检查 SRS 是否在运行、是否已有同名流（见 logs/zl-talking/zl-talking.log）"
            )
        logger.info("[flv] ffmpeg 推流已启动 %dx%d -> %s", self.width, self.height, self.push_url)

    def _spawn(self) -> None:
        scale = video_scale()
        bitrate = video_bitrate()
        encoder = video_encoder()
        bin_path = ffmpeg_bin(encoder)
        v_r, v_w = os.pipe()
        a_r, a_w = os.pipe()

        args = [
            bin_path, "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "%dx%d" % (self.width, self.height),
            "-r", "25", "-i", "pipe:%d" % v_r,
            "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-i", "pipe:%d" % a_r,
        ]
        if 0 < scale < 1:
            w = max(2, int(round(self.width * scale / 2)) * 2)
            h = max(2, int(round(self.height * scale / 2)) * 2)
            args += ["-vf", "scale=%d:%d" % (w, h)]
        if encoder == "libx264":
            args += [
                "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
                "-profile:v", "baseline", "-pix_fmt", "yuv420p",
                "-g", "25", "-keyint_min", "25", "-sc_threshold", "0",
                "-b:v", str(bitrate), "-maxrate", str(int(bitrate * 1.2)),
                "-bufsize", str(bitrate * 2),
            ]
        else:
            # MACA VPUE 硬件编码（h264_mxenc）；-preset 1 等价 faster
            args += [
                "-c:v", encoder, "-preset", "1", "-profile:v", "baseline",
                "-pix_fmt", "yuv420p", "-g", "25", "-b:v", str(bitrate),
            ]
        args += [
            "-c:a", "aac", "-ar", "44100", "-b:a", "96k",
            "-f", "flv", self.push_url,
        ]

        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            pass_fds=(v_r, a_r),
            close_fds=True,
        )
        os.close(v_r)
        os.close(a_r)
        self._vpipe = os.fdopen(v_w, "wb", buffering=0)
        self._apipe = os.fdopen(a_w, "wb", buffering=0)

        threading.Thread(target=self._drain_stderr, name="flv-ffmpeg-log", daemon=True).start()
        threading.Thread(target=self._watch_proc, name="flv-ffmpeg-watch", daemon=True).start()

    def _drain_stderr(self) -> None:
        try:
            for line in iter(self._proc.stderr.readline, b""):
                text = line.decode("utf-8", "replace").strip()
                if text:
                    logger.warning("[flv][ffmpeg] %s", text)
        except Exception:
            pass

    def _watch_proc(self) -> None:
        code = self._proc.wait()
        if self._stop.is_set():
            return
        logger.warning("[flv] ffmpeg 退出 code=%s sessionid=%s", code, self.sessionid)
        if self.on_exit:
            try:
                self.on_exit()
            except Exception:
                logger.debug("[flv] on_exit 回调失败", exc_info=True)

    def _video_loop(self) -> None:
        try:
            self._video_loop_inner()
        except Exception:
            logger.exception("[flv] 视频发送线程异常")
        finally:
            self._abort_if_not_stopping()

    def _video_loop_inner(self) -> None:
        slot = self.player.ws_video_slot()
        start = time.monotonic()
        n = 0
        last = self._first_frame
        while not self._stop.is_set():
            target = start + n * VIDEO_PTIME
            now = time.monotonic()
            if now < target:
                self._stop.wait(target - now)
                continue

            item = slot.pop()
            frame, seq = item if isinstance(item, tuple) else (item, -1)
            if frame is not None:
                last = frame
            if last is None:
                n += 1
                continue

            try:
                data = last.to_ndarray(format="bgr24").tobytes()
                self._vpipe.write(data)
            except (BrokenPipeError, ValueError, OSError):
                break
            self.video_frames += 1
            # frame 已经推给 ffmpeg（25fps 匀速，ffmpeg 内部时间戳自然对齐）
            if frame is not None and seq >= 0 and seq > self.player.flv_video_sent_seq:
                self.player.flv_video_sent_seq = seq
            n += 1
            if now - target > 1.0:  # 被调度卡住超过 1s：重新对齐
                start = now
                n = 0

    def _audio_loop(self) -> None:
        try:
            self._audio_loop_inner()
        except Exception:
            logger.exception("[flv] 音频发送线程异常")
        finally:
            self._abort_if_not_stopping()

    def _audio_loop_inner(self) -> None:
        slot = self.player.ws_audio_slot()
        silence = b"\x00" * AUDIO_CHUNK_BYTES
        start = time.monotonic()
        n = 0
        pending = None
        while not self._stop.is_set():
            if pending is None:
                item = slot.pop()
                if item is not None:
                    pending = item if isinstance(item, tuple) else (item, -1)

            target = start + n * AUDIO_PTIME
            now = time.monotonic()
            if now < target:
                self._stop.wait(target - now)
                continue

            if pending is None:
                pcm = silence
            else:
                frame, seq = pending
                if seq >= 0:
                    # 与 WS/WebRTC 同一套门控：对应视频帧没推出去之前，这个 20ms 槽位先发静音，
                    # 宁可声音稍晚也不要跑到口型前面。
                    deadline = time.monotonic() + 0.2
                    while (self.player.flv_video_sent_seq < seq
                           and time.monotonic() < deadline and not self._stop.is_set()):
                        self._stop.wait(0.005)
                    if self.player.flv_video_sent_seq < seq:
                        pcm = silence
                        n += 1
                        continue
                pending = None
                try:
                    pcm = frame.to_ndarray().astype(np.int16).tobytes()
                except Exception:
                    pcm = silence
                # 每个槽位固定 20ms：多截少补，避免音频时钟漂移
                if len(pcm) > AUDIO_CHUNK_BYTES:
                    pcm = pcm[:AUDIO_CHUNK_BYTES]
                elif len(pcm) < AUDIO_CHUNK_BYTES:
                    pcm += b"\x00" * (AUDIO_CHUNK_BYTES - len(pcm))

            try:
                self._apipe.write(pcm)
            except (BrokenPipeError, ValueError, OSError):
                break
            n += 1
            if now - target > 1.0:
                start = now
                n = 0

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _abort_if_not_stopping(self) -> None:
        """发送线程异常退出时收掉 ffmpeg，让页面心跳失败后自动重建。"""
        if self._stop.is_set():
            return
        logger.warning("[flv] 发送线程退出，停止 ffmpeg sessionid=%s", self.sessionid)
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

    def stop(self) -> None:
        self._stop.set()
        self.player.flv_active = False
        for pipe in (self._vpipe, self._apipe):
            try:
                if pipe is not None:
                    pipe.close()
            except Exception:
                pass
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        for t in self._threads:
            t.join(timeout=2)
        logger.info("[flv] ffmpeg 推流已停止 sessionid=%s", self.sessionid)
