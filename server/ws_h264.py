###############################################################################
#  WS 降级通道的 H.264 编码（libx264 独立进程）
#
#  为什么不用服务进程内编码：talking 进程的 GIL 被推理线程占得很满，进程内 JPEG
#  实测只能稳定输出约 10fps；把编码放到独立 ffmpeg 进程后，WS 侧只做取帧/缩放/写管道，
#  事件循环不再被编码拖住。客户端用 WebCodecs 硬解，画质和 CPU 都优于逐帧 JPEG。
#
#  输出：Annex-B → AVCC（4 字节长度前缀），并从 SPS/PPS 生成 avcC，供 WebCodecs
#  VideoDecoder 的 description 使用。
###############################################################################

import asyncio
import base64
import collections
import os
import queue
import shutil
import signal
import subprocess
import threading
import time

import cv2

from utils.logger import logger

cv2.setNumThreads(1)

BIN_H264 = 0x03  # WS 二进制消息：0x03 + flags(1) + AVCC(4 字节长度前缀的 NAL)


def _ffmpeg_bin() -> str:
    explicit = os.environ.get("WS_FFMPEG_BIN", "")
    if explicit and os.path.isfile(explicit):
        return explicit
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    bundled = os.path.join(root, "runtime", "bin", "ffmpeg")
    if os.path.isfile(bundled):
        return bundled
    return shutil.which("ffmpeg") or "ffmpeg"


def _first_mb_in_slice(nal: bytes) -> int:
    """读 slice header 的 first_mb_in_slice（ue(v)）。

    x264 的 ultrafast 预设默认开 sliced-threads，一帧会被拆成多个 slice NAL；
    只有 first_mb_in_slice==0 才是新帧的第一个 slice，用它来切 access unit。
    """
    data = nal[1:]
    if not data:
        return 0
    bitpos = 0
    total_bits = len(data) * 8

    def read_bit() -> int:
        nonlocal bitpos
        if bitpos >= total_bits or bitpos >= 64:
            return -1
        b = (data[bitpos >> 3] >> (7 - (bitpos & 7))) & 1
        bitpos += 1
        return b

    zeros = 0
    while True:
        bit = read_bit()
        if bit != 0:
            break
        zeros += 1
        if zeros > 32 or bit < 0:
            return 0
    if zeros == 0:
        return 0
    val = 1
    for _ in range(zeros):
        bit = read_bit()
        if bit < 0:
            return 0
        val = (val << 1) | bit
    return val - 1


class H264Encoder:
    """把 av.VideoFrame 匀速喂给 ffmpeg，输出按 access unit 切好的 AVCC。"""

    def __init__(self, width: int, height: int, fps: int = 25, bitrate: int = 2_000_000):
        self.width = width
        self.height = height
        self.fps = max(1, fps)
        self.bitrate = bitrate
        # GOP 0.4s：掉帧/丢参考帧后，最多 0.4s 就能靠下一个关键帧恢复，
        # 而不是像 1s GOP 那样整屏马赛克持续一整秒。
        self._gop = max(1, int(round(self.fps * 0.4)))

        self.avcc = b""
        self.codec_string = ""

        self._loop = None
        self._async_q = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._feed_q = queue.Queue(maxsize=4)

        self._proc = None
        self._threads = []
        self.on_exit = None

        self._sps = None
        self._pps = None
        # 已写入 ffmpeg 的帧序号（FIFO），用于把输出的 AU 和输入帧对上，
        # 让音频门控以「这帧视频真的发出去了」为准。
        self._inflight = collections.deque()
        self._inflight_lock = threading.Lock()
        self._buf = bytearray()
        self._au_nals = []
        self._au_has_vcl = False
        self._au_key = False
        self.dropped = 0
        # 对账计数（只统计，不影响逻辑）
        self.feed_pushed = 0
        self.feed_dropped = 0
        self.feeder_writes = 0
        self.feeder_dups = 0
        self.feeder_starved = 0
        self.aus = 0
        self.async_dropped = 0

    # ─── 生命周期 ────────────────────────────────────────────────────
    def start(self, loop) -> None:
        self._loop = loop
        self._async_q = asyncio.Queue(maxsize=60)

        args = [
            _ffmpeg_bin(), "-hide_banner", "-loglevel", "warning",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", "%dx%d" % (self.width, self.height),
            "-r", str(self.fps), "-i", "pipe:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-threads", str(max(1, int(os.environ.get("WS_VIDEO_X264_THREADS", "2")))),
            "-profile:v", "baseline", "-pix_fmt", "yuv420p",
            "-g", str(self._gop), "-keyint_min", str(self._gop), "-sc_threshold", "0",
            "-b:v", str(self.bitrate), "-maxrate", str(int(self.bitrate * 1.2)),
            "-bufsize", str(self.bitrate * 2),
            "-f", "h264", "pipe:1",
        ]
        self._proc = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            # 独立进程组：ffmpeg 有时会派生子进程，按组回收才不会有残留进程继续吃 CPU
            start_new_session=True,
        )
        for target, name in (
            (self._feeder_loop, "ws-h264-feed"),
            (self._reader_loop, "ws-h264-read"),
            (self._stderr_loop, "ws-h264-log"),
        ):
            t = threading.Thread(target=target, name=name, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._stop.set()
        self._terminate_proc()
        for t in self._threads:
            # 不能 join 自己（_reader_loop 里发现 ffmpeg 退出时可能回调到 stop）
            if t is threading.current_thread():
                continue
            if t.name == "ws-h264-read" and t.is_alive():
                # reader 卡在阻塞 read 上时关掉 stdout 让它退出
                try:
                    if self._proc is not None and self._proc.stdout:
                        self._proc.stdout.close()
                except Exception:
                    pass
            t.join(timeout=2)

    def _terminate_proc(self) -> None:
        """无论如何都要收掉 ffmpeg 进程（含其进程组），防止断连后残留抢 CPU。"""
        proc = self._proc
        if proc is None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is not None:
            return
        pid = proc.pid
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
        try:
            proc.wait(timeout=3)
            return
        except Exception:
            pass
        logger.warning("[ws-h264] ffmpeg 未在 3s 内退出，强制 kill pgid=%s", pid)
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)

    # ─── 喂帧 ────────────────────────────────────────────────────────
    def push_frame(self, frame, seq: int = -1) -> None:
        """由 WS 事件循环调用（只入队，不阻塞）。"""
        item = (frame, seq)
        self.feed_pushed += 1
        try:
            self._feed_q.put_nowait(item)
        except queue.Full:
            self.feed_dropped += 1
            try:
                self._feed_q.get_nowait()
            except queue.Empty:
                pass
            try:
                self._feed_q.put_nowait(item)
            except queue.Full:
                pass

    def _feeder_loop(self) -> None:
        ptime = 1.0 / self.fps
        start = time.monotonic()
        n = 0
        last = None
        last_seq = -1
        while not self._stop.is_set():
            target = start + n * ptime
            now = time.monotonic()
            if now < target:
                self._stop.wait(target - now)
                continue
            try:
                frame, seq = self._feed_q.get_nowait()
                last, last_seq = frame, seq
            except queue.Empty:
                frame, seq = last, last_seq
                if frame is not None:
                    self.feeder_dups += 1
                else:
                    self.feeder_starved += 1
            if frame is not None:
                try:
                    bgr = frame.to_ndarray(format="bgr24")
                    if bgr.shape[1] != self.width or bgr.shape[0] != self.height:
                        bgr = cv2.resize(
                            bgr, (self.width, self.height), interpolation=cv2.INTER_AREA
                        )
                    self._proc.stdin.write(bgr.tobytes())
                    self.feeder_writes += 1
                    with self._inflight_lock:
                        self._inflight.append(seq)
                except (BrokenPipeError, ValueError, OSError):
                    break
            n += 1
            if now - target > 1.0:  # 被调度卡住超过 1s：重新对齐
                start = now
                n = 0

    # ─── Annex-B → AVCC ─────────────────────────────────────────────
    def _reader_loop(self) -> None:
        stream = self._proc.stdout
        while not self._stop.is_set():
            try:
                chunk = stream.read(8192)
            except Exception:
                break
            if not chunk:
                break
            self._buf.extend(chunk)
            self._parse()
        self._flush_au()
        if not self._stop.is_set():
            logger.warning("[ws-h264] ffmpeg 输出结束，通知 WS 重连")
            # 自己收掉 ffmpeg，避免 on_exit 里停不下来留下僵尸进程抢 CPU
            self._stop.set()
            self._terminate_proc()
            if self.on_exit:
                try:
                    self.on_exit()
                except Exception:
                    pass

    def _stderr_loop(self) -> None:
        try:
            for line in iter(self._proc.stderr.readline, b""):
                text = line.decode("utf-8", "replace").strip()
                if text:
                    logger.warning("[ws-h264][ffmpeg] %s", text)
        except Exception:
            pass

    def _parse(self) -> None:
        data = bytes(self._buf)
        parts = data.split(b"\x00\x00\x01")
        if len(parts) <= 1:
            return
        # 最后一段可能不完整：保留（补回起始码），等下次读取
        self._buf = bytearray(b"\x00\x00\x01" + parts[-1])
        for part in parts[1:-1]:
            nal = part.lstrip(b"\x00")
            if nal:
                self._on_nal(nal)

    def _on_nal(self, nal: bytes) -> None:
        nal_type = nal[0] & 0x1F
        if nal_type == 7:  # SPS
            self._sps = nal
            self._try_config()
            return
        if nal_type == 8:  # PPS
            self._pps = nal
            self._try_config()
            return
        if nal_type in (1, 5):  # VCL：切片
            first_mb = _first_mb_in_slice(nal)
            if self._au_has_vcl and first_mb == 0:
                self._flush_au()
            self._au_has_vcl = True
            if nal_type == 5:
                self._au_key = True
        self._au_nals.append(nal)

    def _try_config(self) -> None:
        if self.avcc or not (self._sps and self._pps):
            return
        sps, pps = self._sps, self._pps
        if len(sps) < 4 or len(pps) < 1:
            return
        self.avcc = (
            bytes([1, sps[1], sps[2], sps[3], 0xFF, 0xE1])
            + len(sps).to_bytes(2, "big") + sps
            + b"\x01" + len(pps).to_bytes(2, "big") + pps
        )
        self.codec_string = "avc1." + bytes([sps[1], sps[2], sps[3]]).hex().upper()
        self._ready.set()

    def _flush_au(self) -> None:
        nals = self._au_nals
        self._au_nals = []
        self._au_has_vcl = False
        key = self._au_key
        self._au_key = False
        if not nals or not self.avcc:
            return
        # SPS/PPS 已在 avcC 里；AU 只保留 VCL/SEI/AUD 等
        kept = [n for n in nals if (n[0] & 0x1F) not in (7, 8)]
        if not kept:
            return
        # 只有真正要交付的 AU 才消费一个 inflight 序号
        with self._inflight_lock:
            seq = self._inflight.popleft() if self._inflight else -1
        avcc = b"".join(len(n).to_bytes(4, "big") + n for n in kept)
        self.aus += 1
        self._deliver(key, avcc, seq)

    def _deliver(self, key: bool, avcc: bytes, seq: int = -1) -> None:
        def _put():
            q = self._async_q
            if q is None:
                return
            if q.full():
                # 队列满说明 WS 发送侧被客户端 TCP 窗口卡住（主线程忙/网络抖动）。
                # 丢最旧的 AU 会让解码器失去参考帧，尽量给足缓冲（60 帧 ≈ 2.4s），
                # 真到这一步再丢，并计数用于排查。
                try:
                    q.get_nowait()
                except Exception:
                    pass
                self.dropped += 1
                self.async_dropped += 1
                if self.dropped % 25 == 0:
                    logger.warning("[ws-h264] 发送队列积压，已丢弃 %d 帧", self.dropped)
            q.put_nowait((key, avcc, seq))

        try:
            self._loop.call_soon_threadsafe(_put)
        except Exception:
            pass

    def stats(self) -> dict:
        return {
            'feed_pushed': self.feed_pushed,
            'feed_dropped': self.feed_dropped,
            'feeder_writes': self.feeder_writes,
            'feeder_dups': self.feeder_dups,
            'feeder_starved': self.feeder_starved,
            'aus': self.aus,
            'async_dropped': self.async_dropped,
        }

    # ─── 给 WS 协程使用 ─────────────────────────────────────────────
    async def next_au(self):
        return await self._async_q.get()

    def ready_payload(self) -> dict:
        return {
            "codec": "h264",
            "codecString": self.codec_string,
            "avcC": base64.b64encode(self.avcc).decode("ascii"),
        }
