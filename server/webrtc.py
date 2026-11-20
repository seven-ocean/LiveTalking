###############################################################################
#  Copyright (C) 2025 四川昱扬科技有限公司
#
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import asyncio
import collections
import json
import logging
import os  # [fix-publish-webrtc]:e0bc212ab7
import threading
import time
from typing import Tuple, Dict, Optional, Set, Union
import queue
from av.frame import Frame
from av.packet import Packet
from av import AudioFrame
import fractions
import numpy as np

AUDIO_PTIME = 0.020  # 20ms audio packetization
VIDEO_CLOCK_RATE = 90000
VIDEO_PTIME = 0.040 #1 / 25  # 30fps
VIDEO_TIME_BASE = fractions.Fraction(1, VIDEO_CLOCK_RATE)
# [fix-publish-webrtc]:76e1155fe5 性能优先：WebRTC 视频轨默认 0.75 下采样后再编码（与 WS 通道一致）
_WEBRTC_SCALE = max(0.2, min(1.0, float(os.environ.get("ZL_WEBRTC_SCALE", "0.75"))))
SAMPLE_RATE = 16000
# 播放采样率统一到 48k：WebRTC/Opus 的原生时钟就是 48k，多数声卡也是 48k。
# 这样播放链路在服务端和浏览器两端都**不需要重采样**（44.1k 时 44.1↔48 每边都要转一次）。
# 口型/ASR 仍用 16k，只在服务端转一次，不在播放路径上。
DEFAULT_PLAYBACK_RATE = int(os.environ.get("TALK_PLAYBACK_SR", "48000") or 48000)
# 兼容旧引用；真正的口径按会话取（HumanPlayer.audio_rate）
PLAYBACK_RATE = DEFAULT_PLAYBACK_RATE
AUDIO_TIME_BASE = fractions.Fraction(1, DEFAULT_PLAYBACK_RATE)

#from aiortc.contrib.media import MediaPlayer, MediaRelay
#from aiortc.rtcrtpsender import RTCRtpSender
from aiortc import (
    MediaStreamTrack,
)

logging.basicConfig()
logger = logging.getLogger(__name__)
from utils.logger import logger as mylogger

# WS 降级通道的「消费者占位符」：注册进 HumanPlayer.__started，
# 使渲染线程在 WS 连接期间存活（否则 pc 因 ICE 失败清理时会把渲染线程停掉 → 黑屏）
_WS_CONSUMER = object()


class PlayerStreamTrack(MediaStreamTrack):
    """
    A video track that returns an animated flag.
    """

    def __init__(self, player, kind):
        super().__init__()  # don't forget this!
        self.kind = kind
        self._player = player
        # 视频帧是 720x1280x3 ≈ 2.7MB，队列上限直接决定每会话的内存占用：
        # 原为 100 帧 ≈ 270MB/会话，且 ICE 连不上时这条队列**没人消费、长期占满**
        # （叠加 WS 通道的队列后约 350MB/会话，两路就把 64GB 挤爆 → 进程被 OOM 杀）。
        # 但也不能太小：渲染是**每轮一次性推 batch_size(16) 帧**的突发，
        # 上限必须 ≥ 突发长度，否则每轮被截断（实测 10 帧上限把 25fps 打成 18fps）。
        # 24 帧 = 一轮突发 + 余量 + 约 1s 缓冲 ≈ 65MB/会话；
        # 提到 32 帧（≈86MB/会话）容纳两轮突发，降低网络抖动时的丢帧。
        try:
            _pm = str(getattr(player, 'pacing_mode', '') or '')
        except Exception:
            _pm = ''
        # 默认（legacy/official）保持原值 24；sync25 用 20（同频后无需更大）。
        if kind == 'video':
            _vmax = 20 if _pm == 'sync25' else 24
        else:
            _vmax = 30 if _pm == 'sync25' else 100
        self._queue = queue.Queue(maxsize=_vmax)
        self.timelist = [] #记录最近包的时间戳
        self.current_frame_count = 0
        # 音频断供诊断：补静音次数 / 队列等待次数（现场排查电音/断续用）
        self._silence_fill = 0
        self._audio_underrun = 0
        # [诊断] 音频发送侧：队列等待 / 帧间隔（区分"生产不足"与"发送侧被卡"）
        self._stat_qwait_n = 0
        self._stat_qwait_sum = 0.0
        self._stat_qwait_max = 0.0
        self._stat_gap_n = 0
        self._stat_gap_sum = 0.0
        self._stat_gap_max = 0.0
        self._stat_last_recv = 0.0
        if self.kind == 'video':
            self.framecount = 0
            self.lasttime = time.perf_counter()
            self.totaltime = 0
    
    _start: float
    _timestamp: int

    async def next_timestamp(self) -> Tuple[int, fractions.Fraction]:
        if self.readyState != "live":
            raise Exception

        if self.kind == 'video':
            # 时间戳按帧序号推进（90000/fps），节拍只负责限速。
            # 原来用 (now-start)*90000 且「落后就重置」，时间轴会跳变 → 接收端跳帧/卡顿。
            fps = max(1, int(getattr(self._player, 'video_fps', 25) or 25))
            step = int(round(VIDEO_CLOCK_RATE / fps))
            if not hasattr(self, '_video_ts'):
                self._video_ts = 0
                self._start = time.time()
                self.current_frame_count = 0
                self.timelist.append(self._start)
                mylogger.info('video start:%f fps=%d', self._start, fps)
            self.current_frame_count += 1
            wait = self._start + self.current_frame_count / fps - time.time()
            if wait > 0:
                await asyncio.sleep(wait)
            elif wait < -0.2:
                # 落后超过 200ms：只重置节拍基准，不动时间戳
                self._start = time.time() - self.current_frame_count / fps
            pts = self._video_ts
            self._video_ts += step
            # [诊断] 视频出帧间隔：判断发送端是匀速还是成串（成串 = 接收端 jitter 持续上涨）
            try:
                _now = time.monotonic()
                _prev = getattr(self, '_last_send_ts', 0.0)
                if _prev:
                    _g = (_now - _prev) * 1000.0
                    _p = self._player
                    _p._v_gap_n = getattr(_p, '_v_gap_n', 0) + 1
                    _p._v_gap_sum = getattr(_p, '_v_gap_sum', 0.0) + _g
                    if _g > getattr(_p, '_v_gap_max', 0.0):
                        _p._v_gap_max = _g
                self._last_send_ts = _now
            except Exception:
                pass
            return pts, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):                #self._timestamp = (time.time()-self._start) * SAMPLE_RATE
                now = time.time()
                self._timestamp = int((now - self._start) * self._player.audio_rate)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * AUDIO_PTIME - now
                # wait = self.timelist[0] + len(self.timelist)*AUDIO_PTIME - time.time()
                if wait > 0:
                    await asyncio.sleep(wait)
                elif wait < -0.08:
                    # 落后超过 80ms（生产端抖动/补静音后）：重置节拍基准。
                    # 否则会连续无间隔补发一串 RTP，接收端 jitter 暴涨 → 电音/断续。
                    self._start = now - self.current_frame_count * AUDIO_PTIME
                    self._timestamp = int((now - self._start) * self._player.audio_rate)
                # if len(self.timelist)>=200:
                #     self.timelist.pop(0)
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('audio start:%f',self._start)
            return self._timestamp, fractions.Fraction(1, self._player.audio_rate)

    async def _pace_audio(self) -> None:
        """按 20ms 节拍放行音频帧（只控速，不影响 RTP 时间戳）。"""
        now = time.time()
        if not hasattr(self, '_pace_start'):
            self._pace_start = now
            self._pace_count = 0
        self._pace_count += 1
        target = self._pace_start + self._pace_count * AUDIO_PTIME
        wait = target - now
        if wait > 0:
            await asyncio.sleep(wait)
        elif wait < -0.08:
            # 落后太多只重置节拍基准；时间戳仍按采样数连续推进
            self._pace_start = now - self._pace_count * AUDIO_PTIME

    async def recv(self) -> Union[Frame, Packet]:
        self._player._start(self)
        if self.kind == 'video' and not getattr(self, "_audio_gate_done", False):
            # 唤醒后第一次出声：先等音频轨起播再出视频帧，最多 1.5s
            self._audio_gate_done = True
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, self._player._audio_started.wait, 1.5
                )
            except Exception:
                pass

        frame = None
        eventpoint = None
        seq = -1
        # [诊断] 音频：qWait=等生产的时间；gap=相邻两帧 recv 的间隔（>20ms 说明发送侧被卡）
        _t_wait0 = time.monotonic()
        _silence_samples = max(1, int(self._player.audio_rate * AUDIO_PTIME))
        _starve_deadline = None
        while True:
            while True:
                try:
                    item = self._queue.get_nowait()
                    break
                except queue.Empty:
                    if self.kind == 'video' and getattr(self._player, 'frame_pace', False):
                        last = getattr(self, '_last_video_frame', None)
                        if last is not None:
                            self._player.dup += 1
                            item = (last, None, -1)
                            break
                    if self.kind == 'audio':
                        now = time.monotonic()
                        if _starve_deadline is None:
                            # 放宽到 100ms：生产端抖 20~40ms 时插一段零会在词中间造成
                            # 一次波形硬切（听感就是"电流音/啪"）。宁可让接收端用
                            # NetEq 短暂拉伸，也不要在字中间剪一刀。
                            # 音频固定 100ms 超时才补静音：0ms 会在词中间插零
                            #（实测会明显加重断字/电流音），所以补齐节拍只作用于视频。
                            _starve_deadline = now + 0.100
                            self._audio_underrun += 1   # 断供“事件”计数（不是空轮询次数）
                        elif now >= _starve_deadline:
                            # 生产端暂时断供（推理抖动/换句间隙）：补 20ms 静音，别让音频轨停住。
                            # 停发会让接收端进入丢包隐藏（NetEq），听感就是电音/断续。
                            from av import AudioFrame as _SilentAudioFrame
                            sil = _SilentAudioFrame(
                                format='s16', layout='mono', samples=_silence_samples)
                            sil.sample_rate = self._player.audio_rate
                            sil.planes[0].update(
                                np.zeros(_silence_samples, dtype=np.int16).tobytes())
                            self._silence_fill += 1
                            item = (sil, None, -1)
                            break
                    await asyncio.sleep(0.005)
            if isinstance(item, tuple):
                if len(item) >= 3:
                    frame, eventpoint, seq = item
                else:
                    frame, eventpoint = item
                    seq = -1
            else:
                frame, eventpoint, seq = item, None, -1
            if frame is None:
                self.stop()
                raise Exception
            # 音频不等待视频。实测这段等待会让音频轨每秒少消费约 2 帧
            # （video_sent_seq 落后于音频 seq，每帧白等最多 20ms），
            # 讲话时间一长音频队列 aq 持续上涨：声音越来越滞后，满了就吞字。
            # 口型短暂领先远好于音频延迟/丢帧。
            break

        if self.kind == 'video':
            self._player._webrtc_video_pull_ts = time.monotonic()
            _scale = getattr(self._player, "webrtc_scale", _WEBRTC_SCALE)
            if _scale < 0.999:
                try:
                    # [fix-publish-webrtc]:711b4f0f05 下采样后再交给 libx264，降低 CPU/带宽
                    nw = max(2, int(round(frame.width * _scale / 2.0)) * 2)
                    nh = max(2, int(round(frame.height * _scale / 2.0)) * 2)
                    frame = frame.reformat(width=nw, height=nh)
                except Exception:
                    pass
        if self.kind == 'audio':
            # ── 音频时间戳必须按「采样数」单调推进，不能跟墙钟走 ──────────
            # 之前用 (now - start) * rate 当 pts：只要这一帧晚了 80ms，就把 start
            # 重置一次，时间戳随之跳变。接收端 NetEq 会看到一个个空洞，于是拼命
            # conceal（实测 17%~50%）——听感就是电流音/断字。
            # 现在每一帧只按自己的 samples 前进；节拍只用来限速，绝不动时间戳。
            rate = int(getattr(self._player, 'audio_rate', 0) or PLAYBACK_RATE)
            if not hasattr(self, '_audio_pts'):
                self._audio_pts = 0
                self._audio_ts_rate = rate
            elif rate != getattr(self, '_audio_ts_rate', rate):
                self._audio_pts = int(self._audio_pts * rate / self._audio_ts_rate)
                self._audio_ts_rate = rate
            pts = self._audio_pts
            time_base = fractions.Fraction(1, rate)
            self._audio_pts += int(getattr(frame, 'samples', 0) or 0)
            # [诊断] 队列等待时间
            try:
                _qw = max(0.0, time.monotonic() - _t_wait0)
                self._stat_qwait_n += 1
                self._stat_qwait_sum += _qw
                if _qw > self._stat_qwait_max:
                    self._stat_qwait_max = _qw
            except Exception:
                pass
            await self._pace_audio()
            self._player._audio_started.set()
            # RTP 音频帧计数：和浏览器 audPkts 对照，判断是服务端供给不足还是网络/解码
            self._player.audio_rtp_sent = getattr(self._player, 'audio_rtp_sent', 0) + 1
        else:
            pts, time_base = await self.next_timestamp()
        # 记住最后一帧：?frame_pace=fill 时断供要复用同一帧（时间戳照常前进）
        if self.kind == 'video':
            self._last_video_frame = frame
        frame.pts = pts
        frame.time_base = time_base
        if eventpoint and self._player is not None:
            self._player.notify(eventpoint)
        if self.kind == 'video' and seq >= 0:
            if seq > self._player.video_sent_seq:
                self._player.video_sent_seq = seq
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount == 100:
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime = 0
        if self.kind == 'audio':
            try:
                now_ts = time.monotonic()
                last = getattr(self, '_stat_last_recv', 0.0)
                if last > 0:
                    gap = now_ts - last
                    self._stat_gap_n += 1
                    self._stat_gap_sum += gap
                    if gap > self._stat_gap_max:
                        self._stat_gap_max = gap
                self._stat_last_recv = now_ts
            except Exception:
                pass
        return frame

    def stop(self):
        super().stop()
        # Drain & delete remaining frames
        while not self._queue.empty():
            item = self._queue.get_nowait()
            del item
        if self._player is not None:
            self._player._stop(self)
            self._player = None

def player_worker_thread(
    quit_event,
    container
):
    container.render(quit_event)


class _FifoSlot:
    """有序 FIFO 槽位（线程安全），供 WS 降级通道独占一份帧。

    为什么必须是队列而不是「最新帧」：`process_frames` 是以**推理批为粒度突发产出**
    的（实测每约 600ms 一次性推 16 帧）。若只保留最新帧，每个突发只会送出 1 帧、
    丢掉 15 帧（实测掉到 1.6-2.7fps）。队列把这 16 帧摊平，正好平滑成 25fps
    —— WebRTC 那条路之所以一直正常，就是因为它有一条队列。

    视频：超过上限丢最旧（慢客户端时等效丢帧，不无上限增长）；
    音频：不丢帧优先，上限设得更大以防突发。
    """

    def __init__(self, maxsize: int = 200, drop_oldest: bool = True):
        self._cv = threading.Condition()
        self._buf = collections.deque()
        self._maxsize = maxsize
        self._drop_oldest = drop_oldest
        self.puts = 0
        self.drops = 0
        self.clears = 0

    def put(self, item) -> None:
        with self._cv:
            self.puts += 1
            if len(self._buf) >= self._maxsize:
                self._buf.popleft()
                self.drops += 1
            self._buf.append(item)
            self._cv.notify_all()

    def pop(self):
        with self._cv:
            if not self._buf:
                return None
            return self._buf.popleft()

    def clear(self) -> int:
        with self._cv:
            n = len(self._buf)
            self._buf.clear()
            self.clears += 1
            return n

    def stats(self) -> dict:
        with self._cv:
            return {'puts': self.puts, 'drops': self.drops,
                    'clears': self.clears, 'qsize': len(self._buf)}

    def qsize(self) -> int:
        with self._cv:
            return len(self._buf)


class HumanPlayer:
    def __init__(
        self, avatar_session, format=None, options=None, timeout=None, loop=False, decode=True
    ):
        self.__thread: Optional[threading.Thread] = None
        self.__thread_quit: Optional[threading.Event] = None

        # examine streams
        self.__started: Set[PlayerStreamTrack] = set()
        self.__audio: Optional[PlayerStreamTrack] = None
        self.__video: Optional[PlayerStreamTrack] = None

        # 产出节流模式（URL ?pacing=legacy|official|bounded|sync25）。
        # 必须在建轨之前确定：视频/音频队列容量由它决定。
        try:
            self.pacing_mode = str(getattr(getattr(avatar_session, 'opt', None),
                                           'pacing', 'legacy') or 'legacy').lower()
        except Exception:
            self.pacing_mode = 'legacy'

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")

        # WS 降级通道专用帧源（与 aiortc 那条路完全独立，互不抢帧）
        # 视频队列：必须 ≥ 一轮突发(16 帧)；32 帧可容纳两轮突发，降低网络抖动时的丢帧。
        # 32 帧 ≈ 86MB/会话；满了丢最旧，不会阻塞生产。
        self.__ws_video = _FifoSlot(maxsize=32)
        self.__ws_audio = _FifoSlot(maxsize=500)
        # 音频轨是否已起播：视频轨首帧要等它，保证唤醒时“先有声、再有口型”
        self._audio_started = threading.Event()
        # ── A/V 成对序号与门控 ─────────────────────────────────────────
        # process_frames 按「1 视频 + 2 音频(40ms)」成对产出；这里给每对分配 seq，
        # 音频只在对应视频已经发出后才允许发送，视频慢时音频跟着停/丢，避免声音越跑越前。
        self.av_seq = 0
        self.produced_video = 0
        self.produced_audio = 0
        self._last_video_seq = -1
        self.video_sent_seq = -1          # WebRTC 视频轨已发出的最大 seq
        self.ws_video_sent_seq = -1       # WS 降级通道已发出的最大 seq
        self.flv_video_sent_seq = -1      # FLV 测试通道（默认关闭）
        self._dropped_seqs = collections.deque(maxlen=64)
        self._av_cond = threading.Condition()
        self._webrtc_video_active = False
        # WebRTC 视频轨最近一次真正取帧的时间。判断「有没有人消费」不能只看
        # _webrtc_video_active：ICE 失败/页面切到 WS 后这个标志可能还挂着 True，
        # 那条没人消费的 RTP 队列会一直丢帧并记账，导致对应音频被误丢。
        self._webrtc_video_pull_ts = 0.0
        self.ws_active = False
        self.flv_active = False
        # [perf] 日志用的上一采样点（产出 fps）
        self._perf_last_ts = 0.0
        self._perf_last_v = 0
        # 产出节流模式（URL ?pacing=legacy|official|bounded），已在上面按队列容量确定
        self.dup = 0
        # 补齐节拍（URL ?frame_pace=fill）：缺帧时重复上一帧 / 立即补静音，
        # 让「媒体时间」恒等于「真实时间」，避免接收端抖动缓冲每秒涨 8ms。
        self.frame_pace = bool(getattr(getattr(avatar_session, 'opt', None), 'frame_pace', False))
        self.dup = 0
        # 本会话视频帧率（画质档位：标准 20fps，其余 25fps）
        try:
            self.video_fps = int(getattr(getattr(avatar_session, 'opt', None), 'video_fps', 25) or 25)
        except Exception:
            self.video_fps = 25
        # 本会话的播放口径（auto 时统一 48k；能力中心选了 44.1k/16k 就按配置）
        try:
            self.audio_rate = int(getattr(avatar_session, 'playback_rate', 0)
                                  or DEFAULT_PLAYBACK_RATE)
        except (TypeError, ValueError):
            self.audio_rate = DEFAULT_PLAYBACK_RATE

        self.__container = avatar_session
        if hasattr(self.__container, 'output'):
            self.__container.output._player = self
        # 每会话视频缩放：低配客户端可在 offer 里带 low_perf/scale，降低解码/渲染压力，
        # 把 CPU 留给音频，避免“有画面时电音”。
        try:
            self.webrtc_scale = float(getattr(
                getattr(avatar_session, "opt", None), "webrtc_scale", _WEBRTC_SCALE
            ) or _WEBRTC_SCALE)
        except (TypeError, ValueError):
            self.webrtc_scale = _WEBRTC_SCALE
        self.audio_sent = 0
        self.audio_drop = 0
        self.audio_rtp_sent = 0
        # [诊断] 音频队列峰值、批次突发粒度
        self.aq_max = 0
        self._burst_frames = 0
        self._burst_last_ts = 0.0
        self._burst_n = 0
        self._burst_frames_sum = 0
        self._burst_interval_sum = 0.0
        self._burst_max = 0

    def push_video(self, frame):
        from av import VideoFrame
        self.produced_video += 1
        # [诊断] 推理是"按批突发"产出：统计每批帧数与批次间隔，判断 12 帧队列是否够用
        try:
            _now = time.monotonic()
            if self._burst_last_ts and (_now - self._burst_last_ts) > 0.05:
                self._burst_n += 1
                self._burst_frames_sum += self._burst_frames
                self._burst_interval_sum += (_now - self._burst_last_ts)
                if self._burst_frames > self._burst_max:
                    self._burst_max = self._burst_frames
                self._burst_frames = 0
            self._burst_frames += 1
            self._burst_last_ts = _now
        except Exception:
            pass
        if self.produced_video % 125 == 0:
            # 5s 一条（25fps）：产出/发送/积压/丢弃，用于区分服务端编码慢还是客户端慢
            try:
                now = time.monotonic()
                fps = 0.0
                if self._perf_last_ts > 0:
                    dt = max(1e-3, now - self._perf_last_ts)
                    fps = (self.produced_video - self._perf_last_v) / dt
                self._perf_last_ts = now
                self._perf_last_v = self.produced_video
                mylogger.info(
                    "[perf] session=%s fps=%.1f produced_v=%d produced_a=%d sent_seq=%d "
                    "vq=%d aq=%d wsq=%d dropped=%d audio_sent=%d audio_drop=%d "
                    "asent=%d silence_fill=%d audio_underrun=%d dup=%d scale=%.2f vbps=%.0f vtb=%d "
                    "aqMax=%d qWait=%.1f/%.1fms gap=%.1f/%.1fms burst=%.1f/%.0f "
                    "vgap=%.1f/%.1fms enc=%.1f/%.1fms",
                    getattr(self.__container, "sessionid", "-"), fps,
                    self.produced_video, self.produced_audio, self.video_sent_seq,
                    self.__video._queue.qsize(), self.__audio._queue.qsize(),
                    self.__ws_video.qsize(),
                    len(self._dropped_seqs), self.audio_sent, self.audio_drop,
                    self.audio_rtp_sent,
                    getattr(self.__audio, "_silence_fill", 0),
                    getattr(self.__audio, "_audio_underrun", 0),
                    self.dup,
                    self.webrtc_scale,
                    getattr(self, "vbps", 0.0) / 1000.0,
                    int(getattr(self, "vtb", 0) or 0),
                    int(getattr(self, "aq_max", 0)),
                    (getattr(self.__audio, "_stat_qwait_sum", 0.0) /
                     max(1, getattr(self.__audio, "_stat_qwait_n", 1)) * 1000.0),
                    getattr(self.__audio, "_stat_qwait_max", 0.0) * 1000.0,
                    (getattr(self.__audio, "_stat_gap_sum", 0.0) /
                     max(1, getattr(self.__audio, "_stat_gap_n", 1)) * 1000.0),
                    getattr(self.__audio, "_stat_gap_max", 0.0) * 1000.0,
                    (getattr(self, "_burst_frames_sum", 0) / max(1, getattr(self, "_burst_n", 1))),
                    getattr(self, "_burst_max", 0),
                    (getattr(self, "_v_gap_sum", 0.0) / max(1, getattr(self, "_v_gap_n", 1))),
                    getattr(self, "_v_gap_max", 0.0),
                    (getattr(self, "_enc_sum", 0.0) / max(1, getattr(self, "_enc_n", 1))),
                    getattr(self, "_enc_max", 0.0),
                )
                # 诊断计数清零（每 5s 一段）
                self._v_gap_n = 0
                self._v_gap_sum = 0.0
                self._v_gap_max = 0.0
                self._enc_n = 0
                self._enc_sum = 0.0
                self._enc_max = 0.0
                self.aq_max = 0
                self._burst_n = 0
                self._burst_frames_sum = 0
                self._burst_interval_sum = 0.0
                self._burst_max = 0
                for _t in (self.__audio,):
                    for _f in ("_stat_qwait_n", "_stat_qwait_sum", "_stat_qwait_max",
                               "_stat_gap_n", "_stat_gap_sum", "_stat_gap_max"):
                        setattr(_t, _f, 0.0 if _f.endswith("sum") or _f.endswith("max") else 0)
            except Exception:
                pass
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        with self._av_cond:
            seq = self.av_seq
            self.av_seq += 1
            self._last_video_seq = seq
            # 只有 WebRTC 视频轨**最近真的在取帧**时，丢弃才需要记账（用来丢对应音频）。
            # WS-only 场景下这条 RTP 队列没人取、必然常满；ICE 失败后标志位也可能还挂着，
            # 若只看标志位就会把 WS 通道的音频误丢一半（实测 30s 只收到 13.5s 音频）。
            webrtc_active = (
                self._webrtc_video_active
                and (time.monotonic() - self._webrtc_video_pull_ts) < 1.0
            )
        # ⚠ ICE 未连通时（真实环境里降级到 WS 就是这个状态）**没有任何消费者**会取这条队列，
        # 一旦写满（maxsize=100）`put` 就会阻塞 → process_frames/inference 全部被卡死
        # → 整条渲染管线停在几十帧后不再出画面（实测黑屏的根因）。
        # WS-only 场景直接不入这条队：省内存，也不会把“没人消费的丢帧”算成 audio_drop。
        if self._webrtc_video_active:
            try:
                self.__video._queue.put_nowait((new_frame, None, seq))
            except queue.Full:
                try:
                    old = self.__video._queue.get_nowait()
                    if webrtc_active and isinstance(old, tuple) and len(old) >= 3:
                        self._dropped_seqs.append(old[2])
                except queue.Empty:
                    pass
                try:
                    self.__video._queue.put_nowait((new_frame, None, seq))
                except queue.Full:
                    if webrtc_active:
                        self._dropped_seqs.append(seq)
        # 额外复制一份给 WS 降级通道（同一对象引用，开销可忽略）
        # 注意：不能按 ws_active 门控 —— WS handler 可能持有的是上一个 HumanPlayer
        # 对象（WebRTC 重建时会换 player），门控会导致 WS 客户端收不到帧。
        self.__ws_video.put((new_frame, seq))

    def push_audio(self, frame, eventpoint=None, sample_rate=None):
        from av import AudioFrame
        self.produced_audio += 1
        # 新一句话开始：只丢掉队列头部的「待机静音」，绝不整队清空。
        # 之前这里是清空整条队列，句子流水线下发时会把上一句还没播完的尾音一起删掉
        # ——现场听感就是句中截断（断流）+ 拼接处爆音。
        if eventpoint and eventpoint.get("status") == "start":
            self._audio_gen = getattr(self, "_audio_gen", 0) + 1
            self._drop_leading_silence()
            self._utt_frames = 0
            self._utt_t0 = time.time()
        # 每句实际入队了多少帧（20ms/帧）：和 TTS 文本时长对照，能直接看出"吞字"发生在
        # 服务端入队前（TTS 只合成了一半）还是入队后（消费/网络丢）。
        self._utt_frames = getattr(self, "_utt_frames", 0) + 1
        self._evt_log = getattr(self, '_evt_log', 0) + 1
        if self._evt_log % 200 == 1:
            mylogger.info('[audio] eventpoint=%r samples=%d n=%d',
                          eventpoint, int(getattr(frame, 'shape', [0])[0]), self._evt_log)
        if eventpoint and eventpoint.get("status") == "end":
            mylogger.info("[audio] utterance frames=%d dur=%.2fs wall=%.2fs drop=%d",
                        self._utt_frames, self._utt_frames * 0.02,
                        time.time() - getattr(self, "_utt_t0", time.time()),
                        self.audio_drop)
        new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
        new_frame.planes[0].update(frame.tobytes())
        new_frame.sample_rate = int(sample_rate or self.audio_rate)

        with self._av_cond:
            seq = self._last_video_seq
        # 不再因为「对应视频帧被丢」就丢音频：20ms PCM 的缺口在弱客户端上会变成电音/爆音。
        # 音频连续优先，口型短暂领先由客户端视频丢帧追赶。
        # 同 push_video：ICE 未连通时这条队列也没有消费者，绝不能阻塞（否则 process_frames 被卡死）；
        # WS-only 场景直接不入这条队。
        if self._webrtc_video_active:
            try:
                if getattr(self, 'pacing_mode', 'legacy') in ('bounded', 'sync25'):
                    # 有界队列 + 阻塞背压：宁可让生产端等，也不要在音频时间轴上留洞
                    #（sync25 首句 TTS 会一次性灌入数秒音频，非阻塞写法会把**开头**丢掉）
                    self.__audio._queue.put((new_frame, eventpoint, seq), timeout=0.05)
                else:
                    self.__audio._queue.put_nowait((new_frame, eventpoint, seq))
            except queue.Full:
                self.audio_drop += 1
                try:
                    self.__audio._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.__audio._queue.put_nowait((new_frame, eventpoint, seq))
                except queue.Full:
                    pass
        # 同样不能因为「视频已经发出去了」就丢音频：那是音频偏晚（安全方向），
        # 应该照常入队，由 WS 音频泵按 seq 做门控。
        self.__ws_audio.put((new_frame, seq))
        self.audio_sent += 1
        try:
            qs = self.__audio._queue.qsize()
            if qs > self.aq_max:
                self.aq_max = qs
        except Exception:
            pass

    def ws_video_slot(self) -> _FifoSlot:
        return self.__ws_video

    @staticmethod
    def _frame_is_silent(item) -> bool:
        """判断队列元素是否为全零静音帧（兼容 (frame, meta) 与 (frame, meta, seq)）。"""
        try:
            frame = item[0] if isinstance(item, tuple) else item
            return not np.any(frame.to_ndarray())
        except Exception:
            return False

    def _drop_leading_silence(self) -> None:
        """只清队列头部连续的全零静音，遇到有声帧立即停止。

        目的：TTS 队列里常积着几百毫秒待机静音，新句子开始时要甩掉；
        但**不能**连已有语音一起删（那是句中截断的根因）。
        """
        kept = []
        try:
            while True:
                item = self.__audio._queue.get_nowait()
                if not kept and self._frame_is_silent(item):
                    continue
                kept.append(item)
        except queue.Empty:
            pass
        for item in kept:
            try:
                self.__audio._queue.put_nowait(item)
            except queue.Full:
                break

        kept = []
        while True:
            item = self.__ws_audio.pop()
            if item is None:
                break
            if not kept and self._frame_is_silent(item):
                continue
            kept.append(item)
        for item in kept:
            self.__ws_audio.put(item)

    def ws_audio_slot(self) -> _FifoSlot:
        return self.__ws_audio

    def get_active_backlog(self) -> int:
        """当前真正在消费的输出的视频积压（用于 render 给音频生产做背压）。"""
        sizes = []
        if self._webrtc_video_active:
            sizes.append(self.__video._queue.qsize())
        if self.ws_active:
            sizes.append(self.__ws_video.qsize())
        if self.flv_active:
            sizes.append(self.__ws_video.qsize())
        return max(sizes) if sizes else 0

    def get_buffer_size(self) -> int:
        return self.get_active_backlog()

    def queue_depths(self) -> dict:
        """三个队列的当前深度（bounded/sync25 模式做背压用）。

        必须按「有没有消费者」过滤：WS 音频队列在任何会话里都会被写满，
        若把没人消费的 wsq(500) 也算进背压条件，生产端会永久停在等待上
        （实测 sync25 长文本 35 帧/96s 就是这个原因）。
        """
        out = {"vq": 0, "aq": 0, "wsq": 0}
        try:
            if self._webrtc_video_active:
                out["vq"] = self.__video._queue.qsize()
                out["aq"] = self.__audio._queue.qsize()
        except Exception:
            pass
        try:
            if self.ws_active:
                out["wsq"] = self.__ws_audio.qsize()
        except Exception:
            pass
        return out

    def stats(self) -> dict:
        """供 /metrics 使用的轻量计数（不加锁，允许瞬时不一致）。"""
        try:
            vq = self.__video._queue.qsize() if self.__video is not None else 0
            aq = self.__audio._queue.qsize() if self.__audio is not None else 0
        except Exception:
            vq = aq = 0
        try:
            wsq = self.__ws_video.qsize()
            wsa = self.__ws_audio.qsize()
        except Exception:
            wsq = wsa = 0
        return {
            "produced_video": int(self.produced_video),
            "produced_audio": int(self.produced_audio),
            "video_sent_seq": int(self.video_sent_seq),
            "ws_video_sent_seq": int(self.ws_video_sent_seq),
            "webrtc_video_queue": int(vq),
            "webrtc_audio_queue": int(aq),
            "ws_video_queue": int(wsq),
            "ws_audio_queue": int(wsa),
            "webrtc_video_active": bool(self._webrtc_video_active),
            "ws_active": bool(self.ws_active),
            "dropped_seqs": len(self._dropped_seqs),
            "audio_sent": int(self.audio_sent),
            "audio_drop": int(self.audio_drop),
            "webrtc_scale": float(self.webrtc_scale),
        }

    def notify(self,eventpoint):
        if self.__container is not None:
            self.__container.notify(eventpoint)

    @property
    def audio(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains audio.
        """
        return self.__audio

    @property
    def video(self) -> MediaStreamTrack:
        """
        A :class:`aiortc.MediaStreamTrack` instance if the file contains video.
        """
        return self.__video

    def _start(self, track: PlayerStreamTrack) -> None:
        self.__started.add(track)
        if getattr(track, 'kind', '') == 'video':
            self._webrtc_video_active = True
        if self.__thread is None:
            self.__log_debug("Starting worker thread")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    self.__container
                ),
            )
            self.__thread.start()

    def ensure_worker(self) -> None:
        """确保渲染线程已启动（WS 降级通道专用，幂等）。

        两个必须做的事：
        1. 真实环境里 ICE 常连不上 → WebRTC 发送端**从不取帧** → `_start` 永不被调用，
           渲染线程不会启动 → 一帧都产不出（黑屏 + 只能报占位尺寸）。所以这里主动拉起。
        2. 同时把 WS 通道注册成**消费者占位符**：否则 pc 因 ICE 失败清理时，
           `_stop` 看到「没有任何消费者」会把渲染线程停掉 —— 实测连接后约 2s 就断流。
        """
        self.__started.add(_WS_CONSUMER)
        if self.__thread is None:
            self.__log_debug("Starting worker thread (ensure_worker)")
            self.__thread_quit = threading.Event()
            self.__thread = threading.Thread(
                name="media-player",
                target=player_worker_thread,
                args=(
                    self.__thread_quit,
                    self.__container
                ),
            )
            self.__thread.start()

    def release_worker(self) -> None:
        """WS 断连后调用：注销占位消费者，没有其它消费者时停掉渲染线程。"""
        self._stop(_WS_CONSUMER)

    def _stop(self, track: PlayerStreamTrack) -> None:
        self.__started.discard(track)
        if not any(getattr(t, 'kind', '') == 'video' for t in self.__started):
            self._webrtc_video_active = False

        if not self.__started and self.__thread is not None:
            self.__log_debug("Stopping worker thread")
            self.__thread_quit.set()
            # 捕获线程引用，避免 _finalize_cleanup 中 self.__thread 被提前置空后无法 join
            worker_thread = self.__thread
            self.__thread = None

            # 先在当前线程尝试快速 join（最多 200ms），若失败则放到守护线程异步清理，
            # 避免阻塞 asyncio 事件循环（PlayerStreamTrack.stop 可能在事件循环线程被调用）。
            worker_thread.join(timeout=0.2)
            if worker_thread.is_alive():
                mylogger.warning(
                    "HumanPlayer worker thread did not stop within 200ms, continuing cleanup async"
                )
                cleanup_thread = threading.Thread(
                    target=self._finalize_cleanup,
                    args=(worker_thread, self.__thread_quit),
                    daemon=True,
                )
                cleanup_thread.start()

        # 注意：这里**不能**把 __container 置空。WS 降级通道断开后会话仍在
        # （lease + 5 分钟热会话），下一次连接会复用同一个 HumanPlayer；置空后
        # ensure_worker 重新拉起的渲染线程拿到 None，表现为“重连后一直黑屏/
        # H.264 起不来只能回落 JPEG”。容器引用由 session_manager 回收时一并释放。

    def _finalize_cleanup(self, worker_thread: threading.Thread, quit_event: threading.Event) -> None:
        """在守护线程中完成 HumanPlayer worker thread 的收尾清理。"""
        if worker_thread is not None and worker_thread.is_alive():
            worker_thread.join(timeout=10.0)
            if worker_thread.is_alive():
                mylogger.warning(
                    "HumanPlayer worker thread still alive after 10s async cleanup"
                )
        if quit_event is not None:
            # 确保 quit_event 保持 set 状态，便于 render 循环识别
            quit_event.set()

    def __log_debug(self, msg: str, *args) -> None:
        mylogger.debug(f"HumanPlayer {msg}", *args)
