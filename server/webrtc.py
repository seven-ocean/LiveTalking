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
SAMPLE_RATE = 16000
AUDIO_TIME_BASE = fractions.Fraction(1, SAMPLE_RATE)

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
        # 24 帧 = 一轮突发 + 余量 + 约 1s 缓冲 ≈ 65MB/会话。
        self._queue = queue.Queue(maxsize=24 if kind == 'video' else 100)
        self.timelist = [] #记录最近包的时间戳
        self.current_frame_count = 0
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
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * VIDEO_CLOCK_RATE
                self._timestamp += int(VIDEO_PTIME * VIDEO_CLOCK_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * VIDEO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*VIDEO_PTIME - time.time()               
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=100:
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('video start:%f',self._start)
            return self._timestamp, VIDEO_TIME_BASE
        else: #audio
            if hasattr(self, "_timestamp"):
                #self._timestamp = (time.time()-self._start) * SAMPLE_RATE
                self._timestamp += int(AUDIO_PTIME * SAMPLE_RATE)
                self.current_frame_count += 1
                wait = self._start + self.current_frame_count * AUDIO_PTIME - time.time()
                # wait = self.timelist[0] + len(self.timelist)*AUDIO_PTIME - time.time()
                if wait>0:
                    await asyncio.sleep(wait)
                # if len(self.timelist)>=200:
                #     self.timelist.pop(0)
                #     self.timelist.pop(0)
                # self.timelist.append(time.time())
            else:
                self._start = time.time()
                self._timestamp = 0
                self.timelist.append(self._start)
                mylogger.info('audio start:%f',self._start)
            return self._timestamp, AUDIO_TIME_BASE

    async def recv(self) -> Union[Frame, Packet]:
        # frame = self.frames[self.counter % 30]            
        self._player._start(self)
        # if self.kind == 'video':
        #     frame = await self._queue.get()
        # else: #audio
        #     if hasattr(self, "_timestamp"):
        #         wait = self._start + self._timestamp / SAMPLE_RATE + AUDIO_PTIME - time.time()
        #         if wait>0:
        #             await asyncio.sleep(wait)
        #         if self._queue.qsize()<1:
        #             #frame = AudioFrame(format='s16', layout='mono', samples=320)
        #             audio = np.zeros((1, 320), dtype=np.int16)
        #             frame = AudioFrame.from_ndarray(audio, layout='mono', format='s16')
        #             frame.sample_rate=16000
        #         else:
        #             frame = await self._queue.get()
        #     else:
        while True:
            try:
                frame, eventpoint = self._queue.get_nowait()
                break
            except queue.Empty:
                await asyncio.sleep(0.005)
                
        pts, time_base = await self.next_timestamp()
        frame.pts = pts
        frame.time_base = time_base
        if eventpoint and self._player is not None:
            self._player.notify(eventpoint)
        if frame is None:
            self.stop()
            raise Exception
        if self.kind == 'video':
            self.totaltime += (time.perf_counter() - self.lasttime)
            self.framecount += 1
            self.lasttime = time.perf_counter()
            if self.framecount==100:
                mylogger.info(f"------actual avg final fps:{self.framecount/self.totaltime:.4f}")
                self.framecount = 0
                self.totaltime=0
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

    def put(self, item) -> None:
        with self._cv:
            if len(self._buf) >= self._maxsize:
                self._buf.popleft()
            self._buf.append(item)
            self._cv.notify_all()

    def pop(self):
        with self._cv:
            if not self._buf:
                return None
            return self._buf.popleft()

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

        self.__audio = PlayerStreamTrack(self, kind="audio")
        self.__video = PlayerStreamTrack(self, kind="video")

        # WS 降级通道专用帧源（与 aiortc 那条路完全独立，互不抢帧）
        # 视频队列：必须 ≥ 一轮突发(16 帧)，否则每轮被截断（实测 12 帧上限把 25fps 打成 20fps）。
        # 24 帧 ≈ 65MB/会话；满了丢最旧，不会阻塞生产。
        self.__ws_video = _FifoSlot(maxsize=24)
        self.__ws_audio = _FifoSlot(maxsize=500)

        self.__container = avatar_session
        if hasattr(self.__container, 'output'):
            self.__container.output._player = self

    def push_video(self, frame):
        from av import VideoFrame
        new_frame = VideoFrame.from_ndarray(frame, format="bgr24")
        # ⚠ ICE 未连通时（真实环境里降级到 WS 就是这个状态）**没有任何消费者**会取这条队列，
        # 一旦写满（maxsize=100）`put` 就会阻塞 → process_frames/inference 全部被卡死
        # → 整条渲染管线停在几十帧后不再出画面（实测黑屏的根因）。
        # 所以这里必须丢最旧帧而不是阻塞；WS 通道有自己独立的队列，不受影响。
        try:
            self.__video._queue.put_nowait((new_frame, None))
        except queue.Full:
            try:
                self.__video._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.__video._queue.put_nowait((new_frame, None))
            except queue.Full:
                pass
        # 额外复制一份给 WS 降级通道（同一对象引用，开销可忽略）
        self.__ws_video.put(new_frame)

    def push_audio(self, frame, eventpoint=None):
        from av import AudioFrame
        new_frame = AudioFrame(format='s16', layout='mono', samples=frame.shape[0])
        new_frame.planes[0].update(frame.tobytes())
        new_frame.sample_rate = 16000
        # 同 push_video：ICE 未连通时这条队列也没有消费者，绝不能阻塞（否则 process_frames 被卡死）
        try:
            self.__audio._queue.put_nowait((new_frame, eventpoint))
        except queue.Full:
            try:
                self.__audio._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.__audio._queue.put_nowait((new_frame, eventpoint))
            except queue.Full:
                pass
        self.__ws_audio.put(new_frame)

    def ws_video_slot(self) -> _FifoSlot:
        return self.__ws_video

    def ws_audio_slot(self) -> _FifoSlot:
        return self.__ws_audio

    def get_buffer_size(self) -> int:
        return self.__video._queue.qsize()

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

        if not self.__started and self.__container is not None:
            # 不再强 close container，避免二次阻塞；依赖上层 session_manager.stop_session
            self.__container = None

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
