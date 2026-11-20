"""SoVITS 持久流式会话客户端。

与旧路径（每句一次 POST /tts）的区别：
  旧：文本 → POST → 首包 0.5~1.7s → 音频 → 连接关闭 → 下一句再付一次首包
  新：一条 WebSocket 常驻；文本随时推、音频连续回；段与段之间没有建连/首包往返

服务端协议见 zl-SoVITS/api_v2.py 的 /tts/ws。默认不启用，
由环境变量 TTS_SESSION_MODE=1 打开（回退只需去掉该变量）。
"""
from __future__ import annotations

import json
import threading
import time
from typing import Optional

import os
import time
import numpy as np
import soxr

try:
    import websocket  # websocket-client
except Exception:  # pragma: no cover
    websocket = None

from utils.logger import logger


class _Feeder:
    """把 48k 单声道 PCM16 切成 20ms 帧，同时产出 16k（口型）与播放口径。"""

    def __init__(self, owner):
        self.owner = owner
        self.chunk = int(getattr(owner, 'chunk', 320))
        self.play_chunk = int(getattr(owner, 'play_chunk', 960))
        self.play_rate = int(getattr(owner.parent, 'playback_rate', 48000) or 48000)
        self.rs16 = None
        self.rs_play = None
        self.p16 = np.zeros(0, dtype=np.float32)
        self.pp = np.zeros(0, dtype=np.float32)
        self.text = ''
        # 允许领先播放的最大时长（秒）：会话是 5x 实时到达的，不节流会把
        # 播放队列灌满，反而造成丢帧/卡顿。1.5s 足以吃掉段间首包延迟。
        try:
            self.lead = float(os.environ.get('TTS_SESSION_LEAD', '1.5') or 1.5)
        except (TypeError, ValueError):
            self.lead = 1.5
        self._anchor = None
        self._pushed = 0.0

    def start_segment(self, text: str):
        self.text = text or ''
        self.p16 = np.zeros(0, dtype=np.float32)
        self.pp = np.zeros(0, dtype=np.float32)

    def feed(self, raw: bytes, sr: int, first_flag: list):
        arr = np.frombuffer(raw, dtype='<i2').astype(np.float32) / 32768.0
        if arr.size == 0:
            return
        if self.rs16 is None:
            self.rs16 = soxr.ResampleStream(sr, 16000, 1, dtype='float32')
        s16 = self.rs16.resample_chunk(arr, last=False)
        if self.play_rate != sr:
            if self.rs_play is None:
                self.rs_play = soxr.ResampleStream(sr, self.play_rate, 1, dtype='float32')
            sp = self.rs_play.resample_chunk(arr, last=False)
        else:
            sp = arr
        self.p16 = np.concatenate((self.p16, s16)) if self.p16.size else s16
        self.pp = np.concatenate((self.pp, sp)) if self.pp.size else sp
        self._drain(first_flag)

    def _drain(self, first_flag: list):
        i16 = 0
        ip = 0
        while (self.p16.shape[0] - i16 >= self.chunk
               and self.pp.shape[0] - ip >= self.play_chunk):
            ev = {}
            if not first_flag[0]:
                ev = {'status': 'start', 'text': self.text}
                first_flag[0] = True
            self.owner.parent.put_audio_frame(
                self.p16[i16:i16 + self.chunk], ev,
                data_play=self.pp[ip:ip + self.play_chunk])
            i16 += self.chunk
            ip += self.play_chunk
            self._pace()
        self.p16 = self.p16[i16:] if self.p16.shape[0] > i16 else np.zeros(0, dtype=np.float32)
        self.pp = self.pp[ip:] if self.pp.shape[0] > ip else np.zeros(0, dtype=np.float32)

    def _pace(self) -> None:
        now = time.monotonic()
        if self._anchor is None:
            self._anchor = now
            self._pushed = 0.0
            return
        self._pushed += self.chunk / 16000.0
        ahead = self._pushed - (now - self._anchor)
        if ahead > self.lead:
            time.sleep(min(0.05, ahead - self.lead))

    def end_segment(self, text: str):
        # flush 两个流式重采样器的尾巴（不 flush 会削掉每段末尾几十毫秒）
        try:
            if self.rs16 is not None:
                t = self.rs16.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                if getattr(t, 'size', 0):
                    self.p16 = np.concatenate((self.p16, t))
                self.rs16 = None
            if self.rs_play is not None:
                t = self.rs_play.resample_chunk(np.zeros(0, dtype=np.float32), last=True)
                if getattr(t, 'size', 0):
                    self.pp = np.concatenate((self.pp, t))
                self.rs_play = None
        except Exception:
            logger.debug('session resampler flush failed', exc_info=True)
        # 先发完所有完整帧（不再触发 start 事件）
        self._drain([True])
        # 最后一帧不足 20ms：补零后带上 end 事件一起发
        pad16 = np.zeros(max(0, self.chunk - self.p16.shape[0]), dtype=np.float32)
        padp = np.zeros(max(0, self.play_chunk - self.pp.shape[0]), dtype=np.float32)
        f16 = np.concatenate((self.p16, pad16))[:self.chunk]
        fp = np.concatenate((self.pp, padp))[:self.play_chunk]
        self.p16 = np.zeros(0, dtype=np.float32)
        self.pp = np.zeros(0, dtype=np.float32)
        self.owner.parent.put_audio_frame(
            f16, {'status': 'end', 'text': text or self.text}, data_play=fp)


class SovitsSession:
    """一条常驻连接；send() 推文本，后台线程收音频并送进播放管线。"""

    def __init__(self, owner, url: str, query: dict):
        self.owner = owner
        self.url = url
        self.query = query or {}
        self.ws = None
        self.thread: Optional[threading.Thread] = None
        self.closed = threading.Event()
        self.sr = 48000
        self._last_send = time.monotonic()
        self._lock = threading.Lock()
        self._alive = False

    # ── 连接 ────────────────────────────────────────────────
    def _build_url(self) -> str:
        if not self.query:
            return self.url
        from urllib.parse import urlencode
        sep = '&' if '?' in self.url else '?'
        return f"{self.url}{sep}{urlencode(self.query)}"

    def open(self) -> bool:
        if websocket is None:
            logger.warning('[tts-session] websocket-client 未安装，回退到逐段请求')
            return False
        try:
            self.ws = websocket.create_connection(self._build_url(), timeout=30)
            msg = self.ws.recv()
            if isinstance(msg, str):
                d = json.loads(msg)
                self.sr = int(d.get('rate') or 48000)
            self.ws.settimeout(1.0)
            self._alive = True
            self.thread = threading.Thread(target=self._reader, name='tts-session-reader', daemon=True)
            self.thread.start()
            logger.info('[tts-session] connected rate=%d', self.sr)
            return True
        except Exception:
            logger.exception('[tts-session] connect failed')
            self.ws = None
            return False

    # ── 发送 ────────────────────────────────────────────────
    def send(self, text: str) -> bool:
        with self._lock:
            if not self._alive or self.ws is None:
                return False
            try:
                self.ws.send(json.dumps({'text': text}, ensure_ascii=False))
                self._last_send = time.monotonic()
                return True
            except Exception:
                logger.exception('[tts-session] send failed')
                self._alive = False
                return False

    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_send

    def finish(self, wait: float = 8.0) -> None:
        """告诉服务端没有更多文本，等它把剩余音频发完。"""
        if self.ws is not None and self._alive:
            try:
                self.ws.send(json.dumps({'type': 'end'}))
            except Exception:
                pass
        if self.thread is not None:
            self.thread.join(timeout=wait)
        self.close()

    def close(self) -> None:
        self._alive = False
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass
        self.ws = None

    # ── 收流 ────────────────────────────────────────────────
    def _reader(self) -> None:
        feeder = _Feeder(self.owner)
        cur_text = ''
        first_flag = [False]
        while self._alive:
            try:
                msg = self.ws.recv()
            except Exception as exc:
                name = type(exc).__name__
                if name in ('WebSocketTimeoutException', 'TimeoutError'):
                    continue
                break
            if msg is None or msg == '':
                break
            if isinstance(msg, str):
                try:
                    d = json.loads(msg)
                except Exception:
                    continue
                t = d.get('type')
                if t == 'seg_start':
                    cur_text = d.get('text') or ''
                    feeder.start_segment(cur_text)
                    first_flag = [False]
                elif t == 'seg_end':
                    feeder.end_segment(cur_text)
                elif t == 'seg_error':
                    logger.warning('[tts-session] segment error: %s', d.get('message'))
                continue
            try:
                feeder.feed(msg, self.sr, first_flag)
            except Exception:
                logger.exception('[tts-session] feed failed')
        self._alive = False
        self.closed.set()
