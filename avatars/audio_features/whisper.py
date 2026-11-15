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
#
#  Whisper 音频特征提取 — 用于 MuseTalk
#  迁移自 museasr.py
#

import os
import time
import threading
from collections import OrderedDict

import numpy as np

import queue
from queue import Queue
from avatars.audio_features.base_asr import BaseASR
from avatars.musetalk.whisper.audio2feature import Audio2Feature
from utils.logger import logger

# 进程级串行锁：同一份 whisper 权重被所有会话共享，前向放锁内，
# 避免多路并发时在卡上互相插队（原来各路在自己的渲染线程里直接抢）。
_WHISPER_LOCK = threading.Lock()

# 每个视频帧取多少 whisper 特征帧作为口型条件：'left,right'（单位=音频特征帧，50fps）。
# 原实现 [0,5] 只有前向约 200ms 上下文且不对称，批次边界口型容易跳；
# 可用 MUSETALK_FEAT_WIN='2,2' 切成上游 MuseTalk 的对称窗口做 A/B。
_FEAT_WIN_ENV = os.getenv('MUSETALK_FEAT_WIN', '0,5')
try:
    _FEAT_LEFT, _FEAT_RIGHT = (int(x) for x in _FEAT_WIN_ENV.split(','))
except Exception:
    _FEAT_LEFT, _FEAT_RIGHT = 0, 5

# ── 口型条件取窗模式（官方 MuseTalk 等价实现）─────────────────────────────
# utterance：TTS 整句音频只算一次 whisper 特征，按官方窗口 [2f-4, 2f+6) 取每帧条件
#            （官方 audio_processor.get_whisper_chunk：左补 4 帧零、右补 12 帧零，
#             帧 f 取 [2f, 2f+10)；相对帧起点约 -80ms/+120ms）。
# legacy   ：旧的 1.04s 滑窗 + [0,5] 取窗（仅作回退/麦克风流式输入）。
_FEAT_MODE = os.getenv('MUSETALK_FEAT_MODE', 'utterance').strip().lower()
_UTT_CACHE_MAX = 3
_UTT_WIN = 10      # 10 个 whisper 特征帧 = 200ms
_UTT_PAD_LEFT = 4  # 官方 audio_padding_length_left=2 → 2*2 帧零
_UTT_PAD_RIGHT = 12  # 官方 3*audio_padding_length_right*2 帧零


class _LazyFeat:
    """惰性特征负载：只携带本轮音频窗口，前向推迟到消费端确认「在说话」时再做。"""

    __slots__ = ('inputs', 'meta')

    def __init__(self, inputs, meta=None):
        self.inputs = inputs
        # 本轮批次每个 20ms 块的 (utt_id, chunk_index, type)，整句模式用
        self.meta = meta or []


class WhisperASR(BaseASR):
    def __init__(self, opt, parent, audio_processor:Audio2Feature):
        super().__init__(opt, parent)
        self.audio_processor = audio_processor
        # 整句特征缓存：utt_id -> {'audio': 16k float32, 'feat': padded feature 或 None}
        self._utts = OrderedDict()
        self._utt_lock = threading.Lock()
        logger.info("MuseTalk 口型特征模式: %s（MUSETALK_FEAT_MODE）", _FEAT_MODE)

    def set_utterance(self, utt_id: str, audio: np.ndarray):
        """登记一整句 TTS 音频（16k float32），特征在首次消费时算一次。"""
        if not utt_id or audio is None or len(audio) == 0:
            return
        with self._utt_lock:
            self._utts[utt_id] = {'audio': np.asarray(audio, dtype=np.float32).copy(),
                                  'feat': None}
            while len(self._utts) > _UTT_CACHE_MAX:
                self._utts.popitem(last=False)

    def _compute_utterance_feat(self, audio: np.ndarray) -> np.ndarray:
        """按官方 AudioProcessor.get_audio_feature 的语义算整句特征。

        30s 分段 → whisper encoder → 拼接 → 按真实时长（50fps）截断。
        """
        sr = 16000
        seg = 30 * sr
        feats = []
        for start in range(0, len(audio), seg):
            part = audio[start:start + seg]
            if len(part) == 0:
                continue
            feats.append(self.audio_processor.audio2feat(part))  # (t, 5, 384)
        if not feats:
            return np.zeros((0, 5, 384), dtype=np.float16)
        feat = np.concatenate(feats, axis=0)
        actual = max(1, int(len(audio) / sr * 50))
        return feat[:actual]

    def _padded_utterance_feat(self, utt_id: str):
        with self._utt_lock:
            entry = self._utts.get(utt_id)
        if entry is None:
            return None
        feat = entry['feat']
        if feat is None:
            with _WHISPER_LOCK:
                raw = self._compute_utterance_feat(entry['audio'])
            pad_l = np.zeros((_UTT_PAD_LEFT,) + raw.shape[1:], dtype=raw.dtype)
            pad_r = np.zeros((_UTT_PAD_RIGHT,) + raw.shape[1:], dtype=raw.dtype)
            feat = np.concatenate([pad_l, raw, pad_r], axis=0)
            entry['feat'] = feat
        return feat

    def _frame_feat(self, utt_id: str, frame_idx: int):
        """官方公式：帧 f 取 padded[2f, 2f+10)，即原始特征 [2f-4, 2f+6)。"""
        feat = self._padded_utterance_feat(utt_id)
        if feat is None or feat.shape[0] == 0:
            return None
        start = 2 * int(frame_idx)
        clip = feat[start:start + _UTT_WIN]
        if clip.shape[0] < _UTT_WIN:
            pad = np.zeros((_UTT_WIN - clip.shape[0],) + feat.shape[1:], dtype=feat.dtype)
            clip = np.concatenate([clip, pad], axis=0)
        return clip.reshape(-1, 384)

    def _utterance_chunks(self, meta):
        chunks = []
        zero = np.zeros((_UTT_WIN, 5, 384), dtype=np.float16).reshape(-1, 384)
        for j in range(self.batch_size):
            feat = None
            i0 = 2 * j
            if i0 < len(meta):
                utt_id, chunk_index, ftype = meta[i0]
                if ftype == 0 and utt_id and chunk_index >= 0:
                    feat = self._frame_feat(utt_id, chunk_index // 2)
            chunks.append(feat if feat is not None else zero)
        return chunks
    
    def _feature2chunks(self,feature_array,batch_size,audio_feat_win=[8,8],start=0,feature_idx_multiplier=1.0):
        """
        :param feature_array: 
        :param batch_size: batch大小
        :param audio_feat_win: 音频特征窗口大小，通常为 [左侧窗口大小, 右侧窗口大小]，单位为视频帧数
        :param start: 起始帧索引，通常为 stride_left_size/2
        :param feature_idx_multiplier: 用于将视频帧索引转换为特征索引的乘数，通常为 (特征提取的宽度 / 视频帧率)
        :return: 
        """
        feature_chunks = []
        #start += 10
        #feature_idx_multiplier = 50./fps 
        for i in range(batch_size):
            # start_idx = int(i * whisper_idx_multiplier)
            # if start_idx>=len(feature_array):
            #     break
            selected_feature,selected_idx = self._get_sliced_feature(
                feature_array=feature_array, vid_idx=i+start,
                audio_feat_win=audio_feat_win, feature_idx_multiplier=feature_idx_multiplier)
            #print(f"i:{i},selected_idx {selected_idx},feature_idx_multiplier:{feature_idx_multiplier}")
            feature_chunks.append(selected_feature.reshape(-1, 384))
        return feature_chunks

    def run_step(self):
        ############################################## extract audio feature ##############################################
        batch_meta = []
        for _ in range(self.batch_size*2):
            audio_frame = self.get_audio_frame()
            self.frames.append(audio_frame.data)
            batch_meta.append((audio_frame.utt_id, audio_frame.chunk_index,
                               audio_frame.type))
            self.output_queue.put(audio_frame)

        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return

        # 只投递本轮音频窗口（含 stride 上下文），**不在这里跑前向**：
        # 静音轮次的消费端走静态帧分支，前向结果本来就会被丢弃
        # （原实现每路待机每 640ms 白跑一次 Whisper，既浪费卡又压低待机帧率）。
        inputs = np.concatenate(self.frames) # [N * chunk]
        self.feat_queue.put(_LazyFeat(inputs, batch_meta))
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

    def resolve_feat_batch(self, payload):
        """消费端确认「本轮在说话」后，才真正执行 Whisper 前向。"""
        if not isinstance(payload, _LazyFeat):
            return payload
        # 整句模式（官方等价）：直接用整句特征按官方窗口切片，不再用滑窗重算
        if _FEAT_MODE == 'utterance' and payload.meta:
            try:
                return self._utterance_chunks(payload.meta)
            except Exception:
                logger.exception('整句特征取窗失败，回退 legacy 滑窗')
        t0 = time.perf_counter()
        with _WHISPER_LOCK:
            whisper_feature = self.audio_processor.audio2feat(payload.inputs)
        logger.debug("[whisper] 前向 %.1f ms", (time.perf_counter() - t0) * 1000)
        return self._feature2chunks(feature_array=whisper_feature,batch_size=self.batch_size,
                                    audio_feat_win = [_FEAT_LEFT, _FEAT_RIGHT],
                                    start=self.stride_left_size/2,
                                    feature_idx_multiplier=2)
