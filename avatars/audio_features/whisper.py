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

import time
import threading
import numpy as np

import queue
from queue import Queue
from avatars.audio_features.base_asr import BaseASR
from avatars.musetalk.whisper.audio2feature import Audio2Feature
from utils.logger import logger

# 进程级串行锁：同一份 whisper 权重被所有会话共享，前向放锁内，
# 避免多路并发时在卡上互相插队（原来各路在自己的渲染线程里直接抢）。
_WHISPER_LOCK = threading.Lock()


class _LazyFeat:
    """惰性特征负载：只携带本轮音频窗口，前向推迟到消费端确认「在说话」时再做。"""

    __slots__ = ('inputs',)

    def __init__(self, inputs):
        self.inputs = inputs


class WhisperASR(BaseASR):
    def __init__(self, opt, parent, audio_processor:Audio2Feature):
        super().__init__(opt, parent)
        self.audio_processor = audio_processor
    
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
        for _ in range(self.batch_size*2):
            audio_frame = self.get_audio_frame()
            self.frames.append(audio_frame.data)
            self.output_queue.put(audio_frame)

        if len(self.frames) <= self.stride_left_size + self.stride_right_size:
            return

        # 只投递本轮音频窗口（含 stride 上下文），**不在这里跑前向**：
        # 静音轮次的消费端走静态帧分支，前向结果本来就会被丢弃
        # （原实现每路待机每 640ms 白跑一次 Whisper，既浪费卡又压低待机帧率）。
        inputs = np.concatenate(self.frames) # [N * chunk]
        self.feat_queue.put(_LazyFeat(inputs))
        # discard the old part to save memory
        self.frames = self.frames[-(self.stride_left_size + self.stride_right_size):]

    def resolve_feat_batch(self, payload):
        """消费端确认「本轮在说话」后，才真正执行 Whisper 前向。"""
        if not isinstance(payload, _LazyFeat):
            return payload
        t0 = time.perf_counter()
        with _WHISPER_LOCK:
            whisper_feature = self.audio_processor.audio2feat(payload.inputs)
        logger.debug("[whisper] 前向 %.1f ms", (time.perf_counter() - t0) * 1000)
        return self._feature2chunks(feature_array=whisper_feature,batch_size=self.batch_size,
                                    audio_feat_win = [0,5],start=self.stride_left_size/2,
                                    feature_idx_multiplier=2)
