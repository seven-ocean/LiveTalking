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
#  Avatar 基类 — 合并自 basereal.py，集成到 Async Pipeline
#

import math
from numpy.typing import NDArray
import torch
import numpy as np
import subprocess
import os
import time
import cv2
import glob
import resampy
import soxr
import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf
import asyncio
from enum import Enum
import json
import importlib
import registry

import torch.multiprocessing as mp
from dataclasses import dataclass, field

from av import AudioFrame, VideoFrame
from fractions import Fraction

from utils.logger import logger
from utils.image import read_imgs,mirror_index

# 每帧右下角/左上角水印：默认关闭，避免每帧多一次绘制与像素拷贝
_FRAME_WATERMARK = os.environ.get("ZL_FRAME_WATERMARK", "0").strip().lower() in (
    "1", "true", "yes", "on")

# class State(Enum):
#     INIT=0
#     WAIT=1
#     QUESTION=2
#     ANSWER=3

@dataclass
class AudioFrameData:
    data: NDArray[np.float32]
    type: int = 0  # 默认值
    userdata: dict = field(default_factory=dict)
    # 播放采样（TTS 原生/44.1k）；data 仍为 16k，用于口型/ASR
    data_play: NDArray[np.float32] = None
    # 整句（utterance）元数据：TTS 整段 wav 一次性入队时标记句号与块序号，
    # 供 MuseTalk 按整句 whisper 特征切口型条件（官方等价做法）。
    utt_id: str = ''
    chunk_index: int = -1



class BaseAvatar:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // (opt.fps*2) # 320 samples per chunk (20ms)
        # 产出节流模式：legacy=严格按 batch/fps 节拍；
        # official=官方 LiveTalking"跑在前面+队列背压"；bounded=双队列有界背压；
        # sync25=视频与推理同频 25fps、按音频水位背压。
        self.pacing_mode = str(getattr(opt, 'pacing', '') or 'legacy').strip().lower()
        if self.pacing_mode not in ('legacy', 'official', 'bounded', 'sync25'):
            self.pacing_mode = 'legacy'
        # 播放采样率统一 48k：WebRTC/Opus 与多数声卡的原生时钟都是 48k，
        # 这样播放链路两端都不需要重采样；口型/ASR 仍是 16k（只转一次，不在播放路径）。
        # 口径优先级：本会话参数（能力中心「音频口径」）> 部署环境变量 > 48k
        try:
            self.playback_rate = int(
                getattr(opt, 'playback_rate', 0)
                or os.environ.get("TALK_PLAYBACK_SR", "48000")
                or 48000)
        except (TypeError, ValueError):
            self.playback_rate = 48000
        self.play_chunk = max(1, self.playback_rate // (opt.fps*2))
        # 播放侧有状态重采样器（只有帧没有 data_play 时才用，如待机静音/固定音频）
        self._play_rs = None
        self.sessionid = self.opt.sessionid

        self.speaking = False
        self.recording = False
        self.interrupted = False      # 打断标志，供 put_audio_file 循环检查
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.custom_audiotype = 0 # 0: normal, 1: sinlence, >1: custom audio
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        # self.custom_opt = {}
        self.__loadcustom()

        self.batch_size = opt.batch_size
        self.res_frame_queue = Queue(self.batch_size*2)
        self.render_event = Event()

        _tts_modules = {
            'edgetts': 'tts.edge',
            'gpt-sovits': 'tts.sovits',
            'xtts': 'tts.xtts',
            'cosyvoice': 'tts.cosyvoice',
            'fishtts': 'tts.fish',
            'tencent': 'tts.tencent',
            'doubao': 'tts.doubao',
            'indextts2': 'tts.indextts2',
            'azuretts': 'tts.azure',
            'qwentts': 'tts.qwentts',
            'omnitts': 'tts.omnitts'
        }

        if opt.tts in _tts_modules:
            importlib.import_module(_tts_modules[opt.tts])
            self.tts = registry.create("tts", opt.tts, opt=opt, parent=self)
        else:
            logger.error(f"TTS module {opt.tts} not found.")

        _output_modules = {
            'webrtc': 'streamout.webrtc',
            'rtcpush': 'streamout.webrtc',
            # 客户端 WS 降级通道：帧由 player 的 ws 槽位提供，输出模块仍需存在，
            # 否则 process_frames 会因为没有 self.output 直接崩掉。
            'ws': 'streamout.webrtc',
            'websocket': 'streamout.webrtc',
            'rtmp': 'streamout.rtmp',
            'virtualcam': 'streamout.virtualcam'
        }

        # 初始化 Output 模块
        if opt.transport in _output_modules:
            try:
                importlib.import_module(_output_modules[opt.transport])
                self.output = registry.create("streamout", opt.transport, opt=opt, parent=self)
            except ModuleNotFoundError:
                logger.error(f"Output transport module {_output_modules[opt.transport]} not found.")
        else:
            logger.error(f"Output transport {opt.transport} not found in map.")

    # 如果系统没有使用 pipeline，或者为了向后兼容原来的 ttsreal.py
    def put_msg_txt(self, msg, datainfo:dict={}):
        if hasattr(self, 'tts'):
            self.tts.put_msg_txt(msg, datainfo)
    
    def put_audio_frame(self, audio_chunk:NDArray[np.float32], datainfo:dict={},
                        utt_id:str='', chunk_index:int=-1,
                        data_play:NDArray[np.float32]=None): # 16khz 20ms pcm
        if hasattr(self, 'asr'):
            self.asr.put_audio_frame(audio_chunk, datainfo,
                                     utt_id=utt_id, chunk_index=chunk_index,
                                     data_play=data_play)

    def _next_utt_id(self) -> str:
        self._utt_seq = getattr(self, '_utt_seq', 0) + 1
        sid = getattr(self, 'sessionid', '') or getattr(
            getattr(self, 'opt', None), 'sessionid', '') or 'utt'
        return f"{sid}-{self._utt_seq}"

    def put_audio_file(self, filebyte, datainfo:dict={}): 
        self.interrupted = False  # 每个新文件开始时重置
        # 口型/ASR 用 16k；播放用 TTS 原生/44.1k，避免 44.1→16 的带宽裁切
        stream = self.__create_bytes_stream(BytesIO(filebyte))
        stream_play = self.__create_bytes_stream_play(BytesIO(filebyte))
        # 整句登记：MuseTalk 需要按整句算一次 whisper 特征（官方等价做法）
        utt_id = self._next_utt_id()
        if hasattr(self.asr, 'set_utterance'):
            self.asr.set_utterance(utt_id, stream)
        streamlen = stream.shape[0]
        idx = 0
        idx_play = 0
        first = True
        while streamlen >= self.chunk:
            if self.interrupted:
                self.interrupted = False
                logger.info(f'[INFO] put_audio_file interrupted by flush_talk, remaining {streamlen} samples')
                break
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo)
            dplay = stream_play[idx_play:idx_play+self.play_chunk]
            if dplay.shape[0] < self.play_chunk:
                dplay = np.pad(dplay, (0, self.play_chunk - dplay.shape[0]))
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint,
                                 utt_id=utt_id, chunk_index=idx // self.chunk,
                                 data_play=dplay)
            streamlen -= self.chunk
            idx += self.chunk
            idx_play += self.play_chunk

    def put_audio_filepath(self, filepath, datainfo:dict={}): 
        self.interrupted = False  # 每个新文件开始时重置
        stream = self.__create_bytes_stream(filepath)
        stream_play = self.__create_bytes_stream_play(filepath)
        utt_id = self._next_utt_id()
        if hasattr(self.asr, 'set_utterance'):
            self.asr.set_utterance(utt_id, stream)
        streamlen = stream.shape[0]
        idx = 0
        idx_play = 0
        first = True
        while streamlen >= self.chunk:
            if self.interrupted:
                self.interrupted = False
                logger.info(f'[INFO] put_audio_filepath interrupted by flush_talk, remaining {streamlen} samples')
                break
            eventpoint = {}
            if first:
                eventpoint = {'status': 'start'}
                first = False
            if streamlen - self.chunk < self.chunk:
                eventpoint = {'status': 'end'}
            eventpoint.update(**datainfo)
            dplay = stream_play[idx_play:idx_play+self.play_chunk]
            if dplay.shape[0] < self.play_chunk:
                dplay = np.pad(dplay, (0, self.play_chunk - dplay.shape[0]))
            self.put_audio_frame(stream[idx:idx+self.chunk], eventpoint,
                                 utt_id=utt_id, chunk_index=idx // self.chunk,
                                 data_play=dplay)
            streamlen -= self.chunk
            idx += self.chunk
            idx_play += self.play_chunk

    
    def __create_bytes_stream(self, byte_stream):
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0] > 0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def __create_bytes_stream_play(self, byte_stream):
        """播放采样：保留 TTS 原生；非 48k 时整句只重采样一次到 playback_rate。"""
        stream, sample_rate = sf.read(byte_stream)
        stream = stream.astype(np.float32)
        if stream.ndim > 1:
            stream = stream[:, 0]
        if sample_rate != self.playback_rate and stream.shape[0] > 0:
            logger.info('[INFO] playback resample %s -> %s (once per file)',
                        sample_rate, self.playback_rate)
            stream = soxr.resample(stream, sample_rate, self.playback_rate)
        return stream

    def flush_talk(self):
        if hasattr(self, 'tts') and hasattr(self.tts, 'flush_talk'):
            self.tts.flush_talk()
        if hasattr(self, 'asr') and hasattr(self.asr, 'flush_talk'):
            self.asr.flush_talk()
        self.custom_audiotype = 0
        self.interrupted = True                     # 设置打断标志
        # 清空 mel 特征队列，防止 inference 线程读到旧 mel chunks 导致 ONNX shape mismatch
        while not self.asr.feat_queue.empty():
            try:
                self.asr.feat_queue.get_nowait()
            except queue.Empty:
                break
        # 清空音频帧输出队列，防止 inference 线程读到旧 audio frames
        while not self.asr.output_queue.empty():
            try:
                self.asr.output_queue.get_nowait()
            except queue.Empty:
                break
        # 清空渲染帧队列中的残留帧（避免旧 TTS 帧继续渲染）
        while not self.res_frame_queue.empty():
            try:
                self.res_frame_queue.get_nowait()
            except queue.Empty:
                break
        # 清空 MelASR 音频帧滑动窗口，避免新旧音频混合导致口型异常
        if hasattr(self, 'asr') and hasattr(self.asr, 'frames'):
            self.asr.frames.clear()

    # def flush(self):
    #     self.flush_talk()

    def is_speaking(self) -> bool:
        return self.speaking
    
    def __loadcustom(self):
        """加载自定义动作配置（动作编排）。

        加载顺序（同一 audiotype，高优先级覆盖低优先级）：
          1) per-avatar 本地配置 data/avatars/{avatar_id}/actions.json —— 该形象自己的动作，最高优先；
          2) 显式配置（CLI --customvideo_config / offer custom_config）—— 按 avatar_id 过滤后兜底，
             仅补充 actions.json 未覆盖的 audiotype；配置项 avatar_id 字段为空或缺省视为全局动作。

        这样在 A 形象切换到 B 形象时，B 只加载 B 自己的动作（per-avatar 优先），
        全局/跨形象的动作项被过滤，避免自定义动作串用。
        """
        avatar_id = getattr(self.opt, 'avatar_id', '')
        merged = []
        seen_types = set()

        # 1) per-avatar 本地配置：data/avatars/{avatar_id}/actions.json（操作台生成）
        action_file = os.path.join('data', 'avatars', avatar_id, 'actions.json')
        if avatar_id and os.path.exists(action_file):
            try:
                with open(action_file, 'r', encoding='utf-8') as f:
                    local_items = json.load(f)
                merged.extend(local_items)
                seen_types = {item['audiotype'] for item in local_items}
                logger.info(f"[custom] 形象 {avatar_id} 加载 per-avatar 动作 {len(local_items)} 项: {action_file}")
            except Exception as e:
                logger.error(f"[custom] 加载 {action_file} 失败: {e}")

        # 2) 显式配置（CLI customvideo_config / offer custom_config）：按 avatar_id 过滤
        if hasattr(self.opt, 'customopt') and self.opt.customopt:
            for item in self.opt.customopt:
                item_avatar = item.get('avatar_id', '')
                if item_avatar and item_avatar != avatar_id:
                    logger.warning(f"[custom] 跳过 {item_avatar} 的动作项（当前形象 {avatar_id}）")
                    continue
                if item['audiotype'] in seen_types:
                    continue  # 形象自己的动作优先，全局项不覆盖
                merged.append(item)
                seen_types.add(item['audiotype'])

        # 3) 逐项加载；imgpath 无匹配图片时跳过并告警
        for item in merged:
            if item.get('audiotype') is None:
                logger.warning(f"[custom] 跳过无效动作项（缺少 audiotype）: {item}")
                continue
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            if not input_img_list:
                logger.warning(f"[custom] 跳过动作 audiotype={item['audiotype']}: imgpath 无匹配图片: {item.get('imgpath')}")
                continue
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            if item.get('audiopath'):
                if os.path.exists(item['audiopath']):
                    self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
                    self.custom_audio_index[item['audiotype']] = 0
                else:
                    logger.warning(f"[custom] 动作 audiotype={item['audiotype']} audiopath 不存在: {item['audiopath']}")
            self.custom_index[item['audiotype']] = 0

    def init_customindex(self):
        self.custom_audiotype = 0
        for key in self.custom_audio_index:
            self.custom_audio_index[key] = 0
        for key in self.custom_index:
            self.custom_index[key] = 0

    def notify(self, eventpoint:dict):
        if eventpoint and eventpoint.get('status'):
            logger.info("notify:%s", eventpoint)

    def start_recording(self):
        if self.recording:
            return
        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24',
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    f'temp{self.opt.sessionid}.mp4']
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    '-ac', '1',
                    '-ar', '48000',
                    '-i', '-',
                    '-acodec', 'aac',
                    f'temp{self.opt.sessionid}.aac']
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
    
    def record_video_data(self, image):
        if self.width == 0:
            self.height, self.width, _ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tobytes()) #tostring()

    def record_audio_data(self, frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tobytes())
		
    def stop_recording(self):
        if not self.recording:
            return
        self.recording = False 
        self._record_video_pipe.stdin.close()
        self._record_video_pipe.wait()
        self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()
        
        record_path = os.path.join('data', 'record')
        os.makedirs(record_path, exist_ok=True)
        output_file = os.path.join(record_path, f"{self.opt.sessionid}.mp4")
        
        temp_aac = f"temp{self.opt.sessionid}.aac"
        temp_mp4 = f"temp{self.opt.sessionid}.mp4"
        
        cmd_combine_audio = f"ffmpeg -y -i {temp_aac} -i {temp_mp4} -c:v copy -c:a copy {output_file}"
        os.system(cmd_combine_audio)
        
        # 删除临时文件
        try:
            os.remove(temp_aac)
            os.remove(temp_mp4)
        except Exception as e:
            logger.error(f"Error removing temp files: {e}")

    # def mirror_index(self, size, index):
    #     turn = index // size
    #     res = index % size
    #     if turn % 2 == 0:
    #         return res
    #     else:
    #         return size - res - 1 
    
    def get_custom_audio_stream(self, audiotype):
        if self.custom_audio_index.get(audiotype) is None:
            # 该动作未配置固定音频（如讲解 reply / 无音频的思考），返回静音帧避免 KeyError
            return np.zeros(self.chunk, dtype=np.float32)
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype] >= self.custom_audio_cycle[audiotype].shape[0]:
            self.custom_audiotype = 1
        return stream
    
    def set_custom_state(self, audiotype, reinit=True):
        print('set_custom_state:', audiotype)
        # 以画面帧为门禁：无画面帧的动作无法显式切换；
        # 原实现以 custom_audio_index 为门禁，导致无固定音频的动作（idle=1 / think=3）无法切换
        if self.custom_img_cycle.get(audiotype) is None:
            return
        self.custom_audiotype = audiotype
        if reinit:
            if audiotype in self.custom_audio_index:
                self.custom_audio_index[audiotype] = 0
            if audiotype in self.custom_index:
                self.custom_index[audiotype] = 0

    # ========================== 核心渲染及 Pipeline 桥接 ==========================
    def get_avatar_length(self):
        if hasattr(self, 'frame_list_cycle'):
            return len(self.frame_list_cycle)
        return 1
        
    def inference(self, quit_event):
        length = self.get_avatar_length()
        index = 0
        count = 0
        counttime = 0
        last_speaking = False

        # syncnet_T = 12  # 时间步
        # weight_dtype = torch.float16  # 数据类型
        # infernum = 0
        logger.info('start inference')
        while not quit_event.is_set():
            starttime = time.perf_counter()
            audiofeat_batch = []
            try:
                audiofeat_batch = self.asr.feat_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            is_all_silence = True
            audio_frames: list[AudioFrameData] = []
            for _ in range(self.batch_size * 2):
                audioframe:AudioFrameData = self.asr.output_queue.get()
                if audioframe.type == 0:
                    is_all_silence = False               
                audio_frames.append(audioframe)

            # 只有「本轮确实在说话」才需要口型特征；静音轮次直接走静态帧分支。
            # MuseTalk 的 Whisper 前向正是在这一步（resolve_feat_batch）才真正执行，
            # 所以待机时一次前向都不会跑。
            if not is_all_silence:
                try:
                    audiofeat_batch = self.asr.resolve_feat_batch(audiofeat_batch)
                except Exception:
                    logger.exception('resolve feat batch failed, treating as silence')
                    audiofeat_batch = None
                # 防御：特征批次不完整（flush/reconnect 时常见）不要整轮当静音，
                # 否则说话首尾会掉口型。短则用最后一帧补齐，长则截断。
                if audiofeat_batch is not None:
                    if hasattr(audiofeat_batch, 'detach'):
                        audiofeat_batch = audiofeat_batch.detach().cpu().numpy()
                    try:
                        _got = len(audiofeat_batch)
                    except TypeError:
                        _got = None
                    if _got == 0:
                        audiofeat_batch = None
                    elif _got is not None and _got < self.batch_size:
                        _arr = np.asarray(audiofeat_batch)
                        _pad = self.batch_size - _got
                        _last = _arr[-1:]
                        audiofeat_batch = np.concatenate(
                            [_arr, np.repeat(_last, _pad, axis=0)], axis=0
                        )
                        logger.warning(
                            f'[WARN] audio feat short: got {_got}, padded to {self.batch_size}'
                        )
                    elif _got is not None and _got > self.batch_size:
                        audiofeat_batch = np.asarray(audiofeat_batch)[:self.batch_size]
                        logger.warning(
                            f'[WARN] audio feat long: got {_got}, truncated to {self.batch_size}'
                        )
                if audiofeat_batch is None or len(audiofeat_batch) != self.batch_size:
                    _got = None if audiofeat_batch is None else len(audiofeat_batch)
                    logger.warning(f'[WARN] audio feat batch size mismatch: got {_got}, expected {self.batch_size}. Treating as silence.')
                    for af in audio_frames:
                        af.type = 1
                    is_all_silence = True

             # 检测状态变化
            current_speaking = not is_all_silence

            if is_all_silence: #全为静音数据，只需要取fullimg，不需要推理
                for i in range(self.batch_size):
                    idx = mirror_index(length, index)
                    self.res_frame_queue.put((None, audio_frames[i*2:i*2+2], idx))
                    index = index + 1
            else:
                if current_speaking and not last_speaking and self.custom_index.get(1) is not None: #从静音到说话切换,并且有自定义静态视频
                    index = 0
                t = time.perf_counter()

                try:
                    pred = self.inference_batch(index, audiofeat_batch)
                except Exception:
                    # 单批推理失败（最常见是 CUDA OOM）不能让推理线程退出，
                    # 否则画面会永久停在最后一帧；这里释放缓存并退回静态帧继续播放。
                    logger.exception('inference_batch 失败，本批退回静态帧')
                    self.__dict__.pop('_prev_frame_latent', None)
                    try:
                        import torch as _torch
                        if _torch.cuda.is_available():
                            _torch.cuda.empty_cache()
                    except Exception:
                        pass
                    for i in range(self.batch_size):
                        sidx = mirror_index(length, index)
                        self.res_frame_queue.put(
                            (None, audio_frames[i*2:i*2+2], sidx))
                        index = index + 1
                    continue

                counttime += (time.perf_counter() - t)
                count += self.batch_size
                if count >= 100:
                    logger.info(f"------actual avg infer fps:{count/counttime:.4f}")
                    count = 0
                    counttime = 0
                for i, res_frame in enumerate(pred):
                    self.res_frame_queue.put((res_frame, audio_frames[i*2:i*2+2], mirror_index(length, index)))
                    index = index + 1
                    
            if current_speaking != last_speaking:
                logger.info(f"inference 状态切换：{'说话' if last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                last_speaking = current_speaking         
        logger.info('baseavatar inference thread stop')

    def process_frames(self,quit_event):
        enable_transition = False  # 设置为False禁用过渡效果，True启用
        
        _last_speaking = False
        _transition_start = time.time()
        if enable_transition:
            _transition_duration = 0.1  # 过渡时间
            _last_silent_frame = None  # 静音帧缓存
            _last_speaking_frame = None  # 说话帧缓存

        self.output.start()
        
        while not quit_event.is_set():
            try:
                audio_frames: list[AudioFrameData]
                res_frame,audio_frames,idx = self.res_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
            
            # 检测状态变化
            current_speaking = not (audio_frames[0].type!=0 and audio_frames[1].type!=0)
            if current_speaking != _last_speaking:
                logger.info(f"[{self.sessionid}] 状态切换：{'说话' if _last_speaking else '静音'} → {'说话' if current_speaking else '静音'}")
                _transition_start = time.time()
            _last_speaking = current_speaking

            if audio_frames[0].type!=0 and audio_frames[1].type!=0: #全为静音数据，只需要取fullimg
                self.speaking = False
                audiotype = audio_frames[0].type
                if self.custom_index.get(audiotype) is not None: #有自定义视频
                    mirindex = mirror_index(len(self.custom_img_cycle[audiotype]),self.custom_index[audiotype])
                    target_frame = self.custom_img_cycle[audiotype][mirindex]
                    self.custom_index[audiotype] += 1
                else:
                    target_frame = self.frame_list_cycle[idx]
                
                if enable_transition:
                    # 说话→静音过渡
                    if time.time() - _transition_start < _transition_duration and _last_speaking_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_speaking_frame, 1-alpha, target_frame, alpha, 0)
                    else:
                        combine_frame = target_frame
                    # 缓存静音帧
                    _last_silent_frame = combine_frame.copy()
                else:
                    combine_frame = target_frame
            else:
                self.speaking = True
                try:
                    current_frame = self.paste_back_frame(res_frame,idx)
                except Exception as e:
                    logger.warning(f"paste_back_frame error: {e}")
                    continue
                if enable_transition:
                    # 静音→说话过渡
                    if time.time() - _transition_start < _transition_duration and _last_silent_frame is not None:
                        alpha = min(1.0, (time.time() - _transition_start) / _transition_duration)
                        combine_frame = cv2.addWeighted(_last_silent_frame, 1-alpha, current_frame, alpha, 0)
                    else:
                        combine_frame = current_frame
                    # 缓存说话帧
                    _last_speaking_frame = combine_frame.copy()
                else:
                    combine_frame = current_frame

            if _FRAME_WATERMARK:
                cv2.putText(combine_frame, "zl-talking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            
            # 使用统一输出接口推送视频帧
            self.output.push_video_frame(combine_frame)
            self.record_video_data(combine_frame)

            for audio_frame in audio_frames:
                # 口型/ASR 用 16k 的 data；播放用 data_play（TTS 原生/44.1k）
                play = getattr(audio_frame, 'data_play', None)
                if play is None:
                    play = audio_frame.data
                    if self.playback_rate != self.sample_rate and play.shape[0] > 0:
                        # 逐帧做无状态重采样会在每个 20ms 边界留下一次瞬态（电流音）。
                        # 这里改成每个会话一个有状态重采样器，跨帧连续。
                        if self._play_rs is None:
                            self._play_rs = soxr.ResampleStream(
                                self.sample_rate, self.playback_rate, 1, dtype='float32')
                        play = self._play_rs.resample_chunk(
                            np.asarray(play, dtype='float32'), last=False)
                        if play.shape[0] != self.play_chunk:
                            if play.shape[0] < self.play_chunk:
                                play = np.pad(play, (0, self.play_chunk - play.shape[0]))
                            else:
                                play = play[:self.play_chunk]
                play = np.clip(play, -1.0, 1.0)
                frame = (play * 32767).astype(np.int16)
                self.output.push_audio_frame(frame, audio_frame.userdata,
                                             sample_rate=self.playback_rate)
                self.record_audio_data(frame)
                
            # if self.opt.transport == 'virtualcam' and hasattr(self.output, '_cam') and self.output._cam:
            #     self.output._cam.sleep_until_next_frame()

        self.output.stop()
        logger.info('baseavatar process_frames thread stop') 

    def render(self,quit_event):
        self.quit_event = quit_event
        
        self.init_customindex()
        self.tts.render(quit_event)

        # 注意：infer_thread 是普通 Thread，应使用 threading.Event，而非 mp.Event
        infer_quit_event = Event()
        infer_thread = Thread(target=self.inference, args=(infer_quit_event,))
        infer_thread.start()
        
        process_quit_event = Event()
        process_thread = Thread(target=self.process_frames, args=(process_quit_event,))
        process_thread.start()

        count=0
        totaltime=0
        _starttime=time.perf_counter()
        _totalframe=0
        # 一轮产出 batch_size 帧，对应 batch_size/fps 秒视频 → 这是产出的时间节拍
        _round_interval = self.batch_size / float(self.asr.fps or 25)
        _deadline = time.monotonic()
        _pacing_official = (getattr(self, 'pacing_mode', 'legacy') == 'official')
        _pacing_bounded = (getattr(self, 'pacing_mode', 'legacy') == 'bounded')
        _pacing_sync25 = (getattr(self, 'pacing_mode', 'legacy') == 'sync25')
        _pacing_runway = (getattr(self, 'pacing_mode', 'legacy') == 'runway')
        while not quit_event.is_set():
            t = time.perf_counter()
            self.asr.run_step()
            if quit_event.is_set():
                break

            if _pacing_runway:
                # runway：允许跑在前面（不按 25fps 硬节拍自我限速），只在
                # 视频队列攒到 40 帧 / 音频 100 帧时停等。这样推理慢一拍时
                # 播放端仍有几秒盈余；真断供时由 frame_pace 重复上一帧兜底。
                _player = getattr(self.output, "_player", None)
                while not quit_event.is_set() and _player is not None:
                    try:
                        _d = _player.queue_depths()
                    except Exception:
                        break
                    if int(_d.get("vq", 0) or 0) < 40 and int(_d.get("aq", 0) or 0) < 100:
                        break
                    time.sleep(0.01)
                continue

            if _pacing_sync25:
                # sync25：视频轨与推理同频（25fps），音频水位维持在 0.24~0.5s：
                #  • 水位见底（<12 帧 / WS<25）→ 立刻补产，缓冲只多不少，播放端永不欠载；
                #  • 到顶（音频 25 帧 / WS 50 / 视频 16 帧）→ 停等，绝不丢帧
                #    （丢帧 = 画面停一下 + 音频继续走 → 之后口型永久对不上）；
                #  • 其余时间按 25fps 媒体节拍走。
                # 视频队列参与背压是本轮的关键：发送端被 TURN/TCP 顶住时，
                # 之前会把队列里的帧丢掉，现在改成让生产端等，音画时间轴保持一致。
                _player = getattr(self.output, "_player", None)
                _aq = 0
                _wsq = 0
                _vq = 0
                if _player is not None:
                    _t_wait = time.monotonic()
                    while not quit_event.is_set():
                        try:
                            _d = _player.queue_depths()
                        except Exception:
                            break
                        _aq = int(_d.get("aq", 0) or 0)
                        _wsq = int(_d.get("wsq", 0) or 0)
                        _vq = int(_d.get("vq", 0) or 0)
                        # 视频队列等待最多 0.8s（音频缓冲 ≈0.6s）：发送端卡死时
                        # 不能把音频一起饿死，超时就放行、让音频优先。
                        _v_ok = _vq < 16 or (time.monotonic() - _t_wait) > 0.8
                        if _aq < 25 and _wsq < 50 and _v_ok:
                            break
                        time.sleep(0.01)
                if _player is None or _aq >= 12 or _wsq >= 25:
                    _deadline += _round_interval
                    _now = time.monotonic()
                    if _deadline > _now:
                        _left = _deadline - _now
                        while not quit_event.is_set() and _left > 0:
                            time.sleep(min(0.05, _left))
                            _left = _deadline - time.monotonic()
                    else:
                        _deadline = _now
                else:
                    # 缓冲见底：本轮不睡，连续补产直到回到下限以上
                    _deadline = time.monotonic()
                continue

            if _pacing_bounded:
                # 有界队列 + 阻塞背压：维持在「音频 ~500ms、视频 ~400ms」的水位，
                # 队列满了就让生产端等（绝不丢帧），发送端因此永远不会饿、也不会加速。
                _player = getattr(self.output, "_player", None)
                while not quit_event.is_set() and _player is not None:
                    try:
                        d = _player.queue_depths()
                    except Exception:
                        break
                    if d.get("aq", 0) < 25 and d.get("vq", 0) < 10 and d.get("wsq", 0) < 50:
                        break
                    time.sleep(0.01)
                continue

            if _pacing_official:
                # 官方 LiveTalking 模型：生产者尽量跑在前面，只按输出队列积压做背压。
                # 队列里常驻 ≥5 帧（200ms）缓冲，RTP 发送端永远不会饿。
                try:
                    _bs = self.output.get_buffer_size()
                except Exception:
                    _bs = 0
                if _bs >= 5:
                    time.sleep(min(0.3, 0.04 * _bs * 0.8))
                continue

            # 产出节拍：一轮 = batch_size 帧 = batch_size/fps 秒的视频（25fps、batch16 → 640ms）。
            # 必须按**时间**节流，而且**不能**按输出队列深度做背压：
            #   • asr.run_step() 一次就消费 batch_size*2 个音频帧（= batch_size*40ms 音频），
            #     所以一轮天然对应 640ms 的实时内容；
            #   • process_frames 是**整批突发**写队列的（16 帧一次性入队），任何「队列深度 >= 5
            #     就睡」的公式都会在正常突发时误睡 0.5s，把产出压到 13fps 以下 —— 音频内容
            #     随之少于实时（40ms/帧），客户端就会周期性断音、画面一顿一顿。
            # 音画对齐不靠这里，而是靠传输层的 seq 门控：音频必须等对应视频帧真的发出去，
            # 视频慢时音频跟着等，绝不抢跑。
            _deadline += _round_interval
            _now = time.monotonic()
            if _deadline > _now:
                while not quit_event.is_set():
                    _left = _deadline - time.monotonic()
                    if _left <= 0:
                        break
                    time.sleep(min(0.05, _left))
            else:
                _deadline = _now  # 落后则重新对齐，避免越睡越慢
        logger.info('baseavatar render thread stop')

        infer_quit_event.set()
        infer_thread.join(timeout=3.0)
        if infer_thread.is_alive():
            logger.warning("inference thread did not stop within 3s")

        process_quit_event.set()
        process_thread.join(timeout=3.0)
        if process_thread.is_alive():
            logger.warning("process_frames thread did not stop within 3s")

