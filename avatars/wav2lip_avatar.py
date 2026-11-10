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
#  Wav2Lip 数字人 — 迁移自 lipreal.py + lipasr.py
#

import math
import torch
import numpy as np

import os
import time
import cv2
import glob
import pickle
import copy
import json

import queue
from queue import Queue
from threading import Event
import torch.multiprocessing as mp

from avatars.audio_features.mel import MelASR
import asyncio
from av import AudioFrame, VideoFrame
from avatars.wav2lip.models import Wav2Lip
from avatars.base_avatar import BaseAvatar

from tqdm import tqdm
from utils.logger import logger
from utils.image import read_imgs, mirror_index
from utils.device import initialize_device
from registry import register

device = initialize_device()
logger.info('Using {} for inference.'.format(device))

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    logger.info("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()


class ONNXWav2LipWrapper:
    """ONNX 模型包装器，提供与 PyTorch 模型相同的接口"""
    def __init__(self, onnx_path, opt=None):
        import onnxruntime as ort
        self.onnx_path = onnx_path
        logger.info(f"Loading ONNX model from: {onnx_path}")

        # 配置 ONNX Runtime — 自动适配 NVIDIA CUDA / 沐曦 MACA / CPU
        from utils.device import get_onnx_providers
        use_trt = bool(getattr(opt, 'use_trt', False))
        trt_profiles = None
        if use_trt:
            # wav2lip 是动态 batch，需要给出 TRT 的 min/opt/max 形状（batch 固定为 --batch_size）
            bs = int(getattr(opt, 'batch_size', 16) or 16)
            trt_profiles = {
                'trt_profile_min_shapes': 'audio_sequences:1x1x80x16,face_sequences:1x6x256x256',
                'trt_profile_opt_shapes': f'audio_sequences:{bs}x1x80x16,face_sequences:{bs}x6x256x256',
                'trt_profile_max_shapes': f'audio_sequences:{bs}x1x80x16,face_sequences:{bs}x6x256x256',
            }
        providers = get_onnx_providers(
            use_trt=use_trt,
            trt_cache_dir=getattr(opt, 'trt_cache_dir', None),
            trt_tag='wav2lip',
            trt_profiles=trt_profiles,
            trt_fp16=bool(getattr(opt, 'trt_fp16', False)))
        logger.info(f"ONNX Runtime available providers: {ort.get_available_providers()}")
        logger.info(f"Trying to use providers: {[p if isinstance(p, str) else p[0] for p in providers]}")
        if any(isinstance(p, tuple) for p in providers):
            logger.info("正在构建/加载 wav2lip TensorRT 引擎（首次约 1~2 分钟，之后走缓存）…")
        try:
            self.session = ort.InferenceSession(onnx_path, providers=providers)
        except Exception as e:
            fallback = [p for p in providers if not isinstance(p, tuple)]
            logger.warning(f"TensorRT 初始化失败（{e}），回退 {fallback}")
            self.session = ort.InferenceSession(onnx_path, providers=fallback)
        
        # 获取实际使用的 provider
        actual_providers = self.session.get_providers()
        logger.info(f"ONNX model actually using providers: {actual_providers}")
        
        # 获取输入输出信息
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        logger.info(f"ONNX model inputs: {self.input_names}")
        logger.info(f"ONNX model outputs: {self.output_names}")
    
    def __call__(self, audio_sequences, face_sequences):
        """推理接口，兼容 PyTorch 模型调用方式"""
        import numpy as np
        
        # 将 torch.Tensor 转换为 numpy
        if isinstance(audio_sequences, torch.Tensor):
            audio_np = audio_sequences.cpu().numpy()
        else:
            audio_np = audio_sequences
            
        if isinstance(face_sequences, torch.Tensor):
            face_np = face_sequences.cpu().numpy()
        else:
            face_np = face_sequences
        
        # 运行推理
        outputs = self.session.run(
            None,
            {
                self.input_names[0]: audio_np,
                self.input_names[1]: face_np
            }
        )
        
        # 将输出转换回 torch.Tensor
        return torch.from_numpy(outputs[0]).to(device)
    
    def eval(self):
        """兼容接口，ONNX 模型不需要 eval 模式"""
        return self
    
    def to(self, device):
        """兼容接口，ONNX 模型在 session 创建时已确定设备"""
        return self


def load_onnx_model(onnx_path, opt=None):
    """加载 ONNX 格式的 Wav2Lip 模型"""
    return ONNXWav2LipWrapper(onnx_path, opt)

def _load_frame_range(avatar_id, total):
    """读取 avatar.json 的 frame_range，只让正脸帧进入推理循环。

    偏头/侧脸帧上 Wav2Lip 生成的嘴部贴回原帧会错位，把这段排除在循环之外
    是最直接的解法。未配置时返回整段，行为与历史版本一致。
    """
    meta_file = os.path.join('data', 'avatars', avatar_id, 'avatar.json')
    try:
        with open(meta_file, 'r', encoding='utf-8') as f:
            rng = json.load(f).get('frame_range')
        start, end = int(rng[0]), int(rng[1])
    except Exception:
        return 0, total
    start = max(0, min(total - 1, start))
    end = max(start + 1, min(total, end))
    if end - start >= total:
        return 0, total
    logger.info(f"[wav2lip] 形象 {avatar_id} 仅循环正脸帧 {start}~{end - 1}（共 {total} 帧）")
    return start, end


def load_avatar(avatar_id):
    avatar_path = f"./data/avatars/{avatar_id}"
    full_imgs_path = f"{avatar_path}/full_imgs" 
    face_imgs_path = f"{avatar_path}/face_imgs" 
    coords_path = f"{avatar_path}/coords.pkl"
    align_path = f"{avatar_path}/align.pkl"
    
    with open(coords_path, 'rb') as f:
        coord_list_cycle = pickle.load(f)
    frame_list_cycle = None
    input_img_list = glob.glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    frame_list_cycle = read_imgs(input_img_list)
    input_face_list = glob.glob(os.path.join(face_imgs_path, '*.[jpJP][pnPN]*[gG]'))
    input_face_list = sorted(input_face_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    face_list_cycle = read_imgs(input_face_list)

    # 头部倾斜对齐矩阵（新构建的形象才有），用于把生成的下半脸按原姿态贴回
    align_list_cycle = None
    if os.path.exists(align_path):
        try:
            with open(align_path, 'rb') as f:
                align_list_cycle = pickle.load(f)
            if len(align_list_cycle) != len(coord_list_cycle):
                logger.warning(f"[wav2lip] {avatar_id} align.pkl 帧数不匹配，回退整框贴回")
                align_list_cycle = None
        except Exception as e:
            logger.warning(f"[wav2lip] 读取 {align_path} 失败，回退整框贴回: {e}")
            align_list_cycle = None

    # 嘴部位置（对齐裁剪空间内的归一化框），用于把生成结果只贴回嘴部区域
    mouth_list_cycle = None
    mouth_path = f"{avatar_path}/mouth.pkl"
    if align_list_cycle is not None and os.path.exists(mouth_path):
        try:
            with open(mouth_path, 'rb') as f:
                mouth_list_cycle = pickle.load(f)
            if len(mouth_list_cycle) != len(coord_list_cycle):
                logger.warning(f"[wav2lip] {avatar_id} mouth.pkl 帧数不匹配，回退椭圆掩膜")
                mouth_list_cycle = None
        except Exception as e:
            logger.warning(f"[wav2lip] 读取 {mouth_path} 失败，回退椭圆掩膜: {e}")
            mouth_list_cycle = None

    start, end = _load_frame_range(avatar_id, len(frame_list_cycle))
    if start or end != len(frame_list_cycle):
        frame_list_cycle = frame_list_cycle[start:end]
        face_list_cycle = face_list_cycle[start:end]
        coord_list_cycle = coord_list_cycle[start:end]
        if align_list_cycle is not None:
            align_list_cycle = align_list_cycle[start:end]
        if mouth_list_cycle is not None:
            mouth_list_cycle = mouth_list_cycle[start:end]

    return frame_list_cycle,face_list_cycle,coord_list_cycle,align_list_cycle,mouth_list_cycle

@torch.no_grad()
def warm_up(batch_size,model,modelres):
    # 预热函数
    logger.info('warmup model...')
    img_batch = torch.ones(batch_size, 6, modelres, modelres).to(device)
    mel_batch = torch.ones(batch_size, 1, 80, 16).to(device)
    model(mel_batch, img_batch)

def _build_paste_mask(size: int, mouth_box=None, min_wh=None) -> np.ndarray:
    """生成贴回掩膜（对齐裁剪空间）。

    掩膜要完整盖住「嘴 + 嘴角 + 下唇到下巴」，核心区域保持全不透明，只在外轮廓留
    2~3px 羽化。之前掩膜只有嘴框的 0.6/0.7 且羽化接近嘴高的一半，原帧的旧唇线
    和牙齿会从半透明过渡带里透出来，看起来就是「嘴唇在动、牙齿贴图留在上面」。
    """
    if mouth_box is None:
        mask = np.zeros((size, size), np.float32)
        cv2.ellipse(mask, (size // 2, int(size * 0.68)), (int(size * 0.34), int(size * 0.30)),
                    0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), max(1.5, size * 0.012))
        return np.clip(mask, 0.0, 1.0)

    x1, y1, x2, y2 = [float(v) * size for v in mouth_box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    # 嘴张开的幅度随语音变化，掩膜尺寸取整个形象的嘴部最大值，
    # 否则静音帧的小嘴框会裁掉说话时张大的嘴
    min_w, min_h = min_wh if min_wh else (0.0, 0.0)
    mw = max(x2 - x1, float(min_w))
    mh = max(y2 - y1, float(min_h))
    # 横向盖过嘴角、纵向从鼻底包到下巴；中心略下移，下颌张开也还在掩膜内
    cy += 0.20 * mh
    mask = np.zeros((size, size), np.float32)
    cv2.ellipse(mask, (int(round(cx)), int(round(cy))),
                (int(round(mw * 0.78)), int(round(mh * 1.10))), 0, 0, 360, 1.0, -1)
    # 只羽化轮廓，保证内部 alpha=1，原帧的牙齿和旧唇线不会半透明残留
    mask = cv2.GaussianBlur(mask, (0, 0), max(1.5, size * 0.012))
    return np.clip(mask, 0.0, 1.0)


def _close_small_mouth(pred_bgr: np.ndarray, mouth_box, size: int) -> np.ndarray:
    """把「嘴基本闭合、但生成结果里残留了牙齿纹理」的帧修成干净的闭嘴。

    Wav2Lip 的输入里带完整参考帧，闭口时上下唇之间会残留原帧牙齿的亮纹理，
    看起来就像嘴唇贴图被牙齿「穿透」、缺了一块。这里在嘴部中段取一小块：
    - 亮白像素占比高（张口说话）→ 原样保留牙齿；
    - 只有零星亮斑（嘴是闭着的）→ 用周围唇色修补掉这些亮斑，保留唇形本身。
    """
    if not mouth_box:
        return pred_bgr
    x1, y1, x2, y2 = [float(v) * size for v in mouth_box]
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    hw = max(4, int((x2 - x1) * 0.42))
    hh = max(3, int((y2 - y1) * 0.24))
    ox1, ox2 = max(0, int(cx) - hw), min(size, int(cx) + hw)
    oy1, oy2 = max(0, int(cy) - hh), min(size, int(cy) + hh)
    if oy2 - oy1 < 3 or ox2 - ox1 < 8:
        return pred_bgr

    band = pred_bgr[oy1:oy2, ox1:ox2]
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    white = (hsv[..., 2] > 190) & (hsv[..., 1] < 150)
    if float(white.mean()) > 0.22:
        return pred_bgr  # 张口说话，牙齿是真的
    if float(white.mean()) < 0.01:
        return pred_bgr

    # 闭口时牙齿残影是 1~2px 的细亮纹，用中值滤波抹掉它，同时保留唇形结构：
    # 比整块填色自然，也不会出现方形贴片。
    smoothed = cv2.medianBlur(band, 5)
    h, w = band.shape[:2]
    alpha = np.zeros((h, w), np.float32)
    cv2.ellipse(alpha, (w // 2, h // 2), (max(2, int(w * 0.50)), max(2, int(h * 0.72))),
                0, 0, 360, 1.0, -1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), max(1.2, w * 0.05))[..., None]
    band[:] = np.clip(band.astype(np.float32) * (1 - alpha) + smoothed.astype(np.float32) * alpha,
                      0, 255).astype(np.uint8)
    return pred_bgr


def _build_paste_masks(size: int, mouths) -> list:
    """按整段形像的嘴部框预生成逐帧掩膜；没有嘴部信息时返回 None。"""
    valid = [b for b in mouths if b]
    if not valid:
        return None
    min_w = max(b[2] - b[0] for b in valid) * size
    min_h = max(b[3] - b[1] for b in valid) * size
    masks = []
    for box in mouths:
        mask = _build_paste_mask(size, box, (min_w, min_h)) if box else _build_paste_mask(size)
        masks.append((np.clip(mask, 0.0, 1.0) * 255).astype(np.uint8))
    return masks


@register("avatar", "wav2lip")
class LipReal(BaseAvatar):
    @torch.no_grad()
    def __init__(self, opt, model, avatar):
        super().__init__(opt)

        #self.fps = opt.fps # 20 ms per frame
        
        # self.batch_size = opt.batch_size
        # self.idx = 0
        # self.res_frame_queue = Queue(self.batch_size*2)
        self.model = model

        (self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle,
         self.align_list_cycle, self.mouth_list_cycle) = avatar
        self.paste_mask = None
        self.paste_masks = None
        self.silent_pred_cache = {}
        self.silent_patch_cache = {}
        if self.align_list_cycle is not None and len(self.face_list_cycle):
            size = self.face_list_cycle[0].shape[0]
            if self.mouth_list_cycle is not None:
                self.paste_masks = _build_paste_masks(size, self.mouth_list_cycle)
            if self.paste_masks is None:
                self.paste_mask = _build_paste_mask(size)
            self._warm_silent_preds()

        # ── 讲解独立形象（旧版兼容，仅历史两目录形象触发）────────────
        # 历史版本（形象工作台早期）会把讲解片段单独生成一个形象（{avatar_id}_reply），
        # 并写入基座 avatar.json 的 reply_avatar_id 字段。若存在，则将推理数据源
        # 整体替换为讲解形象，使任何 TTS 说话都在「讲解片段」的姿态/背景上推理对口型。
        # 新版单目录方案创建的形象不再写入 reply_avatar_id（基座推理源已直接由讲解片段
        # 生成），因此不会触发本逻辑；此处仅保证旧形象（如 wav2lip256_avatar7_action）
        # 不回归「说话混用整套视频帧」的行为。
        self.reply_avatar = None
        reply_avatar_id = None
        meta_file = os.path.join('data', 'avatars', opt.avatar_id, 'avatar.json')
        try:
            if os.path.exists(meta_file):
                with open(meta_file, 'r', encoding='utf-8') as f:
                    reply_avatar_id = json.load(f).get('reply_avatar_id')
        except Exception as e:
            logger.warning(f"[wav2lip] 读取 {meta_file} 失败: {e}")
        if reply_avatar_id:
            if os.path.isdir(os.path.join('data', 'avatars', reply_avatar_id)):
                try:
                    self.reply_avatar = load_avatar(reply_avatar_id)
                    (self.frame_list_cycle, self.face_list_cycle, self.coord_list_cycle,
                     self.align_list_cycle, self.mouth_list_cycle) = self.reply_avatar
                    self.paste_mask = None
                    self.paste_masks = None
                    self.silent_pred_cache = {}
                    self.silent_patch_cache = {}
                    if self.align_list_cycle is not None:
                        size = self.face_list_cycle[0].shape[0]
                        if self.mouth_list_cycle is not None:
                            self.paste_masks = _build_paste_masks(size, self.mouth_list_cycle)
                        if self.paste_masks is None:
                            self.paste_mask = _build_paste_mask(size)
                        self._warm_silent_preds()
                    logger.info(f"[wav2lip] 形象 {opt.avatar_id} 加载讲解形象 {reply_avatar_id}，说话自动进讲解")
                except Exception as e:
                    logger.error(f"[wav2lip] 加载讲解形象 {reply_avatar_id} 失败，回退基座推理: {e}")

        self.asr = MelASR(opt,self)
        self.asr.warm_up()
    
    def inference_batch(self, index, audiofeat_batch):
        # 这里的 index 是针对当前 avatar 的索引
        # 返回一个 batch 的推理结果，batch 大小由 self.batch_size 决定
        length = len(self.face_list_cycle)
        img_batch = []
        for i in range(self.batch_size):
            idx = mirror_index(length, index + i)
            face = self.face_list_cycle[idx]
            img_batch.append(face)
        img_batch, audiofeat_batch = np.asarray(img_batch), np.asarray(audiofeat_batch)

        img_masked = img_batch.copy()
        img_masked[:, face.shape[0]//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        audiofeat_batch = np.reshape(audiofeat_batch, [len(audiofeat_batch), audiofeat_batch.shape[1], audiofeat_batch.shape[2], 1])
        
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        audiofeat_batch = torch.FloatTensor(np.transpose(audiofeat_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = self.model(audiofeat_batch, img_batch)
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
        return pred

    def _mask_roi(self, idx: int, w: int, h: int):
        """掩膜（对齐裁剪空间）映射回原帧后的包围盒，只在这块区域做融合。"""
        mask = self.paste_masks[idx] if self.paste_masks is not None else self.paste_mask
        if self.align_list_cycle is None or mask is None:
            return None
        mys, mxs = np.where(mask > 0.02)
        if mys.size == 0:
            return None
        src_pts = np.array([[mxs.min(), mys.min()], [mxs.max() + 1, mys.min()],
                            [mxs.min(), mys.max() + 1], [mxs.max() + 1, mys.max() + 1]], np.float32)
        dst_pts = cv2.transform(src_pts.reshape(-1, 1, 2), self.align_list_cycle[idx]).reshape(-1, 2)
        bx0 = max(0, int(np.floor(dst_pts[:, 0].min())) - 2)
        by0 = max(0, int(np.floor(dst_pts[:, 1].min())) - 2)
        bx1 = min(w, int(np.ceil(dst_pts[:, 0].max())) + 2)
        by1 = min(h, int(np.ceil(dst_pts[:, 1].max())) + 2)
        if bx1 <= bx0 or by1 <= by0:
            return None
        return bx0, by0, bx1, by1

    def paste_back_frame(self,pred_frame,idx:int):
        bbox = self.coord_list_cycle[idx]
        combine_frame = self.frame_list_cycle[idx].copy()
        # 对齐模式：pred 是「端正的 256 人脸」，按当帧头部姿态反向变换回原帧，
        # 只融合嘴部一小块，摇头/偏头时嘴巴位置与口型才不会跑偏，
        # 也不会把原帧露出的牙齿以半透明方式叠到新嘴唇上方。
        mask = self.paste_masks[idx] if self.paste_masks is not None else self.paste_mask
        if self.align_list_cycle is not None and mask is not None:
            h, w = combine_frame.shape[:2]
            matrix = self.align_list_cycle[idx]
            pred = pred_frame.astype(np.uint8)
            if self.mouth_list_cycle is not None and idx < len(self.mouth_list_cycle):
                pred = _close_small_mouth(pred, self.mouth_list_cycle[idx], pred.shape[1])
            if self.sr is not None:
                pred = self.sr.enhance(pred)
            # 只对掩膜覆盖的那一小块做变换和融合：整帧 768x1366 做 float 混合同一帧要搬几十 MB，
            # 25fps 直接被拖垮（静音时尤其明显），按掩膜包围盒裁剪后开销降到几十分之一
            roi = self._mask_roi(idx, w, h)
            if roi is None:
                return combine_frame
            bx0, by0, bx1, by1 = roi
            roi_matrix = matrix.astype(np.float32).copy()
            roi_matrix[0, 2] -= bx0
            roi_matrix[1, 2] -= by0
            rw, rh = bx1 - bx0, by1 - by0
            warped = cv2.warpAffine(pred, roi_matrix, (rw, rh), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
            alpha = cv2.warpAffine(mask, roi_matrix, (rw, rh), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT).astype(np.float32)[..., None] / 255.0
            region = combine_frame[by0:by1, bx0:bx1].astype(np.float32)
            blended = region * (1.0 - alpha) + warped.astype(np.float32) * alpha
            combine_frame[by0:by1, bx0:bx1] = np.clip(blended, 0, 255).astype(np.uint8)
            return combine_frame

        y1, y2, x1, x2 = bbox
        res_frame = cv2.resize(pred_frame.astype(np.uint8),(x2-x1,y2-y1))
        if self.sr is not None:
            res_frame = self.sr.enhance(res_frame)
        combine_frame[y1:y2, x1:x2] = res_frame
        return combine_frame

    def _warm_silent_preds(self):
        """批量预热「闭嘴」人脸，静音帧直接取缓存。

        这部分推理之前是放在输出线程里一帧一帧现算的，会和正式推理抢 GPU，播放就卡；
        但它也不能异步——异步时刚开播的那批静音帧还没算好，会退回原始露齿笑帧，
        说话/静音交替时就会看到「嘴唇和牙齿贴图重叠」。所以在初始化时同步算完，
        141 帧分 16 帧一批，通常几百毫秒。
        """
        try:
            faces = self.face_list_cycle
            total = len(faces)
            for start in range(0, total, 16):
                idxs = [mirror_index(total, start + i) for i in range(min(16, total - start))]
                batch = np.asarray([faces[i] for i in idxs])
                img_masked = batch.copy()
                img_masked[:, batch.shape[1] // 2:] = 0
                img_batch = np.concatenate((img_masked, batch), axis=3) / 255.
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(
                    np.zeros((len(idxs), 80, 16, 1), np.float32), (0, 3, 1, 2))).to(device)
                with torch.no_grad():
                    pred = self.model(mel_batch, img_batch)
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                for k, i in enumerate(idxs):
                    self.silent_pred_cache[i] = pred[k]
            self._build_silent_patches()
            logger.info(f"[wav2lip] 静音闭嘴帧预热完成，共 {len(self.silent_pred_cache)} 帧")
        except Exception as e:
            logger.warning(f"[wav2lip] 静音闭嘴帧预热失败，静音时回退原帧: {e}")

    def _build_silent_patches(self):
        """把静音帧的嘴部融合结果预先生成成「小补丁」，静默时只做一次内存拷贝。

        静默帧的内容是固定的（同一个 idx 的合成结果恒定），如果每帧现算变换 + 浮点混合，
        输出线程 25fps 的节奏会被这点开销拖出抖动。这里一次性算好，运行时只保留嘴部
        包围盒内的像素（每帧约几十 KB），开销与不做合成时基本一致。
        """
        self.silent_patch_cache = {}
        if self.align_list_cycle is None:
            return
        for idx, pred in self.silent_pred_cache.items():
            try:
                composed = self.paste_back_frame(pred, idx)
                h, w = composed.shape[:2]
                roi = self._mask_roi(idx, w, h)
                if roi is None:
                    continue
                bx0, by0, bx1, by1 = roi
                self.silent_patch_cache[idx] = (bx0, by0, composed[by0:by1, bx0:bx1].copy())
            except Exception:
                continue

    def apply_silent_frame(self, frame, idx: int):
        """静音段输出：嘴部走同一套合成，避免露出底视频的原始笑容和牙齿。

        只读预热好的补丁缓存，不在输出线程里做推理/变换；缓存缺失时回退原帧。
        """
        patch = self.silent_patch_cache.get(idx)
        if patch is not None:
            bx0, by0, region = patch
            base = self.apply_sr_to_frame(frame, idx)
            out = frame.copy() if base is frame else base
            out[by0:by0 + region.shape[0], bx0:bx0 + region.shape[1]] = region
            return out
        if self.align_list_cycle is None or self.paste_masks is None:
            return self.apply_sr_to_frame(frame, idx)
        pred = self.silent_pred_cache.get(idx)
        if pred is None:
            return self.apply_sr_to_frame(frame, idx)
        try:
            return self.paste_back_frame(pred, idx)
        except Exception as e:
            logger.warning(f"[wav2lip] 静音帧合成失败，回退原帧: {e}")
            return self.apply_sr_to_frame(frame, idx)