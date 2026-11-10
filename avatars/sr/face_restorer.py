###############################################################################
#  人脸超分/复原（GFPGANv1.4）
#
#  用途：数字人推理把低分辨率模型输出（wav2lip 96/256、musetalk 256、
#        ultralight 160）双线性放大贴回原始人脸 bbox，细节会明显发虚。
#        开启 --use_sr 后，在贴回前用 GFPGAN 做一次人脸复原/超分。
#
#  后端：
#    - onnx : models/onnx/gfpgan/GFPGANv1.4.onnx（onnxruntime，优先 TensorRT，其次 CUDA EP）
#    - torch: models/common/gfpgan/GFPGANv1.4.pth（如沐曦等 onnxruntime 无 GPU 版时使用）
#    - auto : 跟随 --use_onnx（开了先试 onnx+CUDA，失败退回 torch；没开则 torch 优先）
#
#  默认关闭；任一环节失败都只记录日志并降级为「不超分」，不影响主流程。
###############################################################################

import os
import threading
import time

import cv2
import numpy as np

from utils.logger import logger

DEFAULT_SR_ONNX_PATH = './models/onnx/gfpgan/GFPGANv1.4.onnx'
DEFAULT_SR_PT_PATH = './models/common/gfpgan/GFPGANv1.4.pth'

_restorer = None
_restorer_lock = threading.Lock()
_restorer_init_failed = False


def _preprocess(face_bgr, face_size):
    # blobFromImage 一次完成 resize + BGR→RGB + 归一化，比手写 resize/cvtColor/乘加快约 3 倍
    return cv2.dnn.blobFromImage(
        face_bgr, 1.0 / 127.5, (face_size, face_size), (127.5, 127.5, 127.5),
        swapRB=True, crop=False)


def _postprocess(y, size):
    h, w = size
    # convertScaleAbs 一次完成 (x+1)*127.5 与裁剪/取整，比 clip+乘加+astype 快约 2 倍
    img = cv2.convertScaleAbs(np.ascontiguousarray(y.transpose(1, 2, 0)), alpha=127.5, beta=127.5)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if (h, w) != img.shape[:2]:
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return img


def _blend(enhanced, original, strength):
    """强度融合：out = s*enhanced + (1-s)*original（等价 CodeFormer 的 fidelity weight）。

    s=0 完全不超分（保留原图），s=1 为 GFPGAN 全量输出。
    """
    if strength >= 1.0:
        return enhanced
    if strength <= 0.0:
        return original.copy()
    return cv2.addWeighted(enhanced, strength, original, 1.0 - strength, 0)


class GFPGANOnnxRestorer:
    """GFPGANv1.4 ONNX 封装：输入/输出均为 BGR uint8，尺寸保持不变。"""

    backend = 'onnx'

    def __init__(self, model_path, face_size=512, strength=0.6, use_trt=False, trt_cache_dir=None):
        import onnxruntime as ort

        from utils.device import get_onnx_providers

        self.face_size = face_size
        self._strength = min(1.0, max(0.0, float(strength)))
        self.provider = 'CPUExecutionProvider'

        # Windows 下 onnxruntime-gpu 依赖 cuDNN/CUDA 的 DLL，而它们通常由 torch 带进进程；
        # 若进程里还没人加载过，CUDA EP 会因找不到 cudnn 而静默退化成 CPU（~1.2s/张）。
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            try:
                import torch  # noqa: F401
            except Exception:
                pass

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # GFPGAN 输入固定 [1,3,512,512]，不需要 TRT profile
        providers = get_onnx_providers(
            use_trt=use_trt, trt_cache_dir=trt_cache_dir, trt_tag='gfpgan')

        started = time.time()
        if any(isinstance(p, tuple) for p in providers):
            logger.info("[SR] 正在构建/加载 GFPGAN TensorRT 引擎（首次约 1~2 分钟，之后走缓存）…")
        try:
            self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
        except Exception as e:
            fallback = []
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                fallback.append('CUDAExecutionProvider')
            fallback.append('CPUExecutionProvider')
            logger.warning("[SR] onnxruntime GPU 初始化失败（%s），回退 %s", e, fallback)
            try:
                self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=fallback)
            except Exception as e2:
                logger.warning("[SR] onnxruntime 回退失败（%s），强制 CPU", e2)
                self.session = ort.InferenceSession(
                    model_path, sess_options=sess_options, providers=['CPUExecutionProvider'])

        used = self.session.get_providers()
        self.provider = used[0] if used else 'CPUExecutionProvider'
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info("[SR] GFPGAN(ONNX) 已加载: %s (provider=%s, %.1fs)",
                    model_path, self.provider, time.time() - started)

    @property
    def has_gpu(self):
        return self.provider != 'CPUExecutionProvider'

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, value):
        self._strength = min(1.0, max(0.0, float(value)))

    def enhance(self, face_bgr: np.ndarray) -> np.ndarray:
        h, w = face_bgr.shape[:2]
        if h < 16 or w < 16:
            return face_bgr
        x = _preprocess(face_bgr, self.face_size)
        y = self.session.run([self.output_name], {self.input_name: x})[0][0]
        return _blend(_postprocess(y, (h, w)), face_bgr, self._strength)


class GFPGANTorchRestorer:
    """GFPGANv1.4 PyTorch 封装：固定跑 GPU，禁止静默回退 CPU。"""

    backend = 'torch'

    def __init__(self, model_path, face_size=512, allow_cpu=False, strength=0.6):
        import torch

        from avatars.sr.gfpgan_arch import GFPGANExportWrapper, build_gfpgan_v14

        self.torch = torch
        self.face_size = face_size
        self._strength = min(1.0, max(0.0, float(strength)))
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.backends.cudnn.benchmark = True  # 输入尺寸固定，启用 cuDNN 自动调优
        elif allow_cpu:
            self.device = torch.device('cpu')
            logger.warning("[SR] 未检测到可用 GPU，超分将在 CPU 上运行（--sr_allow_cpu）")
        else:
            raise RuntimeError("未检测到可用 GPU；如确需 CPU 超分请加 --sr_allow_cpu")
        self.provider = str(self.device)

        started = time.time()
        model = build_gfpgan_v14(out_size=face_size)
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dict = checkpoint.get('params_ema', checkpoint.get('params', checkpoint))
        model.load_state_dict(state_dict, strict=True)
        # 噪声固定为 0，保证逐帧输出稳定
        for buf in model.stylegan_decoder.noises.buffers():
            buf.zero_()
        self.model = GFPGANExportWrapper(model).eval().to(self.device)

        if self.device.type == 'cuda':
            logger.info("[SR] GFPGAN(torch) 已加载: %s (device=%s, gpu=%s, %.1fs)",
                        model_path, self.device, torch.cuda.get_device_name(0), time.time() - started)
        else:
            logger.info("[SR] GFPGAN(torch) 已加载: %s (device=%s, %.1fs)",
                        model_path, self.device, time.time() - started)

    @property
    def has_gpu(self):
        return self.device.type != 'cpu'

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, value):
        self._strength = min(1.0, max(0.0, float(value)))

    def enhance(self, face_bgr: np.ndarray) -> np.ndarray:
        h, w = face_bgr.shape[:2]
        if h < 16 or w < 16:
            return face_bgr
        x = self.torch.from_numpy(_preprocess(face_bgr, self.face_size)).to(self.device, non_blocking=True)
        with self.torch.inference_mode():
            y = self.model(x)[0].float().cpu().numpy()
        return _blend(_postprocess(y, (h, w)), face_bgr, self._strength)


def _torch_cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _build_restorer(opt):
    backend = (getattr(opt, 'sr_backend', 'auto') or 'auto').lower()
    allow_cpu = bool(getattr(opt, 'sr_allow_cpu', False))
    onnx_path = getattr(opt, 'sr_model_path', '') or DEFAULT_SR_ONNX_PATH
    pt_path = getattr(opt, 'sr_pt_model_path', '') or DEFAULT_SR_PT_PATH
    strength = float(getattr(opt, 'sr_strength', 0.6))
    use_trt = bool(getattr(opt, 'use_trt', False))
    trt_cache_dir = getattr(opt, 'trt_cache_dir', None)

    def _onnx():
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f'ONNX 超分模型不存在: {onnx_path}')
        restorer = GFPGANOnnxRestorer(onnx_path, strength=strength,
                                      use_trt=use_trt, trt_cache_dir=trt_cache_dir)
        if not restorer.has_gpu and not allow_cpu:
            raise RuntimeError(
                f'onnxruntime 未使用 GPU（provider={restorer.provider}）；如确需 CPU 超分请加 --sr_allow_cpu')
        return restorer

    def _torch():
        if not os.path.exists(pt_path):
            raise FileNotFoundError(f'PyTorch 超分模型不存在: {pt_path}')
        return GFPGANTorchRestorer(pt_path, allow_cpu=allow_cpu, strength=strength)

    if backend == 'onnx':
        return _onnx()
    if backend in ('torch', 'pt'):
        return _torch()

    # auto：跟随主模型的 --use_onnx。开了 --use_onnx 就先走 ONNX（要求 onnxruntime 有 GPU），
    # 拿不到 GPU 再退回 torch；没开则 torch 有 GPU 时用 torch，否则试 ONNX。
    if bool(getattr(opt, 'use_onnx', False)):
        try:
            return _onnx()
        except Exception as e:
            logger.warning("[SR] ONNX GPU 后端不可用（%s），回退 torch 后端", e)
        return _torch()

    if _torch_cuda_available():
        try:
            return _torch()
        except Exception as e:
            logger.warning("[SR] torch GPU 后端不可用（%s），尝试 ONNX 后端", e)
    try:
        return _onnx()
    except Exception as e:
        logger.warning("[SR] ONNX 后端不可用（%s），尝试 torch 后端", e)
        return _torch()


def get_face_restorer(opt):
    """按 opt.use_sr 返回全局单例；关闭或初始化失败时返回 None。"""
    global _restorer, _restorer_init_failed

    if not getattr(opt, 'use_sr', False):
        return None
    if _restorer is not None:
        return _restorer
    with _restorer_lock:
        if _restorer is not None:
            return _restorer
        if _restorer_init_failed:
            return None
        try:
            _restorer = _build_restorer(opt)
        except Exception:
            logger.exception("[SR] 超分模型初始化失败，已跳过超分")
            _restorer_init_failed = True
            return None
    return _restorer


def get_active_restorer():
    """返回当前已加载的全局超分实例（未启用/未加载时为 None）。"""
    return _restorer


def set_sr_strength(value):
    """运行时调整超分强度（0~1）；返回设置后的实际值，未启用超分时返回 None。"""
    restorer = _restorer
    if restorer is None:
        return None
    try:
        restorer.strength = float(value)
    except Exception:
        logger.exception("[SR] 设置超分强度失败: %r", value)
        return None
    logger.info("[SR] 超分强度已设为 %.2f", restorer.strength)
    return restorer.strength
