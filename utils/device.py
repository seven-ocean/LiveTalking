import os

import torch
import warnings


DEFAULT_TRT_CACHE_DIR = './models/onnx/trt_cache'


def _is_metax():
    """检测是否为沐曦 Metax GPU"""
    if not torch.cuda.is_available():
        return False
    try:
        name = torch.cuda.get_device_name(0).lower()
        return 'metax' in name or 'maca' in name
    except Exception:
        return False


def initialize_device():
    if torch.cuda.is_available():
        # Metax C500 兼容: 某些 Metax 驱动版本在 device_count 上不稳定
        # 强制使用 cuda:0，避免驱动层对 device>0 的检查
        return torch.device('cuda:0')
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def is_metax_gpu():
    return _is_metax()


def tensorrt_available():
    """TensorRT EP 是否可用：onnxruntime 带该 EP，且 tensorrt 运行时能导入。

    import tensorrt 会顺带把 tensorrt_libs 目录（nvinfer_10.dll 等）注册进进程，
    onnxruntime 才能加载 TRT EP；只装了 onnxruntime-gpu 而没装 tensorrt 时在这里返回 False。
    """
    try:
        import onnxruntime as ort
        if 'TensorrtExecutionProvider' not in ort.get_available_providers():
            return False
        import tensorrt  # noqa: F401
        return True
    except Exception:
        return False


def _trt_provider(cache_dir, tag, profiles=None, fp16=False):
    """构造 TensorrtExecutionProvider（fp32 + 引擎/精度缓存，失败自动回退 CUDA）。"""
    cache_path = os.path.join(cache_dir or DEFAULT_TRT_CACHE_DIR, tag)
    os.makedirs(cache_path, exist_ok=True)
    opts = {
        'device_id': 0,
        'trt_max_workspace_size': 4 << 30,
        'trt_fp16_enable': bool(fp16),
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_path,
        'trt_timing_cache_enable': True,
    }
    if profiles:
        opts.update(profiles)
    return ('TensorrtExecutionProvider', opts)


def get_onnx_providers(use_trt=False, trt_cache_dir=None, trt_tag='onnx', trt_profiles=None,
                       trt_fp16=False):
    """
    获取 ONNX Runtime 可用的 GPU ExecutionProvider。

    参数:
        use_trt       : 是否启用 TensorRT（自动检测，不可用时静默跳过）
        trt_cache_dir : TRT 引擎缓存目录（默认 ./models/onnx/trt_cache）
        trt_tag       : 缓存子目录名（按模型区分，如 wav2lip / gfpgan）
        trt_profiles  : 额外的 TRT provider 选项，如动态 batch 的 trt_profile_*_shapes
        trt_fp16      : TRT 是否用 fp16（注意：GFPGAN 上 fp16 会明显掉画质，不要开）

    返回:
        list: 优先使用 GPU provider，最后兜底 CPU
              沐曦 Metax → ['MACAExecutionProvider', 'CPUExecutionProvider']
              NVIDIA     → [('TensorrtExecutionProvider', {...}), 'CUDAExecutionProvider', 'CPUExecutionProvider']
              纯 CPU     → ['CPUExecutionProvider']
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        return ['CPUExecutionProvider']

    if 'CUDAExecutionProvider' in available:
        providers = []
        if use_trt and tensorrt_available():
            providers.append(_trt_provider(trt_cache_dir, trt_tag, trt_profiles, trt_fp16))
        providers += ['CUDAExecutionProvider', 'CPUExecutionProvider']
        return providers

    if 'MACAExecutionProvider' in available:
        return ['MACAExecutionProvider', 'CPUExecutionProvider']

    if 'ROCMExecutionProvider' in available:
        return ['ROCMExecutionProvider', 'CPUExecutionProvider']

    return ['CPUExecutionProvider']
