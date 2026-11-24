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

# server.py
from flask import Flask, render_template,send_from_directory,request, jsonify
#from flask_sockets import Sockets
import base64
import json
#import gevent
#from gevent import pywsgi
#from geventwebsocket.handler import WebSocketHandler
import re
import os
import signal
import faulthandler
import numpy as np

# 诊断：kill -USR1 <pid> 时把所有线程的 Python 栈打到 stderr（日志）。
# 事件循环被卡住时这是唯一能定位阻塞点的手段；无副作用。
try:
    faulthandler.register(signal.SIGUSR1, all_threads=True)
except Exception:
    pass
from threading import Thread,Event,Lock
#import multiprocessing
import torch.multiprocessing as mp

from aiohttp import web
import aiohttp
import aiohttp_cors
from aiortc import RTCPeerConnection, RTCSessionDescription,RTCIceServer,RTCConfiguration
from aiortc.rtcrtpsender import RTCRtpSender
from server.webrtc import HumanPlayer
from avatars.base_avatar import BaseAvatar
from llm import llm_response
import registry
from server.routes import setup_routes
from server.rtc_manager import RTCManager
from server.session_manager import session_manager
from server.model_switch import ModelSwitchRequired  # [fix-publish-webrtc]:0693f81638

import argparse
import random
import shutil
import time
import asyncio
import importlib
import torch
import os
from io import BytesIO
from typing import Dict
from utils.logger import logger
import copy
import gc
from dotenv import load_dotenv

from utils.hardware_monitor import log_device_info, log_resource_usage, start_monitoring, get_gpu_memory


app = Flask(__name__)
#sockets = Sockets(app)
opt = None
model = None
global_avatars = {} # avatar_id: payload

# ─── 模型运行时（单进程多模型）────────────────────────────────────
# 模型名 -> 插件模块（从 main() 内提升到模块级，供懒加载复用）
_AVATAR_MODULES = {
    'musetalk':   'avatars.musetalk_avatar',
    'wav2lip':    'avatars.wav2lip_avatar',
    'ultralight': 'avatars.ultralight_avatar',
}
# 已就绪的模型运行时：model_name -> {"model": <weights>, "load_avatar": fn}
_model_runtimes: Dict[str, dict] = {}
_model_load_lock = Lock()  # 保证同一模型只加载一次（并发去重）
# 模型最近一次有会话使用的时间：供「空闲模型释放」判断（见 _release_idle_model_runtimes）
_model_last_used: Dict[str, float] = {}

# ─── avatar 加载状态注册表（线程安全）────────────────────────────────
# avatar_id -> {"status": "loading"|"ready"|"error", "startedAt": float,
#               "elapsedMs": int|None, "error": str|None}
_avatar_state: Dict[str, dict] = {}
_avatar_state_lock = Lock()
_avatar_load_lock = Lock()  # 保证同一时刻只有一个 avatar 冷加载（并发去重）

# ─── 形象缓存策略 ───────────────────────────────────────────────
# 模型权重按模型常驻（_model_runtimes），所有形象共享：A 切到 B 不会重新加载
# 权重、也不会多占显存，只是换一份形象帧数据（存在内存里）。这里只对帧数据做
# LRU 淘汰，默认保留最近用过的 2 个形象；0 表示不限制。
_AVATAR_CACHE_MAX = int(os.getenv("ZL_AVATAR_CACHE_MAX", "2"))
_avatar_last_used: Dict[str, float] = {}
_avatar_evict_lock = Lock()

# ─── 进程级模型切换 ─────────────────────────────────────────────
# app.py 只加载一个模型；切换到另一个模型时写目标 avatar 后以 42 退出，
# 由 run_talking.sh 重启新进程。这样 GPU/CUDA 上下文由 OS 彻底回收。
_PROCESS_SWITCH = os.getenv("ZL_PROCESS_SWITCH", "0") == "1"
_DEFER_MODEL_LOAD = os.getenv("ZL_DEFER_MODEL_LOAD", "0") == "1"
_SWITCH_TARGET_FILE = os.getenv("ZL_SWITCH_TARGET_FILE", "")

# ─── 动态模型切换（auto）─────────────────────────────────────────
# keep：始终进程内共存；restart/release：换模型就重启；
# auto：显存充足时 keep，free < max(TALK_GPU_RESERVE_MB, total*TALK_GPU_RESERVE_PCT)
#       且需要加载另一个模型时，写目标 avatar 后 rc=42 让 wrapper 重启，避免 OOM。
_MODEL_SWITCH_MODE = (os.getenv("TALK_MODEL_SWITCH_MODE", "auto") or "auto").strip().lower()
_TALK_GPU_RESERVE_MB = float(os.getenv("TALK_GPU_RESERVE_MB", "0") or 0)
_TALK_GPU_RESERVE_PCT = float(os.getenv("TALK_GPU_RESERVE_PCT", "0.15") or 0.15)
logger.info(
    "[auto-switch] mode=%s reserve_mb=%.0f reserve_pct=%.2f",
    _MODEL_SWITCH_MODE, _TALK_GPU_RESERVE_MB, _TALK_GPU_RESERVE_PCT,
)


#####webrtc###############################
# rtc_manager replaces the old pcs set and duplicate offer handlers.
rtc_manager = None

def randN(N)->int:
    '''生成长度为 N的随机数 '''
    min = pow(10, N - 1)
    max = pow(10, N)
    return random.randint(min, max - 1)

def _update_avatar_state(avatar_id: str, **fields):
    """线程安全地更新 avatar 加载状态并返回最新快照。"""
    with _avatar_state_lock:
        st = dict(_avatar_state.get(avatar_id) or {})
        st.update(fields)
        _avatar_state[avatar_id] = st
        return st

def _get_avatar_state(avatar_id: str) -> dict:
    with _avatar_state_lock:
        return dict(_avatar_state.get(avatar_id) or {})

def resolve_avatar_model(avatar_id: str) -> str:
    """解析形象所属模型：优先 avatar.json.model，其次按目录内容推断，最后回退 opt.model。"""
    avatar_dir = os.path.join('data', 'avatars', avatar_id)
    meta_file = os.path.join(avatar_dir, 'avatar.json')
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                m = (json.load(f) or {}).get('model')
            if m:
                return m
        except Exception:
            logger.warning("读取形象元信息失败，回退目录推断: %s", meta_file)
    # 目录内容推断（兼容仅 avator_info.json / 命令行创建的历史形象）
    if os.path.exists(os.path.join(avatar_dir, 'latents.pt')) or \
       os.path.isdir(os.path.join(avatar_dir, 'mask')):
        return 'musetalk'
    if os.path.isdir(os.path.join(avatar_dir, 'face_imgs')):
        return 'wav2lip'
    return opt.model

def _resolve_onnx_paths(model_name: str, opt):
    """按模型解析 ONNX 路径（避免多模型共用单一 opt.onnx_model_path 而相互覆盖）。"""
    onnx_path = getattr(opt, 'onnx_model_path', None)
    vae_path = getattr(opt, 'vae_onnx_path', None)
    if getattr(opt, 'use_onnx', False) and not onnx_path:
        if model_name == 'musetalk':
            onnx_path = './models/onnx/musetalk/unet.onnx'
            vae_path = './models/onnx/musetalk/vae_decoder.onnx'
        elif model_name == 'wav2lip':
            onnx_path = './models/onnx/wav2lip/wav2lip.onnx'
    return onnx_path, vae_path

def _load_model_weights(model_name: str, avatar_mod, opt):
    """加载指定模型权重（迁移自原 main() 的 if/elif 分支）。"""
    onnx_path, vae_path = _resolve_onnx_paths(model_name, opt)
    if model_name == 'musetalk':
        if getattr(opt, 'use_onnx', False):
            return avatar_mod.load_onnx_model(onnx_path, vae_path)
        return avatar_mod.load_model()
    if model_name == 'wav2lip':
        if getattr(opt, 'use_onnx', False):
            return avatar_mod.load_onnx_model(onnx_path)
        return avatar_mod.load_model("./models/wav2lip/wav2lip.pth")
    # ultralight
    return avatar_mod.load_model(opt)

def _gpu_free_total_mb():
    """返回 (free_mb, total_mb)；读不到时返回 (None, None)。"""
    try:
        mem = get_gpu_memory()
        if mem:
            return float(mem.get("free_mb") or 0), float(mem.get("total_mb") or 0)
    except Exception:
        logger.debug("读取 GPU 显存失败", exc_info=True)
    return None, None


def _gpu_reserve_mb(total_mb: float) -> float:
    return max(_TALK_GPU_RESERVE_MB, total_mb * _TALK_GPU_RESERVE_PCT)


def _gpu_low_for_model_load() -> bool:
    """当前显存是否低于安全余量；读不到显存时保守返回 False（不额外重启）。"""
    free_mb, total_mb = _gpu_free_total_mb()
    if free_mb is None or total_mb <= 0:
        return False
    reserve = _gpu_reserve_mb(total_mb)
    low = free_mb < reserve
    logger.info(
        "[auto-switch] GPU free=%.0fMB total=%.0fMB reserve=%.0fMB low=%s",
        free_mb, total_mb, reserve, low,
    )
    return low


def _needs_model_switch(avatar_id: str) -> bool:
    """是否需要进程级切换模型。

    - restart/release：换模型就退出重启（原低配机逻辑）。
    - auto：显存充足 keep；free 低于安全余量且要加载另一个模型时，
      写目标 avatar 后 rc=42 重启，避免把 SoVITS/ASR 挤到 OOM。
    - keep：不切换，进程内共存。
    """
    model_name = resolve_avatar_model(avatar_id)
    if not _model_runtimes:
        # 进程刚启动、还没有加载任何模型：先加载，不做切换。
        return False
    if model_name in _model_runtimes:
        # 同一个模型，权重共享，不需要重启。
        return False
    if _PROCESS_SWITCH or _MODEL_SWITCH_MODE in ("restart", "release"):
        return True
    if _MODEL_SWITCH_MODE == "auto":
        if _gpu_low_for_model_load():
            logger.info(
                "[auto-switch] 显存不足，模型切换走进程重启: loaded=%s target=%s",
                list(_model_runtimes.keys()), model_name,
            )
            return True
        logger.info(
            "[auto-switch] 显存充足，keep 模式加载新模型: %s", model_name,
        )
        return False
    return False



def _write_switch_target(avatar_id: str) -> None:
    if _SWITCH_TARGET_FILE:
        try:
            with open(_SWITCH_TARGET_FILE, "w", encoding="utf-8") as f:
                f.write(avatar_id)
        except Exception:
            logger.exception("写入进程级切换目标失败")
    logger.info("请求进程级切换模型: avatar=%s", avatar_id)


def _release_other_models(keep_model: str) -> None:
    """切换模型前释放旧模型资源：停旧会话、清形象缓存、删旧模型 runtime。

    Wav2Lip 和 MuseTalk 的权重/推理上下文不能在同一进程里长期共存；
    切换过去时先把旧模型相关的会话、形象帧和权重引用全部释放，再加载新模型。
    """
    if not _PROCESS_SWITCH:
        # keep 模式：客户高配机，两个模型保留并发，不做释放/重启
        return
    released = []
    for sid, sess in list(session_manager.sessions.items()):
        model = getattr(getattr(sess, "opt", None), "model", None)
        if model and model != keep_model:
            try:
                session_manager.stop_session(sid)
                session_manager.remove_session(sid)
                released.append(f"session:{sid}")
            except Exception:
                logger.exception("释放旧会话失败: %s", sid)
    for aid in list(global_avatars):
        try:
            if resolve_avatar_model(aid) != keep_model:
                global_avatars.pop(aid, None)
                _avatar_last_used.pop(aid, None)
                with _avatar_state_lock:
                    _avatar_state.pop(aid, None)
                released.append(f"avatar:{aid}")
        except Exception:
            pass
    for model in list(_model_runtimes):
        if model != keep_model:
            _model_runtimes.pop(model, None)
            released.append(f"model:{model}")
    if released:
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            logger.exception("empty_cache 失败")
        logger.info("切换模型释放旧资源: %s", ", ".join(released))


def _warmup_resolution(model_name: str) -> int:
    """warmup 分辨率由模型结构决定；--modelres 只做兼容提示，避免误设导致 warmup 失败。"""
    expected = 256 if model_name == 'wav2lip' else 160 if model_name == 'ultralight' else getattr(opt, 'modelres', 192)
    configured = getattr(opt, 'modelres', None)
    if configured and configured != expected:
        logger.warning(
            'modelres=%s is ignored for %s warmup; using %s (model input is fixed)',
            configured, model_name, expected,
        )
    return expected


def ensure_model_runtime(model_name: str) -> dict:
    """确保某模型的权重与 load_avatar 就绪（懒加载 + 常驻缓存，并发去重）。"""
    rt = _model_runtimes.get(model_name)
    if rt is not None:
        return rt
    with _model_load_lock:
        rt = _model_runtimes.get(model_name)
        if rt is not None:
            return rt
        if model_name not in _AVATAR_MODULES:
            raise ValueError(f"未知模型: {model_name}")
        _release_other_models(model_name)
        avatar_mod = importlib.import_module(_AVATAR_MODULES[model_name])
        logger.info("懒加载模型运行时: %s", model_name)
        weights = _load_model_weights(model_name, avatar_mod, opt)
        # warm_up 签名各模型不同；ultralight 的 warm_up 依赖形象数据，延后到 main() 处理（保持原行为）
        if model_name == 'wav2lip':
            # Wav2Lip 的 warmup 必须保留，不能跳过；只对 MuseTalk 做延迟 warmup
            avatar_mod.warm_up(opt.batch_size, weights, _warmup_resolution('wav2lip'))
        elif model_name == 'musetalk':
            # 跳过 warmup：MuseTalk 的 warmup 会一次性分配大块内存/显存，
            # 在 16GB sGPU 配额下会直接把进程打掉；首次真实请求再按需分配。
            logger.info("MuseTalk warmup skipped (deferred to first inference)")
        rt = {"model": weights, "load_avatar": avatar_mod.load_avatar}
        _model_runtimes[model_name] = rt
        _model_last_used[model_name] = time.time()
        logger.info("模型运行时就绪: %s", model_name)
        return rt

def get_or_load_avatar(avatar_id: str):
    """获取 avatar（缓存优先）；未缓存则加载。

    并发调用同一 avatar 时由 _avatar_load_lock 串行去重，只有一个真正执行
    load_avatar，其余在拿到锁后发现缓存已就绪直接复用。
    """
    if _needs_model_switch(avatar_id):
        _write_switch_target(avatar_id)
        # [fix-publish-webrtc]:f6e9ef8fda 请求线程里不能杀进程，交给 RTCManager 返回 503 后再退出
        raise ModelSwitchRequired(avatar_id)
    cached = global_avatars.get(avatar_id)
    if cached is not None:
        _avatar_last_used[avatar_id] = time.time()
        return cached
    with _avatar_load_lock:
        cached = global_avatars.get(avatar_id)
        if cached is not None:
            _avatar_last_used[avatar_id] = time.time()
            return cached
        # 并发加载不同模型时，第一个模型可能刚加载完；拿到锁后再判断一次，
        # 避免 keep 模式下同时加载两套模型把显存顶爆。
        if _needs_model_switch(avatar_id):
            _write_switch_target(avatar_id)
            # [fix-publish-webrtc]:026e07ad4d 同上：拿到锁后再判一次时也不能杀进程
            raise ModelSwitchRequired(avatar_id)
        started = time.time()
        model_name = resolve_avatar_model(avatar_id)
        _update_avatar_state(avatar_id, status="loading", startedAt=started,
                             elapsedMs=None, error=None, model=model_name)
        try:
            rt = ensure_model_runtime(model_name)
            avatar = rt["load_avatar"](avatar_id)
            global_avatars[avatar_id] = avatar
            _avatar_last_used[avatar_id] = time.time()
            _update_avatar_state(avatar_id, status="ready",
                                 elapsedMs=int((time.time() - started) * 1000))
            _evict_avatars_async(keep=avatar_id)
            return avatar
        except BaseException as e:
            _update_avatar_state(avatar_id, status="error",
                                 elapsedMs=int((time.time() - started) * 1000),
                                 error=str(e))
            raise


def _active_avatar_ids() -> set:
    """当前会话正在使用的形象（建会话中的占位会话不计）。"""
    ids = set()
    for sess in list(session_manager.sessions.values()):
        aid = getattr(getattr(sess, "opt", None), "avatar_id", None)
        if aid:
            ids.add(aid)
    return ids


def _cgroup_mem_limit_mb() -> float:
    """读取容器的内存上限（cgroup v1/v2），读不到返回 0。"""
    for path in ("/sys/fs/cgroup/memory/memory.limit_in_bytes",
                 "/sys/fs/cgroup/memory.max"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = f.read().strip()
            if raw in ("max", ""):
                continue
            value = int(raw)
            if value > 0 and value < (1 << 60):
                return value / 1048576.0
        except Exception:
            continue
    return 0.0


def _process_rss_mb() -> float:
    """当前进程 RSS（MB）。"""
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _evict_avatars(keep: str = "") -> None:
    """形象帧数据按 LRU 淘汰。

    三层保护，避免“时间越久越卡”：
      1) 数量上限 `ZL_AVATAR_CACHE_MAX`；
      2) 内存护栏：RSS 超过 cgroup 上限的 `ZL_AVATAR_MEM_HIGH_PCT` 时只保留 1 套；
      3) 空闲超时：无会话使用且空闲超过 `ZL_AVATAR_IDLE_EVICT_SECONDS` 的形象主动释放。
    淘汰的是内存里的形象帧，不是模型权重；正在会话里用着的形象不淘汰。
    """
    active = _active_avatar_ids()
    dropped = []
    max_keep = _AVATAR_CACHE_MAX
    _t0 = time.monotonic()

    # 2) 内存护栏
    limit_mb = _cgroup_mem_limit_mb()
    if limit_mb > 0:
        try:
            high_pct = float(os.environ.get("ZL_AVATAR_MEM_HIGH_PCT", "80")) / 100.0
        except (TypeError, ValueError):
            high_pct = 0.8
        rss_mb = _process_rss_mb()
        if rss_mb > limit_mb * high_pct and max_keep > 1:
            max_keep = 1
            logger.warning(
                "形象缓存内存护栏触发：RSS %.0fMB / 上限 %.0fMB（%.0f%%），只保留 1 套形象",
                rss_mb, limit_mb, rss_mb * 100.0 / limit_mb,
            )

    # 1) 数量上限
    if max_keep > 0:
        while len(global_avatars) > max_keep:
            candidates = [aid for aid in global_avatars if aid not in active and aid != keep]
            if not candidates:
                break
            victim = min(candidates, key=lambda aid: _avatar_last_used.get(aid) or 0.0)
            global_avatars.pop(victim, None)
            _avatar_last_used.pop(victim, None)
            with _avatar_state_lock:
                _avatar_state.pop(victim, None)
            dropped.append(victim)

    # 3) 空闲超时（默认 300s；0 表示不按空闲淘汰）
    try:
        idle_limit = float(os.environ.get("ZL_AVATAR_IDLE_EVICT_SECONDS", "300"))
    except (TypeError, ValueError):
        idle_limit = 300.0
    if idle_limit > 0:
        now = time.time()
        for aid in list(global_avatars):
            if aid in active or aid == keep:
                continue
            if now - (_avatar_last_used.get(aid) or 0.0) < idle_limit:
                continue
            global_avatars.pop(aid, None)
            _avatar_last_used.pop(aid, None)
            with _avatar_state_lock:
                _avatar_state.pop(aid, None)
            dropped.append(aid)

    if dropped:
        # 不要在这里 gc.collect()：进程里常驻 1~2 套（每套 ~1GB）形象帧，全量 GC
        # 会持有 GIL 把推理/渲染/事件循环一起卡住 1~4 秒，切换后第一帧表现为掉到 15fps。
        # 帧数组是 numpy，pop 掉引用后靠引用计数立即释放；循环引用交给解释器自动分代回收。
        logger.info("形象缓存已释放: %s（当前 %d 套，耗时 %.0fms）",
                    ", ".join(dropped), len(global_avatars), (time.monotonic() - _t0) * 1000)


def _evict_avatars_async(keep: str = "") -> None:
    """把形象淘汰放到后台线程，避免压在新会话首帧/模型加载的关键路径上。

    同一时刻只允许一个淘汰任务在跑（非阻塞抢锁），抢不到说明已有任务在收尾。
    """
    def _run():
        if not _avatar_evict_lock.acquire(blocking=False):
            return
        try:
            _evict_avatars(keep=keep)
        except Exception:
            logger.exception("形象缓存后台淘汰失败")
        finally:
            _avatar_evict_lock.release()

    Thread(target=_run, name="avatar-evict", daemon=True).start()


def _release_idle_model_runtimes() -> None:
    """释放长时间没有任何会话使用的模型运行时，把显存还给驱动。

    背景：`TALK_MODEL_SWITCH_MODE=auto` 在显存充足时会让 wav2lip / musetalk
    两套权重常驻（实测 GPU 11.5GB → 27GB，之后即使没有任何会话也不回落）。
    两套模型常驻后新会话只剩 ~4GB 余量，切换后会明显更卡。这里在会话回收任务里
    定期把「无会话 + 空闲超阈值」的模型释放掉；切回旧模型时按懒加载重新加载。
    """
    if not _model_runtimes:
        return
    try:
        idle_limit = float(os.environ.get("ZL_MODEL_IDLE_RELEASE_SECONDS", "180"))
    except (TypeError, ValueError):
        idle_limit = 180.0
    if idle_limit <= 0:
        return
    if not _model_load_lock.acquire(blocking=False):
        return  # 正在加载模型，下一轮再判断
    try:
        in_use = set()
        for sess in list(session_manager.sessions.values()):
            if sess is None:
                continue
            opt = getattr(sess, "opt", None)
            model = getattr(opt, "model", None)
            if model:
                in_use.add(model)
                continue
            aid = getattr(opt, "avatar_id", None)
            if aid:
                try:
                    in_use.add(resolve_avatar_model(aid))
                except Exception:
                    pass
        now = time.time()
        for model in in_use:
            _model_last_used[model] = now

        released = []
        for model in list(_model_runtimes):
            if model in in_use:
                continue
            last = _model_last_used.get(model)
            if last is None:
                _model_last_used[model] = now
                continue
            if now - last < idle_limit:
                continue
            _model_runtimes.pop(model, None)
            _model_last_used.pop(model, None)
            for aid in list(global_avatars):
                try:
                    if resolve_avatar_model(aid) != model:
                        continue
                    global_avatars.pop(aid, None)
                    _avatar_last_used.pop(aid, None)
                    with _avatar_state_lock:
                        _avatar_state.pop(aid, None)
                except Exception:
                    pass
            released.append(model)
        if released:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "模型空闲回收: %s（空闲 > %.0fs；剩余常驻 %s）",
                ", ".join(released), idle_limit, list(_model_runtimes.keys()) or "无",
            )
    except Exception:
        logger.exception("模型空闲回收失败")
    finally:
        _model_load_lock.release()


def _safe_load_avatar(avatar_id: str):
    """后台线程入口：吞掉异常，仅记录日志（状态已写入注册表）。"""
    try:
        get_or_load_avatar(avatar_id)
    except BaseException:
        logger.exception("背景预热 avatar 失败: %s", avatar_id)

def ensure_avatar_async(avatar_id: str) -> str:
    """异步预热 avatar：后台线程加载，立即返回当前状态。

    返回 "ready"（已缓存）或 "loading"（已触发/进行中）。幂等：
    重复调用（含并发）不会重复加载。
    """
    if global_avatars.get(avatar_id) is not None:
        return "ready"
    with _avatar_state_lock:
        st = _avatar_state.get(avatar_id) or {}
        if st.get("status") == "loading":
            return "loading"
        _avatar_state[avatar_id] = {
            "status": "loading",
            "startedAt": time.time(),
            "elapsedMs": None,
            "error": None,
        }
    asyncio.get_event_loop().run_in_executor(None, _safe_load_avatar, avatar_id)
    return "loading"

def _session_int(params: dict, key: str, current: int, min_value: int, max_value: int) -> int:
    """读取 /offer 会话参数中的整数；非法/越界时回退启动默认值。"""
    raw = params.get(key)
    if raw is None or raw == '':
        return current
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning('invalid %s=%r, keep %s', key, raw, current)
        return current
    if value < min_value or value > max_value:
        logger.warning('invalid %s=%r (range %s~%s), keep %s', key, raw, min_value, max_value, current)
        return current
    return value


def _session_str(params: dict, *keys: str):
    """读取 /offer 会话参数中的非空字符串，支持驼峰/下划线别名。"""
    for key in keys:
        raw = params.get(key)
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = params.get('avatar') or opt.avatar_id
    opt_this.avatar_id = avatar_id
    ref_audio = _session_str(params, 'refaudio', 'refAudio', 'ref_audio') or ''
    ref_text = _session_str(params, 'reftext', 'refText', 'ref_text') or ''

    # 配置端 Human 能力 defaultParams / 真实人配置 modelsParams 透传到 /offer 的会话参数。
    # 优先级：/offer 显式值 > 能力中心默认值（由 zl-ai 注入）> 启动参数。
    opt_this.batch_size = _session_int(params, 'batch_size', opt_this.batch_size, 1, 64)
    opt_this.l = _session_int(params, 'l', opt_this.l, 0, 30)
    opt_this.m = _session_int(params, 'm', opt_this.m, 0, 30)
    opt_this.r = _session_int(params, 'r', opt_this.r, 0, 30)
    # 分辨率/帧率固定为推理直出：25fps，不做缩放/变速。
    opt_this.fps = 25

    # 音频口径（能力中心「音频口径」）：auto = 生产 48k，WS 侧再按客户端设备率自适应；
    # 显式 48000/44100/16000 时整条播放链按该口径生产（WebRTC 线上仍会被 Opus 转到 48k）。
    raw_rate = params.get('audio_rate', params.get('audioRate'))
    audio_profile = 'auto'
    if raw_rate is not None and str(raw_rate).strip():
        v = str(raw_rate).strip().lower()
        if v in ('auto', '', '0'):
            audio_profile = 'auto'
        elif v in ('48000', '48000hz', '48k', '48000.0'):
            audio_profile = 48000
        elif v in ('44100', '44100hz', '44.1k', '44100.0'):
            audio_profile = 44100
        elif v in ('16000', '16000hz', '16k', '16000.0'):
            audio_profile = 16000
        else:
            logger.warning('audio_rate=%s 不受支持，回落 auto', raw_rate)
    try:
        default_rate = int(os.environ.get('TALK_PLAYBACK_SR', '48000') or 48000)
    except (TypeError, ValueError):
        default_rate = 48000
    opt_this.playback_rate = default_rate if audio_profile == 'auto' else int(audio_profile)
    logger.info('[session] %s audio_rate=%s -> playback=%d', sessionid, audio_profile,
                opt_this.playback_rate)

    # 按形象解析其真实模型，并用该模型的权重/loader 构建会话（支持单进程跨模型）
    model_name = resolve_avatar_model(avatar_id)
    rt = ensure_model_runtime(model_name)
    # 缓存优先，未缓存则冷加载（与 /warmup 共用同一去重逻辑）
    avatar_this = get_or_load_avatar(avatar_id)

    # ── 直出：推理出什么画面就发什么画面（不做缩放/变速）─────────────────
    # 2026-11-20 决策：砍掉「画质档位 + FPS 切换」。之前按档位把 540P 定成 20fps、
    # 还把画面缩到 0.75，导致「生产 25fps / 传输 20fps」的队列溢出与口型错位。
    # 现在：分辨率 = 推理帧原始尺寸（scale=1），帧率 = 推理原生 fps，码率取 env。
    def _native_size(data):
        try:
            frames = None
            if isinstance(data, (tuple, list)) and data:
                frames = data[0]
            elif isinstance(data, dict):
                frames = data.get('frame_list') or data.get('frame_list_cycle')
            else:
                frames = getattr(data, 'frame_list_cycle', None)
            if frames:
                h, w = frames[0].shape[:2]
                return int(w), int(h)
        except Exception:
            pass
        return None

    native = _native_size(avatar_this)
    nw, nh = (native if native else (0, 0))
    short = min(nw, nh) if native else 0
    try:
        fps = int(getattr(opt, 'fps', 25) or 25)
    except (TypeError, ValueError):
        fps = 25
    # 能力中心「数字人推理 → WebRTC 视频码率」优先走 CLI（ops.sh 映射）；
    # 客户现场 ops.sh 可能未更新，这里再直接读 zl-talking.launch.json 作为兜底，
    # 避免为了几个码率参数去覆盖客户的启动脚本。
    launch_overrides = {}
    try:
        launch_file = os.environ.get('ZL_TALKING_LAUNCH_FILE')
        if not launch_file:
            _root = os.environ.get('AIHUMAN_ROOT')
            if _root:
                launch_file = os.path.join(_root, 'runtime', 'conf', 'zl-talking.launch.json')
            else:
                launch_file = os.path.join(os.getcwd(), '..', 'runtime', 'conf',
                                           'zl-talking.launch.json')
        with open(launch_file, encoding='utf-8') as _lf:
            _data = json.load(_lf)
        if isinstance(_data, dict):
            launch_overrides = _data
    except Exception:
        launch_overrides = {}

    def _override_int(name, default=0):
        try:
            value = int(getattr(opt, name, 0) or 0)
        except (TypeError, ValueError):
            value = 0
        if value:
            return value
        raw = launch_overrides.get(name)
        if raw is None or str(raw).strip() == '':
            return default
        try:
            return int(raw)
        except (TypeError, ValueError):
            return default

    def _override_float(name, default):
        try:
            value = float(getattr(opt, name, 0) or 0)
        except (TypeError, ValueError):
            value = 0.0
        if value:
            return value
        raw = launch_overrides.get(name)
        if raw is None or str(raw).strip() == '':
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    try:
        max_bitrate = int(os.environ.get('ZL_WEBRTC_MAX_BITRATE', '8000000') or 8000000)
    except (TypeError, ValueError):
        max_bitrate = 8000000
    cli_bitrate_kbps = _override_int('webrtc_bitrate', 0)
    if cli_bitrate_kbps > 0:
        bitrate = cli_bitrate_kbps * 1000
    else:
        try:
            bitrate = int(os.environ.get('ZL_WEBRTC_BITRATE', '2000000') or 2000000)
        except (TypeError, ValueError):
            bitrate = 2000000
    bitrate = max(500000, min(bitrate, max_bitrate))

    cli_min_kbps = _override_int('webrtc_min_bitrate', 0)
    cap_factor = max(1.0, _override_float('webrtc_bitrate_cap_factor', 1.5))
    maxrate_factor = max(1.0, _override_float('webrtc_maxrate_factor', 1.2))
    bufsize_ms = max(50.0, _override_float('webrtc_bufsize_ms', 200.0))
    try:
        env_min = int(os.environ.get('ZL_WEBRTC_H264_MIN_BITRATE', '750000') or 750000)
    except (TypeError, ValueError):
        env_min = 750000
    min_bps = cli_min_kbps * 1000 if cli_min_kbps > 0 else env_min
    min_bps = max(0, min(min_bps, bitrate))

    opt_this.webrtc_scale = 1.0
    opt_this.video_fps = fps
    opt_this.video_bitrate = bitrate
    opt_this.webrtc_min_bitrate = min_bps
    opt_this.webrtc_bitrate_cap_factor = cap_factor
    opt_this.webrtc_maxrate_factor = maxrate_factor
    opt_this.webrtc_bufsize_ms = bufsize_ms
    if native:
        logger.info('[session] %s 直出 video=%dx%d(short=%d) @%.2fMbps %dfps（不缩放/不降帧）',
                    sessionid, nw, nh, short, bitrate / 1e6, fps)
    logger.info(
        '[session] %s 码率护栏: target=%.2fMbps cap=%.2fMbps(%.2fx) min=%.2fMbps '
        'maxrate=%.2fMbps bufsize=%.0fms',
        sessionid, bitrate / 1e6, bitrate * cap_factor / 1e6, cap_factor,
        min_bps / 1e6,
        bitrate * cap_factor * maxrate_factor / 1e6, bufsize_ms)

    # 产出节流模式（URL 试参，只影响本会话；默认 legacy = 原有行为）
    _pacing = str(params.get('pacing') or '').strip().lower()
    opt_this.pacing = _pacing if _pacing in ('legacy', 'official', 'bounded', 'sync25', 'runway') else 'legacy'
    # 发送码率上限（仅 ?bwcap= 显式指定时生效；默认不夹取，走 env 码率）
    try:
        _bwcap = int(float(params.get('bwcap') or 0))
    except (TypeError, ValueError):
        _bwcap = 0
    opt_this.bwcap_kbps = _bwcap if _bwcap > 0 else 0
    # 补齐节拍（重复上一帧）：runway 分支用它把"等帧"变成"补帧"，画面不冻结
    opt_this.frame_pace = (opt_this.pacing == 'runway')
    logger.info('[session] %s 生效参数: video=%dx%d fps=%d scale=%.2f',
                sessionid, nw, nh, opt_this.video_fps, opt_this.webrtc_scale)

    if ref_audio: #请求参数配置了参考音频
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
    custom_config = params.get('custom_config') or params.get('customConfig') or ''
    if custom_config:
        try:
            opt_this.customopt = custom_config if isinstance(custom_config, (list, dict)) else json.loads(custom_config)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            logger.warning('invalid custom_config, ignored: %s', exc)

    logger.info(
        '[session] %s params: avatar=%s, batch_size=%s, fps=%s, l/m/r=%s/%s/%s, scale=%s, refaudio=%s',
        sessionid, avatar_id, opt_this.batch_size, opt_this.fps,
        opt_this.l, opt_this.m, opt_this.r, opt_this.webrtc_scale,
        opt_this.REF_FILE,
    )

    opt_this.model = model_name  # 会话记录真实模型（供 /admin 状态等使用）
    avatar_session = registry.create("avatar", model_name, opt=opt_this, model=rt["model"], avatar=avatar_this)
    return avatar_session

async def offer(request):
    return await rtc_manager.handle_offer(request)


# ─── HTTP-FLV 降级通道（见 server/rtc_manager.handle_flv_publish）─────────
def _flv_response(data=None, msg="ok", status=200):
    body = {"code": 0 if status == 200 else -1, "msg": msg}
    if data is not None:
        body["data"] = data
    return web.Response(status=status, content_type="application/json", text=json.dumps(body))


async def flv_publish(request):
    """浏览器打开降级页时调用：起一路会话并 WHIP 推到本机 SRS。"""
    # 正式环境只走 WebRTC / WebSocket；FLV 仅用于测试，默认关闭，避免测试进程抢 CPU。
    if os.environ.get("FLV_TEST_ENABLED", "0").strip().lower() not in ("1", "true", "yes", "on"):
        return web.json_response(
            {"code": -1, "msg": "FLV 测试链路未启用（FLV_TEST_ENABLED=0）"}, status=404
        )
    body = {}
    if request.can_read_body:
        try:
            body = await request.json()
        except Exception:
            body = {}
    body = body if isinstance(body, dict) else {}
    sessionid = (body.get("sessionid") or "").strip() or None
    try:
        sid = await rtc_manager.handle_flv_publish(body.get("push_url") or "", sessionid)
    except Exception as exc:
        logger.exception("flv_publish failed")
        return _flv_response(msg=str(exc), status=500)
    return _flv_response({"sessionid": sid, "url": "/live/livestream.flv"})


async def flv_ping(request):
    body = await request.json() if request.can_read_body else {}
    sessionid = (body or {}).get("sessionid", "")
    if not rtc_manager.touch_flv_publish(sessionid):
        return _flv_response(msg="publish not found", status=404)
    return _flv_response(rtc_manager.flv_status(sessionid))


async def flv_unpublish(request):
    body = {}
    if request.can_read_body:
        try:
            body = await request.json()
        except Exception:
            body = {}
    sessionid = (body or {}).get("sessionid", "")
    await rtc_manager.stop_flv_publish(sessionid)
    return _flv_response()


# ─── 带宽测试（bw-test.html 使用；走同一个 nginx 端口，测真实链路）──────────────
async def speedtest_ping(request):
    return web.json_response({"ok": True, "ts": time.time()})


async def speedtest_download(request):
    try:
        total = int(request.query.get("bytes", str(32 * 1024 * 1024)))
    except (TypeError, ValueError):
        total = 32 * 1024 * 1024
    total = max(1024 * 1024, min(total, 1024 * 1024 * 1024))
    resp = web.StreamResponse(
        headers={
            "Content-Type": "application/octet-stream",
            "Cache-Control": "no-store",
            "Content-Length": str(total),
        }
    )
    await resp.prepare(request)
    chunk = b"\x00" * 65536
    sent = 0
    try:
        while sent < total:
            n = min(len(chunk), total - sent)
            await resp.write(chunk[:n])
            sent += n
        await resp.write_eof()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    return resp


async def speedtest_upload(request):
    total = 0
    try:
        async for chunk in request.content.iter_chunked(256 * 1024):
            total += len(chunk)
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass
    return web.json_response({"ok": True, "bytes": total})

async def warmup_avatar(request):
    """预热 avatar：前端切换形象前调用，后台异步加载，立即返回，避免阻塞事件循环。

    幂等：重复/并发调用不会重复加载。契约：{code:0, status:"ready"|"loading"}。
    """
    params = await request.json()
    avatar_id = params.get('avatar') or opt.avatar_id
    if _needs_model_switch(avatar_id):
        _write_switch_target(avatar_id)
        asyncio.get_event_loop().call_later(0.3, os._exit, 42)
        return web.Response(content_type="application/json",
                            text=json.dumps({"code": 0, "status": "loading", "switch": True}))
    status = ensure_avatar_async(avatar_id)
    return web.Response(content_type="application/json",
                        text=json.dumps({"code": 0, "status": status}))

async def warmup_status(request):
    """查询 avatar 预热状态。

    契约：{code:0, status:"ready"|"loading"|"error", elapsedMs, error?}
    """
    avatar_id = request.query.get('avatar') or opt.avatar_id
    if global_avatars.get(avatar_id) is not None:
        st_ready = _get_avatar_state(avatar_id)
        resp_ready = {"code": 0, "status": "ready"}
        if st_ready.get("model"):
            resp_ready["model"] = st_ready["model"]
        return web.Response(content_type="application/json",
                            text=json.dumps(resp_ready))
    st = _get_avatar_state(avatar_id)
    if not st:
        # 未触发过预热：顺手触发，避免调用方轮询死等
        ensure_avatar_async(avatar_id)
        st = _get_avatar_state(avatar_id)
    resp = {
        "code": 0,
        "status": st.get("status") or "loading",
        "elapsedMs": st.get("elapsedMs"),
    }
    if st.get("model"):
        resp["model"] = st["model"]
    if st.get("error"):
        resp["error"] = st["error"]
    return web.Response(content_type="application/json", text=json.dumps(resp))

async def on_shutdown(app):
    await rtc_manager.shutdown()

async def download_record(request):
    sessionid = request.match_info.get('sessionid')
    if not sessionid:
        return web.Response(status=400, text="sessionid is required")
    
    record_file = os.path.join('data', 'record', f"{sessionid}.mp4")
    
    if os.path.exists(record_file):
        return web.FileResponse(record_file)
    else:
        return web.Response(status=404, text="Record not found")


def main():
    global rtc_manager, opt, model,load_avatar

    log_device_info()
    start_monitoring(interval=120)

    # 解析命令行参数
    from config import parse_args
    opt = parse_args()

    # 能力中心「输出方式」里的 ws/websocket 只表示客户端走 WS 降级通道，
    # 服务端仍需要 webrtc 输出模块来产出帧/音频（WS 通道从 player 的槽位取帧）。
    # 之前把 ws 直接传进 --transport，base_avatar 找不到输出模块 → process_frames
    # 第一行 self.output.start() 就 AttributeError，整条音视频管线直接不工作。
    # transport 三态：webrtc（直连）/ webrtc-fixed（固定端口，媒体仍走 TURN）/ ws。
    # 服务端输出模块只认 webrtc，这里统一归一化。
    _tr = str(getattr(opt, 'transport', '') or '').lower()
    if _tr in ('webrtc-fixed', 'webrtc_fixed', 'webrtc-tcp', 'webrtc51000'):
        logger.info('transport=%s（固定端口）按 webrtc 输出处理', opt.transport)
        opt.transport = 'webrtc'
    if str(getattr(opt, 'transport', '') or '').lower() in ('ws', 'websocket', 'ws-fallback'):
        logger.info('transport=%s 是客户端降级方式，服务端输出按 webrtc 处理', opt.transport)
        opt.transport = 'webrtc'

    # ─── 一机多卡负载均衡（GPU 网关模式入口）────────────────────────────
    # --gpu-ids 非空 → app.py 自身作为网关：为每张参与卡拉起一个 worker 子进程并转发信令。
    # --gpu-ids 留空   → 走下方原有单实例逻辑，行为与现状 100% 一致。
    # worker 子进程的命令行由 gpu_lb 剔除全部多卡参数后回调本文件，不会递归进入本分支。
    if getattr(opt, 'gpu_id_list', None):
        from gpu_lb import main as lb_main
        lb_main(opt)
        return

    logger.info(opt)

    # ─── 启动授权校验（license_check，默认关闭）────────────────────
    # 开启后向 zl-ai 主包 /aihuman/license/status 校验授权，失败则拒绝启动
    from license_check import run_license_check
    run_license_check(opt)

    # ─── 加载主模型运行时（触发 @register 注册 + 权重加载）──────────
    # 进程级切换时设置 ZL_DEFER_MODEL_LOAD=1：先启动 HTTP 服务，
    # 由首个 /offer 或 /warmup 再加载目标模型，端口能尽快恢复。
    if _DEFER_MODEL_LOAD:
        model = None
        load_avatar = None
        logger.info("延迟模型加载（进程级切换模式），等待首个请求")
    else:
        rt = ensure_model_runtime(opt.model)
        model = rt["model"]                 # 兼容旧全局引用
        load_avatar = rt["load_avatar"]     # 兼容旧全局引用

        # 预加载默认形象（缓存优先，内部按形象解析模型）
        get_or_load_avatar(opt.avatar_id)

        # ultralight 的 warm_up 依赖形象数据，载入默认形象后预热（保持原行为）
        if opt.model == 'ultralight':
            _ul = importlib.import_module(_AVATAR_MODULES['ultralight'])
            _ul.warm_up(opt.batch_size, global_avatars[opt.avatar_id], _warmup_resolution('ultralight'))

    # init rtc manager
    session_manager.init_builder(build_avatar_session)
    rtc_manager = RTCManager(opt)

    def _model_stats():
        """当前常驻模型：让 /capacity 一眼看出显存被哪套权重占着。"""
        counts: Dict[str, int] = {}
        try:
            for sess in list(session_manager.sessions.values()):
                if sess is None:
                    continue
                sopt = getattr(sess, "opt", None)
                model = getattr(sopt, "model", None)
                if not model:
                    aid = getattr(sopt, "avatar_id", None)
                    model = resolve_avatar_model(aid) if aid else None
                if model:
                    counts[model] = counts.get(model, 0) + 1
        except Exception:
            pass
        now = time.time()
        out = []
        for model in list(_model_runtimes):
            last = _model_last_used.get(model)
            out.append({
                "name": model,
                "resident": True,
                "sessions": counts.get(model, 0),
                "idle": round(now - last, 1) if last else None,
            })
        return out

    rtc_manager.model_stats_provider = _model_stats
    # share avatar_sessions (RTCManager handles it but routes.py expects it)
    
    if opt.transport=='virtualcam' or opt.transport=='rtmp':
        thread_quit = Event()
        params = {}
        # session 0 for virtualcam
        session_manager.add_session('0', build_avatar_session('0', params))
        rendthrd = Thread(target=session_manager.get_session('0').render,args=(thread_quit,))
        rendthrd.start()

    #############################################################################
    appasync = web.Application(client_max_size=1024**2*100)
    appasync["llm_response"] = llm_response
    appasync["opt"] = opt
    appasync["rtc_manager"] = rtc_manager

    appasync.on_shutdown.append(on_shutdown)

    # ─── 空闲会话回收（兜底）────────────────────────────────────────
    # ICE 卡在 checking 时 pc 永不 failed → 「pc 失败即清理」不触发，浏览器异常退出
    # （标签页崩溃/断网/直接关掉）时 WS 断开路径也可能走不到，会话就会一直挂着占内存。
    # 这里每 30s 扫一遍：既无 WS 租约、又无活着 pc、且超时无活跃 → 销毁。
    _REAP_INTERVAL = int(os.environ.get("TALK_REAP_INTERVAL", "10"))
    # 默认 10 秒回收空闲会话（TALK_REAP_IDLE_SECONDS 可调），避免掉线客户端长期占坑
    _REAP_IDLE_SECONDS = int(os.environ.get("TALK_REAP_IDLE_SECONDS", "10"))
    # 租约兜底：WS 客户端异常断开、finally 没跑到时，租约超过该时长且没有活跃 WS 消费 → 强制清理
    _LEASE_TTL_SECONDS = int(os.environ.get("TALK_LEASE_TTL_SECONDS", "120"))

    async def _session_reaper():
        while True:
            await asyncio.sleep(_REAP_INTERVAL)
            try:
                for sid in list(session_manager.sessions.keys()):
                    if session_manager.is_leased(sid):
                        sess = session_manager.get_session(sid)
                        player = getattr(getattr(sess, "output", None), "_player", None) if sess else None
                        ws_active = bool(getattr(player, "ws_active", False)) if player else False
                        if ws_active or session_manager.lease_age(sid) < _LEASE_TTL_SECONDS:
                            continue          # WS 降级通道还连着
                        logger.warning(
                            "会话回收：WS 租约超时未释放，强制清理 sessionid=%s（租约 %.0fs）",
                            sid, session_manager.lease_age(sid),
                        )
                        session_manager.force_release(sid)
                    try:
                        if rtc_manager.session_has_live_pc(sid):
                            continue      # WebRTC 还在用
                    except Exception:
                        continue
                    if session_manager.idle_seconds(sid) > _REAP_IDLE_SECONDS:
                        logger.info("会话回收：销毁空闲会话 sessionid=%s（无租约、无活 pc、空闲 %.0fs）",
                                    sid, session_manager.idle_seconds(sid))
                        session_manager.stop_session(sid)
                        session_manager.remove_session(sid)
                        try:
                            await rtc_manager.drop_session_pcs(sid)
                        except Exception:
                            logger.exception("关闭会话 pc 失败 sessionid=%s", sid)
                # FLV 降级页每 5s 心跳一次；后台标签页定时器会被浏览器节流，
                # 阈值给宽松些，避免切走一会儿就被回收（表现为“老是掉线”）。
                await rtc_manager.reap_flv_publishers(180.0)
                _evict_avatars()
                # 空闲模型释放会 empty_cache（可能上百 ms），放线程池避免卡事件循环
                await asyncio.get_event_loop().run_in_executor(
                    None, _release_idle_model_runtimes
                )
            except Exception:
                logger.exception("session reaper error")

    async def _start_reaper(app):
        app["_reaper_task"] = asyncio.ensure_future(_session_reaper())

    async def _stop_reaper(app):
        task = app.get("_reaper_task")
        if task is not None:
            task.cancel()

    appasync.on_startup.append(_start_reaper)
    appasync.on_shutdown.append(_stop_reaper)

    async def evict_idle_session(request):
        """LB 满员时调用：回收一个会话。

        默认只回收「无租约、无活 PC、空闲≥阈值」的会话；
        `oldest=1` 时放宽为按 LRU 释放最旧会话（有用户排队时用，保证新用户能连入）。
        """
        aggressive = (request.query.get("oldest", "0").strip().lower()
                      in ("1", "true", "yes", "on"))
        try:
            threshold = float(request.query.get("idle", os.environ.get("TALK_EVICT_IDLE_SECONDS", "10")))
        except (TypeError, ValueError):
            threshold = 10.0

        def _free(sid):
            try:
                return (not session_manager.is_leased(sid)
                        and not rtc_manager.session_has_live_pc(sid)
                        and session_manager.idle_seconds(sid) >= threshold)
            except Exception:
                return False

        mode = "oldest" if aggressive else "idle"
        if aggressive:
            # 先找真正空闲的；没有就按 LRU 释放最旧会话（新用户已在排队）
            sid = (session_manager.evict_oldest(predicate=_free)
                   or session_manager.evict_oldest())
        else:
            sid = session_manager.evict_oldest(predicate=_free)

        if not sid:
            return web.json_response({"code": 1, "msg": "no idle session"})
        try:
            await rtc_manager.drop_session_pcs(sid)
        except Exception:
            logger.exception("evict_idle: 关闭 pc 失败 sessionid=%s", sid)
        logger.info("evict_idle(%s): 回收会话 %s", mode, sid)
        return web.json_response({"code": 0, "evicted": sid})

    async def capacity_status(request):
        """运维/安装脚本使用的容量快照；页面不调用、不展示数字。"""
        data = rtc_manager.metrics()
        return web.json_response({
            "code": 0,
            "data": {
                "used": data.get("used", 0),
                "max": data.get("max", 0),
                "queue_evict": data.get("queue_evict"),
                "reuse_on_offer": data.get("reuse_on_offer"),
                "sessions": data.get("sessions", []),
                "models": data.get("models", []),
            },
        })

    async def metrics_status(request):
        """性能与资源指标：会话、fps/队列/丢帧、RSS、线程数。"""
        import threading
        data = rtc_manager.metrics()
        try:
            with open("/proc/self/status", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        data["rss_kb"] = int(line.split()[1])
                        break
        except Exception:
            pass
        data["threads"] = threading.active_count()
        return web.json_response(data)

    async def ws_ping(request):
        """WS 连通性测试端点：握手成功立即回 pong 并关闭，不建会话、不加载模型。"""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_str("pong")
        await ws.close()
        return ws
    appasync.router.add_get("/ws/ping", ws_ping)
    appasync.router.add_post("/evict_idle", evict_idle_session)
    appasync.router.add_get("/capacity", capacity_status)
    async def avatar_size(request):
        """素材原生分辨率：发布页据此禁用"素材达不到"的画质档位。

        只从已加载缓存里读（不主动触发冷加载，避免页面被拖住）。
        """
        avatar_id = (request.query.get("avatar") or "").strip()
        payload = global_avatars.get(avatar_id) if avatar_id else None
        if payload is None:
            return web.json_response({"code": 0, "data": {"cached": False}})
        try:
            frames = None
            if isinstance(payload, (tuple, list)) and payload:
                frames = payload[0]
            elif isinstance(payload, dict):
                frames = payload.get("frame_list") or payload.get("frame_list_cycle")
            else:
                frames = getattr(payload, "frame_list_cycle", None)
            if not frames:
                return web.json_response({"code": 0, "data": {"cached": False}})
            h, w = frames[0].shape[:2]
            w, h = int(w), int(h)
            short = min(w, h)
            max_tier = 540 if short < 720 else (720 if short < 1080 else 1080)
            return web.json_response({"code": 0, "data": {
                "cached": True, "width": w, "height": h, "short": short,
                "max_short": max_tier,
            }})
        except Exception:
            logger.exception("avatar_size failed")
            return web.json_response({"code": 0, "data": {"cached": False}})

    appasync.router.add_get("/metrics", metrics_status)
    appasync.router.add_get("/avatar-size", avatar_size)
    appasync.router.add_post("/offer", offer)
    appasync.router.add_post("/flv/publish", flv_publish)
    appasync.router.add_post("/flv/ping", flv_ping)
    appasync.router.add_post("/flv/unpublish", flv_unpublish)
    appasync.router.add_get("/speedtest/ping", speedtest_ping)
    appasync.router.add_get("/speedtest/download", speedtest_download)
    appasync.router.add_post("/speedtest/upload", speedtest_upload)
    appasync.router.add_get("/record/{sessionid}", download_record)
    appasync.router.add_post("/warmup", warmup_avatar)
    appasync.router.add_get("/warmup/status", warmup_status)

    # ─── WS 音视频降级通道（特殊环境，默认不启用，见 server/ws_video.py）──
    # 必须在 setup_routes(内含 add_static '/') 之前注册，否则会被静态路由吞掉
    from server.ws_video import ws_video, ws_stats
    appasync.router.add_get("/ws/video", ws_video)
    appasync.router.add_get("/ws/stats", ws_stats)
    
    # 注册 server/routes.py 中的通用 API 路由
    setup_routes(appasync)

    # ─── 入站鉴权（X-Api-Key，ZL_API_KEY 未配置时放行）────────────
    # 需注册在 aiohttp_cors.setup 之前，保证预检 OPTIONS 先放行
    from server.auth import make_auth_middleware
    appasync.middlewares.append(make_auth_middleware())

    # Configure default CORS settings.
    cors = aiohttp_cors.setup(appasync, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
            )
        })
    # Configure CORS on all routes.
    for route in list(appasync.router.routes()):
        cors.add(route)

    pagename='index.html'
    if opt.transport=='rtmp':
        pagename='rtmpapi.html'
    elif opt.transport=='rtcpush':
        pagename='rtcpushapi.html'
    logger.info('start http server; http://<serverip>:'+str(opt.listenport)+'/'+pagename)
    # logger.info('如果使用webrtc，推荐访问webrtc集成前端: http://<serverip>:'+str(opt.listenport)+'/dashboard.html')
    def run_server(runner):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(runner.setup())
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        loop.run_until_complete(site.start())
        if opt.transport=='rtcpush':
            for k in range(opt.max_session):
                push_url = opt.push_url
                if k!=0:
                    push_url = opt.push_url+str(k)
                loop.run_until_complete(rtc_manager.handle_rtcpush(push_url, str(k)))
        loop.run_forever()    
    #Thread(target=run_server, args=(web.AppRunner(appasync),)).start()
    run_server(web.AppRunner(appasync))

    #app.on_shutdown.append(on_shutdown)
    #app.router.add_post("/offer", offer)

    # print('start websocket server')
    # server = pywsgi.WSGIServer(('0.0.0.0', 8000), app, handler_class=WebSocketHandler)
    # server.serve_forever()


# os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
# os.environ['MULTIPROCESSING_METHOD'] = 'forkserver'                                                    
if __name__ == '__main__':
    mp.set_start_method('spawn')
    load_dotenv()  # Load environment variables from .env file, if it exists
    main()
    
    
    
