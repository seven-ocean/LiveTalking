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
import numpy as np
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

from utils.hardware_monitor import log_device_info, log_resource_usage, start_monitoring


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

# ─── avatar 加载状态注册表（线程安全）────────────────────────────────
# avatar_id -> {"status": "loading"|"ready"|"error", "startedAt": float,
#               "elapsedMs": int|None, "error": str|None}
_avatar_state: Dict[str, dict] = {}
_avatar_state_lock = Lock()
_avatar_load_lock = Lock()  # 保证同一时刻只有一个 avatar 冷加载（并发去重）
        

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
        avatar_mod = importlib.import_module(_AVATAR_MODULES[model_name])
        logger.info("懒加载模型运行时: %s", model_name)
        weights = _load_model_weights(model_name, avatar_mod, opt)
        # warm_up 签名各模型不同；ultralight 的 warm_up 依赖形象数据，延后到 main() 处理（保持原行为）
        if model_name == 'wav2lip':
            avatar_mod.warm_up(opt.batch_size, weights, 256)
        elif model_name == 'musetalk':
            avatar_mod.warm_up(opt.batch_size, weights)
        rt = {"model": weights, "load_avatar": avatar_mod.load_avatar}
        _model_runtimes[model_name] = rt
        logger.info("模型运行时就绪: %s", model_name)
        return rt

def get_or_load_avatar(avatar_id: str):
    """获取 avatar（缓存优先）；未缓存则加载。

    并发调用同一 avatar 时由 _avatar_load_lock 串行去重，只有一个真正执行
    load_avatar，其余在拿到锁后发现缓存已就绪直接复用。
    """
    cached = global_avatars.get(avatar_id)
    if cached is not None:
        return cached
    with _avatar_load_lock:
        cached = global_avatars.get(avatar_id)
        if cached is not None:
            return cached
        started = time.time()
        model_name = resolve_avatar_model(avatar_id)
        _update_avatar_state(avatar_id, status="loading", startedAt=started,
                             elapsedMs=None, error=None, model=model_name)
        try:
            rt = ensure_model_runtime(model_name)
            avatar = rt["load_avatar"](avatar_id)
            global_avatars[avatar_id] = avatar
            _update_avatar_state(avatar_id, status="ready",
                                 elapsedMs=int((time.time() - started) * 1000))
            return avatar
        except BaseException as e:
            _update_avatar_state(avatar_id, status="error",
                                 elapsedMs=int((time.time() - started) * 1000),
                                 error=str(e))
            raise

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

def build_avatar_session(sessionid:str, params:dict)->BaseAvatar:
    opt_this = copy.deepcopy(opt)
    opt_this.sessionid = sessionid

    avatar_id = params.get('avatar',opt.avatar_id)
    opt_this.avatar_id = avatar_id
    ref_audio = params.get('refaudio','') #音色
    ref_text = params.get('reftext','')

    # 按形象解析其真实模型，并用该模型的权重/loader 构建会话（支持单进程跨模型）
    model_name = resolve_avatar_model(avatar_id)
    rt = ensure_model_runtime(model_name)
    # 缓存优先，未缓存则冷加载（与 /warmup 共用同一去重逻辑）
    avatar_this = get_or_load_avatar(avatar_id)

    if ref_audio: #请求参数配置了参考音频
        opt_this.REF_FILE = ref_audio
        opt_this.REF_TEXT = ref_text
    custom_config=params.get('custom_config','') #动作编排配置
    if custom_config:
        opt_this.customopt = json.loads(custom_config)

    opt_this.model = model_name  # 会话记录真实模型（供 /admin 状态等使用）
    avatar_session = registry.create("avatar", model_name, opt=opt_this, model=rt["model"], avatar=avatar_this)
    return avatar_session

async def offer(request):
    return await rtc_manager.handle_offer(request)

async def warmup_avatar(request):
    """预热 avatar：前端切换形象前调用，后台异步加载，立即返回，避免阻塞事件循环。

    幂等：重复/并发调用不会重复加载。契约：{code:0, status:"ready"|"loading"}。
    """
    params = await request.json()
    avatar_id = params.get('avatar') or opt.avatar_id
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
    # 启动只加载 opt.model 一套权重（与现状启动开销一致）；其它模型在
    # /offer 或 /warmup 首次用到时按形象懒加载并常驻（见 ensure_model_runtime）。
    rt = ensure_model_runtime(opt.model)
    model = rt["model"]                 # 兼容旧全局引用
    load_avatar = rt["load_avatar"]     # 兼容旧全局引用

    # 预加载默认形象（缓存优先，内部按形象解析模型）
    get_or_load_avatar(opt.avatar_id)

    # ultralight 的 warm_up 依赖形象数据，载入默认形象后预热（保持原行为）
    if opt.model == 'ultralight':
        _ul = importlib.import_module(_AVATAR_MODULES['ultralight'])
        _ul.warm_up(opt.batch_size, global_avatars[opt.avatar_id], 160)

    # init rtc manager
    session_manager.init_builder(build_avatar_session)
    rtc_manager = RTCManager(opt)
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
    appasync.router.add_post("/offer", offer)
    appasync.router.add_get("/record/{sessionid}", download_record)
    appasync.router.add_post("/warmup", warmup_avatar)
    appasync.router.add_get("/warmup/status", warmup_status)

    # ─── WS 音视频降级通道（特殊环境，默认不启用，见 server/ws_video.py）──
    # 必须在 setup_routes(内含 add_static '/') 之前注册，否则会被静态路由吞掉
    from server.ws_video import ws_video
    appasync.router.add_get("/ws/video", ws_video)
    
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
    
    
    
