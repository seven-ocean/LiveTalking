# -*- coding: utf-8 -*-
###############################################################################
#  Copyright (C) 2025 四川昱扬科技有限公司
#
#  gpu_lb.py — 一机多卡（GPU 负载均衡）网关
#
#  设计文档：.trae/documents/zl-talking-一机多卡负载均衡方案.md
#
#  职责：
#    · 由 app.py main() 在 --gpu-ids 非空时导入调用；单卡模式下不会被加载；
#    · 不提供独立启动入口（无 __main__ 分支）：
#        主进程  = 网关(master)：监听 --listenport，按会话做 HTTP 7 层转发；
#        worker  = 每张参与卡一个子进程：python app.py <原参数(剔除全部多卡参数)> --listenport <port>
#                  并注入环境 CUDA_VISIBLE_DEVICES=<物理卡>（进程内仍只见 cuda:0，推理代码零改动）
#    · 会话级调度：POST /offer 按策略选卡并记录 sessionid→worker，后续带 sessionid 的请求按表
#      转发到同一 worker（会话亲和，与“会话粘死创建它的进程”现状一致）；
#    · worker 守护：周期 GET /health 更新负载；进程异常/退出 → 自动重启并作废其会话映射；
#    · 选卡策略：least_session（空闲优先，默认）/ round_robin（轮询兜底）。
#
#  说明：
#    · 本模块内的大多数选择/解析函数均为纯函数（无 IO），可直接被单卡机上的单测覆盖；
#    · 集成测试可通过环境变量 ZL_LB_WORKER_APP 把 worker 程序替换为 Stub（见 tests/），
#      从而在不依赖任何 GPU/模型的环境下验证网关逻辑（对应方案 §7.1 方法 1）。
###############################################################################

import os
import re
import sys
import json
import asyncio
import signal
import time
from urllib.parse import unquote

import aiohttp
from aiohttp import web

from utils.logger import logger


# ─── 常量 ──────────────────────────────────────────────────────────────────

# 测试/调试钩子：worker 启动程序路径（默认取当前 app.py；集成测试指向 Stub Worker）
_WORKER_APP = os.environ.get('ZL_LB_WORKER_APP', '').strip()

# 与 app.py 入站鉴权一致（server/auth.py 读取同一环境变量 ZL_API_KEY）
_API_KEY = os.environ.get('ZL_API_KEY', '').strip()

# 需从 worker 子进程 argv 中剥离的网关参数（防 worker 再次进入多卡分支 → 防递归）
_LB_ARG_NAMES = (
    '--gpu-ids', '--gpu-ids-strategy', '--lb-worker-base-port',
    '--lb-max-session-per-gpu', '--lb-health-interval',
    '--lb-worker-wait-timeout', '--listenport',
)

# 转发时剔除的逐跳头（长度/连接由 aiohttp 客户端自行处理）
_HOP_BY_HOP = {
    'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization',
    'proxy-connection', 'te', 'trailer', 'transfer-encoding', 'upgrade',
    'host', 'content-length', 'content-encoding',
}

# 单次业务转发总超时（worker 侧 /offer 自带 offer_timeout 兜底，此处只需给足余量）
_CLIENT_TIMEOUT = aiohttp.ClientTimeout(total=180, connect=5)
_HEALTH_TIMEOUT = aiohttp.ClientTimeout(total=4)

# multipart/form-data 中提取 name="sessionid" 文本字段（仅 /humanaudio 路由用）
_FORM_SID_RE = re.compile(rb'name="sessionid"\s*\r?\n\s*\r?\n\s*([^\r\n]+)')


# ─── 纯函数：命令行 / 选卡 / 会话提取（供单测直接覆盖）───────────────────

def strip_lb_args(argv):
    """剔除 argv 中的全部多卡网关参数与 --listenport，返回清理后的参数列表。

    子进程据此 + --listenport <worker端口> 回调 app.py，由于不再含 --gpu-ids，
    会走单卡分支，不会递归进入多卡网关。支持 '--opt value' 与 '--opt=value'。
    """
    out = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        name = arg.split('=', 1)[0]
        if name in _LB_ARG_NAMES:
            i += 1 if '=' in arg else 2
            continue
        out.append(arg)
        i += 1
    return out


def build_worker_cmd(worker_app, stripped_argv, port):
    """构造 worker 子进程命令行（纯函数，便于单测）。"""
    return [sys.executable, worker_app] + list(stripped_argv) + ['--listenport', str(port)]


def pick_worker(workers, strategy, rr_start, max_per_gpu):
    """按策略选一个未满员的 worker。

    参数:
        workers:     list[Worker]（或任何带 .ready/.load 的对象）
        strategy:    'least_session' | 'round_robin'
        rr_start:    本轮轮询起点（workers 列表下标，兼顾公平）
        max_per_gpu: 每卡新会话上限（达到则跳过该卡）
    返回: (选中下标, 下一轮轮询起点)；无可用 worker 返回 (None, rr_start)。纯函数。
    """
    n = len(workers)
    if n == 0:
        return None, rr_start
    avail = []
    for k in range(n):
        idx = (rr_start + k) % n
        w = workers[idx]
        if w.ready and w.load < max_per_gpu:
            avail.append((k, idx))
    if not avail:
        return None, rr_start
    if strategy == 'least_session':
        # 负载最小者优先；负载相同时按轮询扫描序（k 小者先）兼顾公平
        _, idx = min(avail, key=lambda ki: (workers[ki[1]].load, ki[0]))
    else:  # round_robin
        _, idx = avail[0]
    return idx, (idx + 1) % n


def _extract_sessionid(raw: bytes) -> str:
    """从请求体提取 sessionid：JSON 优先，其次 multipart 文本字段（纯逻辑，供单测）。"""
    if not raw:
        return ''
    head = raw[:64].lstrip()
    if head[:1] in (b'{', b'['):  # JSON 请求（体积小）才尝试 json 解析，避免解码大文件体
        try:
            obj = json.loads(raw.decode('utf-8', 'replace'))
            if isinstance(obj, dict):
                sid = obj.get('sessionid')
                if sid:
                    return str(sid)
        except Exception:
            pass
    m = _FORM_SID_RE.search(raw)
    if m:
        try:
            return m.group(1).decode('utf-8')
        except UnicodeDecodeError:
            return ''
    return ''


# ─── 运行时句柄与状态 ──────────────────────────────────────────────────────

class Worker:
    """单个物理卡 worker 的运行时句柄。"""

    def __init__(self, gpu_idx, gpu_id, port):
        self.gpu_idx = gpu_idx          # 参与均衡的卡序号（0 起，仅逻辑顺序）
        self.gpu_id = gpu_id            # 物理卡号（--gpu-ids 中的元素）
        self.port = port                # worker HTTP 端口
        self.proc = None                # asyncio.subprocess.Process
        self.ready = False              # /health 通过 = 模型就绪
        self.sessions = 0               # 活跃会话数（权威来源：worker /health data.sessions）
        self.reserved = 0               # 在途 /offer 预占名额（防止并发 offer 超卖）

    @property
    def load(self):
        """当前负载 = 已确认会话 + 在途预占。"""
        return self.sessions + self.reserved

    @property
    def alive(self):
        return self.proc is not None and self.proc.returncode is None


class LBState:
    def __init__(self):
        self.opt = None
        self.workers = []
        self.route = {}                 # sessionid -> Worker
        self.client = None              # aiohttp.ClientSession
        self.lock = asyncio.Lock()
        self.rr = 0                     # 下一轮询起点（workers 下标）
        self.health_task = None
        self.stop_ev = None             # 退出信号事件


state = LBState()


# ─── 小工具：响应构造 / 转发 ───────────────────────────────────────────────

def _json_err(msg, code=-1, status=200):
    """错误响应（与 server/routes.py json_error 相同的 HTTP 语义：默认 200 + code）。"""
    return web.Response(status=status, content_type='application/json',
                        text=json.dumps({'code': code, 'msg': str(msg)}, ensure_ascii=False))


def _base_url(w):
    return 'http://127.0.0.1:%d' % w.port


def _req_headers(request):
    headers = {k: v for k, v in request.headers.items() if k.lower() not in _HOP_BY_HOP}
    if _API_KEY:  # 转发到 worker 时补上鉴权头（master 与 worker 同源同 .env）
        headers['X-Api-Key'] = _API_KEY
    return headers


async def _exchange(w, method, url, headers, raw):
    """把请求体 raw 转发给 worker，透传其状态码/关键响应头/响应体。"""
    try:
        async with state.client.request(method, url, headers=headers,
                                        data=raw, timeout=_CLIENT_TIMEOUT) as resp:
            body = await resp.read()
            resp_headers = {}
            for h in ('content-type', 'content-disposition'):
                v = resp.headers.get(h)
                if v:
                    resp_headers[h] = v
            return web.Response(status=resp.status, body=body, headers=resp_headers)
    except Exception as e:
        logger.error('LB: 转发 %s %s → worker gpu %s (port %d) 失败: %s',
                     method, url, w.gpu_id, w.port, e)
        return _json_err('backend worker (gpu %s) unavailable: %s' % (w.gpu_id, e), status=502)


async def _forward(w, request):
    """整包转发（JSON/表单等中小体量请求）。"""
    raw = await request.read()
    url = _base_url(w) + request.rel_url.path
    if request.rel_url.query_string:
        url += '?' + request.rel_url.query_string
    return await _exchange(w, request.method, url, _req_headers(request), raw)


async def _forward_stream(w, request):
    """流式转发（录制文件下载等大响应，避免整包缓存内存）。"""
    url = _base_url(w) + request.rel_url.path
    if request.rel_url.query_string:
        url += '?' + request.rel_url.query_string
    try:
        async with state.client.request(request.method, url,
                                        headers=_req_headers(request),
                                        timeout=_CLIENT_TIMEOUT) as resp:
            out = web.StreamResponse(status=resp.status)
            ct = resp.headers.get('content-type')
            if ct:
                out.content_type = ct
            cd = resp.headers.get('content-disposition')
            if cd:
                out.headers['Content-Disposition'] = cd
            await out.prepare(request)
            async for chunk in resp.content.iter_any():
                await out.write(chunk)
            await out.write_eof()
            return out
    except Exception as e:
        logger.error('LB: 流式转发 %s 失败: %s', url, e)
        return _json_err('backend worker (gpu %s) unavailable: %s' % (w.gpu_id, e), status=502)


# ─── worker 托管：拉起 / 就绪等待 / 健康自愈 ───────────────────────────────

async def _kill_proc(w):
    """终止 worker 进程并等待退出（尽力而为，不抛异常）。"""
    proc = w.proc
    if proc is not None and proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(proc.wait(), 5)
        except asyncio.TimeoutError:
            logger.error('LB: worker gpu %s (port %d) 未能及时退出', w.gpu_id, w.port)


async def _wait_worker_ready(w, timeout):
    """轮询 worker /health 直到可服务（= 模型加载并启动完成）。"""
    deadline = time.monotonic() + timeout
    while True:
        if not w.alive:
            raise RuntimeError('worker (gpu %s, port %d) 进程已退出 code=%s'
                               % (w.gpu_id, w.port,
                                  w.proc.returncode if w.proc else 'n/a'))
        try:
            async with state.client.get(_base_url(w) + '/health', timeout=_HEALTH_TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    w.sessions = int(data.get('data', {}).get('sessions', 0))
                    w.ready = True
                    logger.info('LB: worker gpu %s (port %d) 就绪, sessions=%d',
                                w.gpu_id, w.port, w.sessions)
                    return
        except Exception:
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError('worker (gpu %s, port %d) 在 %.0fs 内未就绪，'
                               '请检查该卡是否可用（nvidia-smi）或调大 --lb-worker-wait-timeout'
                               % (w.gpu_id, w.port, timeout))
        await asyncio.sleep(1)


async def _start_worker(w):
    """（重新）拉起 worker 子进程并等待 /health 就绪。启动前先终止旧进程。"""
    await _kill_proc(w)
    stripped = strip_lb_args(sys.argv[1:])
    worker_app = _WORKER_APP or os.path.abspath(sys.argv[0])
    cmd = build_worker_cmd(worker_app, stripped, w.port)
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(w.gpu_id)  # 进程内设备枚举恒为 cuda:0，推理零改动
    w.ready = False
    w.sessions = 0
    logger.info('LB: 拉起 worker gpu=%s port=%d cmd=%s', w.gpu_id, w.port, cmd)
    w.proc = await asyncio.create_subprocess_exec(*cmd, env=env)
    await _wait_worker_ready(w, state.opt.lb_worker_wait_timeout)


def _drop_routes_of(w):
    """作废该 worker 的全部会话映射（worker 重启后其会话必然丢失）。"""
    stale = [sid for sid, tw in state.route.items() if tw is w]
    for sid in stale:
        state.route.pop(sid, None)
    if stale:
        logger.warning('LB: 已作废 worker gpu %s (port %d) 的 %d 个会话映射',
                       w.gpu_id, w.port, len(stale))


async def _health_loop():
    """周期健康/负载轮询：更新每卡会话数；异常 worker 自动重启。"""
    interval = max(1, int(state.opt.lb_health_interval))
    while True:
        await asyncio.sleep(interval)
        for w in list(state.workers):
            if w.proc is None:
                continue
            ok = False
            try:
                async with state.client.get(_base_url(w) + '/health',
                                            timeout=_HEALTH_TIMEOUT) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        w.sessions = int(data.get('data', {}).get('sessions', 0))
                        w.ready = True
                        ok = True
            except Exception:
                pass
            if ok:
                continue
            # 不健康/进程退出 → 作废映射并重启
            w.ready = False
            w.sessions = 0
            _drop_routes_of(w)
            logger.warning('LB: worker gpu %s (port %d) 健康检查失败，自动重启',
                           w.gpu_id, w.port)
            try:
                await _start_worker(w)
            except Exception as e:
                logger.error('LB: worker gpu %s (port %d) 重启失败: %s', w.gpu_id, w.port, e)


# ─── 网关路由分派 ──────────────────────────────────────────────────────────

async def _handle_offer(request):
    """/offer：选卡 → 转发 → 解析 sessionid 建映射 → 原样返回响应。"""
    opt = state.opt
    async with state.lock:
        idx, state.rr = pick_worker(state.workers, opt.gpu_ids_strategy,
                                    state.rr, opt.lb_max_session_per_gpu)
        if idx is None:
            logger.warning('LB: 所有 worker 均已满员（每卡上限 %d），拒绝新会话',
                           opt.lb_max_session_per_gpu)
            return _json_err('all gpu workers reach max sessions (%d per gpu), try later'
                             % opt.lb_max_session_per_gpu)
        w = state.workers[idx]
        w.reserved += 1  # 预占名额，防止并发 /offer 超卖
    try:
        resp = await _forward(w, request)
        if resp.status == 200 and resp.body:
            sid = None
            try:
                sid = json.loads(resp.body.decode('utf-8', 'replace')).get('sessionid')
            except Exception:
                pass
            if sid:
                async with state.lock:
                    state.route[sid] = w
                w.sessions += 1
                logger.info('LB: 新会话 %s → gpu %s (port %d)，当前会话数=%d',
                            sid, w.gpu_id, w.port, w.sessions)
            else:
                # 失败/超时响应体无 sessionid：不建映射（重试/原样透传给客户端）
                logger.info('LB: /offer 转发完成但无 sessionid（status=%s）', resp.status)
        return resp
    finally:
        w.reserved = max(0, w.reserved - 1)


def _remove_route(sid, w):
    """stop_session 成功后删除映射并乐观递减负载（health 轮询会再次校正）。"""
    if state.route.get(sid) is w:
        state.route.pop(sid, None)
        w.sessions = max(0, w.sessions - 1)
        logger.info('LB: 会话 %s 结束(stop_session)，worker gpu %s (port %d) 会话数=%d',
                    sid, w.gpu_id, w.port, w.sessions)


async def _handle_warmup(request):
    """/warmup：广播所有 worker（保证每个 worker 都预热新 avatar，避免冷启动超时）。"""
    raw = await request.read()
    headers = _req_headers(request)
    if not state.workers:
        return _json_err('no worker available', status=503)

    async def _warm(w):
        resp = await _exchange(w, 'POST', _base_url(w) + '/warmup', headers, raw)
        return w.port, resp.status

    results = await asyncio.gather(*[_warm(w) for w in state.workers],
                                   return_exceptions=True)
    ok = sum(1 for r in results
             if not isinstance(r, Exception) and r[1] == 200)
    logger.info('LB: /warmup 广播完成 ok=%d/%d', ok, len(results))
    if ok == 0:
        return _json_err('warmup failed on all workers')
    return web.json_response({'code': 0, 'msg': 'ok',
                              'data': {'warmed': ok, 'total': len(results)}})


def _health_payload(request):
    """GET /health：聚合各 worker 存活/会话数/路由表。detail=1 时附路由明细。"""
    detail = request.rel_url.query.get('detail') in ('1', 'true', 'yes')
    total = sum(w.sessions for w in state.workers)
    data = {
        'mode': 'lb',
        'sessions': total,
        'workers': [
            {'gpu_id': w.gpu_id, 'port': w.port, 'ready': w.ready,
             'sessions': w.sessions, 'reserved': w.reserved}
            for w in state.workers
        ],
        'route_size': len(state.route),
    }
    if detail:
        data['route'] = {sid: w.port for sid, w in state.route.items()}
    return web.json_response({'code': 0, 'msg': 'ok', 'data': data})


async def dispatch(request):
    """网关统一入口：按路径/会话规则分派（路由规则见方案 §5.3）。"""
    path = request.rel_url.path
    method = request.method

    if path == '/health' and method == 'GET':
        return _health_payload(request)
    if path == '/offer':
        return await _handle_offer(request)
    if path == '/warmup':
        return await _handle_warmup(request)

    # ── 录制文件下载：GET /record/{sessionid}（文件在共享磁盘，映射未命中则回源 worker0）──
    if method == 'GET' and path.startswith('/record/'):
        sid = unquote(path[len('/record/'):])
        w = state.route.get(sid)
        if w is not None:
            return await _forward_stream(w, request)
        return await _forward(state.workers[0], request)

    # ── 会话级路由（带 sessionid 的请求按映射转发，会话亲和）──────────────
    sid = _extract_sessionid(await request.read()) if method == 'POST' else ''
    if sid:
        w = state.route.get(sid)
        if w is not None:
            resp = await _forward(w, request)
            if path == '/stop_session' and resp.status == 200:
                _remove_route(sid, w)
            return resp
        # 带 sessionid 但映射已作废（worker 重启等）→ 与单实例语义一致返回 not found
        return _json_err('session not found')

    # ── 与会话无关的请求（/config、静态 /web、/api/avatar/task*、/api/admin 等）→ worker0 ──
    if not state.workers:
        return _json_err('no worker available', status=503)
    return await _forward(state.workers[0], request)


# ─── CORS 中间件（保持与单实例 aiohttp_cors 等同的宽松策略）────────────────

def _cors_middleware():
    @web.middleware
    async def cors_mw(request, handler):
        origin = request.headers.get('Origin')
        if request.method == 'OPTIONS':
            resp = web.Response(status=204)
        else:
            resp = await handler(request)
        if origin:
            resp.headers['Access-Control-Allow-Origin'] = origin
            resp.headers['Vary'] = 'Origin'
            resp.headers['Access-Control-Allow-Credentials'] = 'true'
            resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
            resp.headers['Access-Control-Allow-Headers'] = '*'
            resp.headers['Access-Control-Expose-Headers'] = '*'
        return resp
    return cors_mw


# ─── 主入口（被 app.py 调用）───────────────────────────────────────────────

async def _amain(opt):
    """网关主协程：拉起全部 worker → 启动 HTTP 网关 → 等待退出并清理。"""
    state.opt = opt
    state.workers.clear()
    state.route.clear()
    state.rr = 0
    state.lock = asyncio.Lock()
    state.stop_ev = asyncio.Event()
    state.health_task = None
    state.client = aiohttp.ClientSession()

    try:
        # 1) 逐个拉起 worker（顺序启动避免显存尖峰；任一失败 → 整体失败退出，避免“半集群”上线）
        try:
            for i, gpu_id in enumerate(opt.gpu_id_list):
                w = Worker(gpu_idx=i, gpu_id=gpu_id, port=opt.lb_worker_base_port + i)
                state.workers.append(w)
                await _start_worker(w)
        except Exception as e:
            logger.error('LB: worker 启动失败，已退出网关。请检查参与卡是否可用/显存/'
                         'CUDA_VISIBLE_DEVICES 及 --lb-worker-wait-timeout。原因：%s', e)
            raise

        # 2) 构建网关 aiohttp 应用（与单实例保持一致的 body 上限与 CORS/鉴权）
        from server.auth import make_auth_middleware
        app = web.Application(client_max_size=1024 ** 2 * 100,
                              middlewares=[_cors_middleware(), make_auth_middleware()])
        app.router.add_route('*', '/{tail:.*}', dispatch)

        state.health_task = asyncio.create_task(_health_loop())

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', opt.listenport)
        await site.start()
        logger.info('LB: 网关已监听 0.0.0.0:%d，workers=%s',
                    opt.listenport, [(w.gpu_id, w.port) for w in state.workers])

        # 3) 等待退出信号（Windows 不支持 add_signal_handler，Ctrl+C 走 KeyboardInterrupt 兜底）
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, state.stop_ev.set)
            except (NotImplementedError, RuntimeError):
                break
        try:
            await state.stop_ev.wait()
        finally:
            try:
                await asyncio.shield(runner.cleanup())
            except asyncio.CancelledError:
                pass  # 取消路径下 cleanup 在后台收尾，进程退出兜底
    finally:
        if state.health_task is not None:
            state.health_task.cancel()
            try:
                await state.health_task
            except (asyncio.CancelledError, Exception):
                pass
        if state.client is not None:
            await state.client.close()
        for w in state.workers:
            await _kill_proc(w)
        logger.info('LB: 网关已停止')


def main(opt):
    """多卡网关模式入口（app.py 检测到 --gpu-ids 非空时调用）。"""
    logger.info('========== GPU 负载均衡网关模式（master） ==========')
    logger.info('gpu-ids=%s strategy=%s max-session-per-gpu=%d worker-base-port=%d listenport=%d',
                opt.gpu_id_list, opt.gpu_ids_strategy, opt.lb_max_session_per_gpu,
                opt.lb_worker_base_port, opt.listenport)
    asyncio.run(_amain(opt))
