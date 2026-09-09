# -*- coding: utf-8 -*-
###############################################################################
#  run_gpu_lb_single_gpu_verify.py — 单卡机多卡逻辑验证（方案 §7.1 方法 1 + 3）
#
#  运行（在 zl-talking 仓库根目录下）：
#      python tests/run_gpu_lb_single_gpu_verify.py
#
#  覆盖内容：
#    A. 纯函数单测（方法 3）：
#       strip_lb_args / build_worker_cmd / pick_worker(least/round_robin/满员跳过/
#       全满拒绝) / _extract_sessionid(JSON + multipart)
#    B. Stub Worker 集成测试（方法 1）：master 网关 + 假 worker（真实子进程），
#       端到端断言：
#       1) least_session 下 4 个新会话在两张“卡”上平分（2/2）
#       2) 第 5 个 offer 因每卡容量上限被拒
#       3) 会话亲和：同一 sessionid 的后续 /human 全部命中同一 worker
#       4) /stop_session 释放容量后，新会话被调度到已空出的卡
#       5) kill 一个 worker → 其会话映射作废、自动重启、回归服务
#       6) round_robin 策略下新会话交替分配
#
#  说明：
#    · 本脚本只依赖 Python 标准库 + aiohttp（仓库既有依赖），不加载模型/不绑 GPU；
#    · master 的 worker 启动程序通过环境变量 ZL_LB_WORKER_APP 指向 gpu_lb_stub_worker.py，
#      生产环境未设置该变量时仍以 app.py 为 worker 程序（代码路径完全一致）。
#
#  真机（多卡）部署验收见方案 §7 步骤 2/4/5；真 worker CPU 模拟“第二张卡”的冒烟
#  （方法 2）需真实 app.py + 模型，单独手工执行：
#      python app.py <原有参数> --gpu-ids 0,1
###############################################################################

import os
import sys
import json
import time
import asyncio
import socket
import aiohttp
from types import SimpleNamespace

# ─── 环境准备（必须在 import gpu_lb 之前）──────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO_ROOT)
sys.path.insert(0, REPO_ROOT)
STUB_WORKER = os.path.join(REPO_ROOT, 'tests', 'gpu_lb_stub_worker.py')
os.environ['ZL_LB_WORKER_APP'] = STUB_WORKER   # master 以 Stub 作为 worker 程序
os.environ.pop('ZL_API_KEY', None)             # 测试环境关闭鉴权

import gpu_lb  # noqa: E402


# ─── 工具 ──────────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0


def ok(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print('  [PASS] ' + msg)
    else:
        FAIL += 1
        print('  [FAIL] ' + msg)


def _free_ports(n):
    socks, ports = [], []
    for _ in range(n):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('127.0.0.1', 0))
        socks.append(s)
        ports.append(s.getsockname()[1])
    for s in socks:
        s.close()
    return ports


def _opt(strategy, ports, max_per_gpu=2, health_interval=1):
    master, w0, w1 = ports
    return SimpleNamespace(
        gpu_id_list=[0, 1],
        gpu_ids_strategy=strategy,
        lb_worker_base_port=w0,
        lb_max_session_per_gpu=max_per_gpu,
        lb_health_interval=health_interval,
        lb_worker_wait_timeout=30,
        listenport=master,
    )


# ─── A. 纯函数单测 ───────────────────────────────────────────────────────

def unit_tests():
    print('== A. 纯函数单测（选卡 / argv 剥离 / sessionid 提取） ==')

    # strip_lb_args
    stripped = gpu_lb.strip_lb_args(
        ['--model', 'wav2lip', '--gpu-ids', '0,1', '--gpu-ids-strategy=least_session',
         '--lb-worker-base-port', '8011', '--listenport', '8010',
         '--use_onnx', '--avatar_id', 'a1'])
    ok(stripped == ['--model', 'wav2lip', '--use_onnx', '--avatar_id', 'a1'],
       'strip_lb_args 剔除多卡参数并保留业务参数: %s' % stripped)
    ok(gpu_lb.strip_lb_args(['--gpu-ids=0,1', '--listenport=8010']) == [],
       'strip_lb_args 支持 --opt=value 写法')
    ok(gpu_lb.strip_lb_args([]) == [], 'strip_lb_args 空参')

    # build_worker_cmd
    cmd = gpu_lb.build_worker_cmd('/x/app.py', ['--model', 'wav2lip'], 8012)
    ok(cmd[-2:] == ['--listenport', '8012'] and cmd[1] == '/x/app.py',
       'build_worker_cmd 追加 worker 端口: %s' % cmd)

    # pick_worker 用假负载
    def wk(sessions, ready=True, reserved=0):
        w = gpu_lb.Worker(0, 0, 0)
        w.sessions, w.ready, w.reserved = sessions, ready, reserved
        return w

    w0, w1, w2 = wk(0), wk(2), wk(1)
    # least_session：选负载最小
    idx, rr = gpu_lb.pick_worker([w0, w1, w2], 'least_session', 0, 2)
    ok(idx == 0 and rr == 1, 'least_session 选负载最小的卡0 (idx=%s)' % idx)
    # least_session 同负载时轮询公平
    idx, rr = gpu_lb.pick_worker([wk(0), wk(0)], 'least_session', 1, 2)
    ok(idx == 1, 'least_session 从轮询起点开始扫描 (idx=%s)' % idx)
    # 满员跳过
    full, free = wk(2), wk(0)
    idx, _ = gpu_lb.pick_worker([full, free], 'least_session', 0, 2)
    ok(idx == 1, '满员卡被跳过，选空闲卡1 (idx=%s)' % idx)
    # 全满拒绝
    idx, _ = gpu_lb.pick_worker([wk(2), wk(2)], 'least_session', 0, 2)
    ok(idx is None, '全满时返回 None')
    # round_robin 交替
    a, b = wk(0), wk(0)
    i1, rr = gpu_lb.pick_worker([a, b], 'round_robin', 0, 2)
    i2, rr = gpu_lb.pick_worker([a, b], 'round_robin', rr, 2)
    i3, rr = gpu_lb.pick_worker([a, b], 'round_robin', rr, 2)
    ok((i1, i2, i3) == (0, 1, 0), 'round_robin 交替分配: %s' % ((i1, i2, i3),))
    # round_robin 满员跳过并回绕
    i1, rr = gpu_lb.pick_worker([wk(2), wk(0), wk(0)], 'round_robin', 0, 2)
    i2, rr = gpu_lb.pick_worker([wk(2), wk(0), wk(0)], 'round_robin', rr, 2)
    ok((i1, i2) == (1, 2), 'round_robin 满员卡被跳过且按序回绕: %s' % ((i1, i2),))
    # 未就绪 worker 不可选
    idx, _ = gpu_lb.pick_worker([wk(0, ready=False), wk(0)], 'least_session', 0, 2)
    ok(idx == 1, '未就绪 worker 不可选 (idx=%s)' % idx)

    # _extract_sessionid：JSON
    ok(gpu_lb._extract_sessionid(b'{"sessionid":"s1","type":"echo"}') == 's1',
       '从 JSON body 提取 sessionid')
    ok(gpu_lb._extract_sessionid(b'not-json-bytes-here') == '',
       '非 JSON 非表单返回空')
    # _extract_sessionid：multipart（/humanaudio 形态）
    boundary = b'----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = (b'--' + boundary + b'\r\n'
            b'Content-Disposition: form-data; name="sessionid"\r\n\r\n'
            b'sess-abc-001\r\n'
            b'--' + boundary + b'\r\n'
            b'Content-Disposition: form-data; name="file"; filename="a.wav"\r\n'
            b'Content-Type: audio/wav\r\n\r\n'
            b'\x00\x01fake-audio\r\n'
            b'--' + boundary + b'--\r\n')
    ok(gpu_lb._extract_sessionid(body) == 'sess-abc-001',
       '从 multipart 表单提取 sessionid')
    ok(gpu_lb._extract_sessionid(b'') == '', '空 body 返回空')


# ─── B. Stub Worker 集成测试 ──────────────────────────────────────────────

async def _wait_gateway_ready(sess, master, n_workers, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            async with sess.get('http://127.0.0.1:%d/health' % master) as r:
                if r.status == 200:
                    data = await r.json()
                    ws = data.get('data', {}).get('workers', [])
                    if len(ws) == n_workers and all(w['ready'] for w in ws):
                        return
        except Exception:
            pass
        await asyncio.sleep(0.2)
    raise RuntimeError('gateway 未在 %ds 内就绪' % timeout)


async def _post_json(sess, url, payload):
    async with sess.post(url, json=payload) as r:
        return r.status, await r.json()


async def _get_json(sess, url):
    async with sess.get(url) as r:
        return r.status, await r.json()


def _shutdown_gateway(task):
    """优雅停止网关（stop_ev 触发 → 内部完成 worker 清理与端口释放）。"""
    try:
        if gpu_lb.state.stop_ev is not None:
            gpu_lb.state.stop_ev.set()
    except Exception:
        pass
    return task


async def scenario_least(sess, ports, max_per_gpu=2):
    print('== B1. least_session：平分 / 容量拒绝 / 亲和 / 释放后调度 / worker 自愈 ==')
    master, w0, w1 = ports
    task = asyncio.create_task(gpu_lb._amain(_opt('least_session', ports, max_per_gpu)))
    try:
        await _wait_gateway_ready(sess, master, 2)
        m = 'http://127.0.0.1:%d' % master

        # 1) 连续 4 个 offer → 两张卡各 2
        sids = []
        for _ in range(4):
            st, b = await _post_json(sess, m + '/offer', {'sdp': 'x', 'type': 'offer'})
            assert st == 200 and b.get('sessionid'), b
            sids.append(b['sessionid'])
        st, d = await _get_json(sess, m + '/health?detail=1')
        data = d['data']
        counts = {w['port']: w['sessions'] for w in data['workers']}
        route = data['route']
        ok(sorted(counts.values()) == [2, 2], '4 个会话被 least_session 平分: %s' % counts)
        ok(len(route) == 4, '路由表记录 4 个会话')

        # 2) 第 5 个 offer 被拒
        st, b = await _post_json(sess, m + '/offer', {'sdp': 'x', 'type': 'offer'})
        ok(st == 200 and b.get('code') != 0 and 'max sessions' in b.get('msg', ''),
           '第 5 个 offer 因容量上限被拒: %s' % b)

        # 3) 会话亲和：sids[0] 的 3 次 /human 全部落在同一 worker
        port_a = route[sids[0]]
        for _ in range(3):
            st, b = await _post_json(sess, m + '/human',
                                     {'sessionid': sids[0], 'type': 'echo', 'text': 'hi'})
            assert b.get('code') == 0, b
        st, d = await _get_json(sess, 'http://127.0.0.1:%d/sess/%s/hits' % (port_a, sids[0]))
        ok(d['data']['hits'] == 3, '同一 sessionid 3 次 /human 均命中同一 worker(port=%d)' % port_a)

        # 4) stop 释放容量 → 新会话被调度到该卡
        st, b = await _post_json(sess, m + '/stop_session', {'sessionid': sids[0]})
        assert b.get('code') == 0, b
        st, b = await _post_json(sess, m + '/offer', {'sdp': 'x', 'type': 'offer'})
        sid5 = b['sessionid']
        st, d = await _get_json(sess, m + '/health?detail=1')
        route = d['data']['route']
        ok(route.get(sid5) == port_a,
           '/stop_session 后新会话被调度到已空出的卡: %s' % route.get(sid5))

        # 5) kill worker1 → 映射作废 + 自动重启 + 回归服务
        w1 = gpu_lb.state.workers[1]
        sid_on_w1 = next(s for s, p in route.items() if p == w1.port)
        w1.proc.kill()
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            await asyncio.sleep(0.3)
            st, d = await _get_json(sess, m + '/health?detail=1')
            workers = d['data']['workers']
            if workers[1]['ready'] and workers[1]['sessions'] == 0:
                break
        ok(workers[1]['ready'] and workers[1]['sessions'] == 0,
           '被 kill 的 worker 已自动重启并回归服务')
        ok(sid_on_w1 not in d['data']['route'],
           '被 kill worker 的会话映射已作废')
        st, b = await _post_json(sess, m + '/human',
                                 {'sessionid': sid_on_w1, 'type': 'echo', 'text': 'x'})
        ok('session not found' in b.get('msg', ''), '已作废会话后续请求返回 session not found')
        st, b = await _post_json(sess, m + '/offer', {'sdp': 'x', 'type': 'offer'})
        st, d = await _get_json(sess, m + '/health?detail=1')
        ok(d['data']['route'].get(b['sessionid']) == w1.port,
           '重启后的 worker 重新参与会话分配')
    finally:
        await _shutdown_gateway(task)
        try:
            await asyncio.wait_for(task, 30)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()


async def scenario_rr(sess, ports):
    print('== B2. round_robin：新会话交替分配 ==')
    master, w0, w1 = ports
    task = asyncio.create_task(gpu_lb._amain(_opt('round_robin', ports)))
    try:
        await _wait_gateway_ready(sess, master, 2)
        m = 'http://127.0.0.1:%d' % master
        # worker 端口以网关实际分配为准（free port 不保证连续，不能假设 == ports[1]/ports[2]）
        st, d = await _get_json(sess, m + '/health')
        ws = d['data']['workers']
        pw0, pw1 = ws[0]['port'], ws[1]['port']
        ports_alloc = []
        for _ in range(3):
            st, b = await _post_json(sess, m + '/offer', {'sdp': 'x', 'type': 'offer'})
            assert b.get('sessionid'), b
            st, d = await _get_json(sess, m + '/health?detail=1')
            ports_alloc.append(d['data']['route'][b['sessionid']])
        expected = [pw0, pw1, pw0]
        ok(ports_alloc == expected,
           'round_robin 交替分配期望=%s 实际=%s' % (expected, ports_alloc))
    finally:
        await _shutdown_gateway(task)
        try:
            await asyncio.wait_for(task, 30)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            task.cancel()


async def main_test():
    ports = _free_ports(3)
    async with aiohttp.ClientSession() as sess:
        await scenario_least(sess, ports)
        await scenario_rr(sess, ports)


if __name__ == '__main__':
    unit_tests()
    print('== B. Stub Worker 集成测试 ==')
    asyncio.run(main_test())
    print('\n结果: PASS=%d FAIL=%d' % (PASS, FAIL))
    sys.exit(0 if FAIL == 0 else 1)
