# -*- coding: utf-8 -*-
###############################################################################
#  gpu_lb_stub_worker.py — LB 集成测试用的 Stub Worker（方案 §7.1 方法 1）
#
#  用几十行 aiohttp 模拟 zl-talking 单实例对外的关键 HTTP 行为，供 master 网关在
#  无 GPU/无模型的环境下做端到端集成测试。启动方式与真实 worker 完全一致：
#
#      python gpu_lb_stub_worker.py <原参数…> --listenport <port>
#
#  argv 中其它参数（--model 等）一律忽略。由 master 注入 ZL_LB_WORKER_APP=本文件
#  即可替换真实 worker，不影响 master 任何逻辑。
#
#  对齐真实 worker 的行为：
#      POST /offer           → 每次生成新 sessionid，返回顶层 {"sessionid": ...}
#      POST /human           → 按 sessionid 累计命中次数，返回 {"code":0,...}
#      POST /interrupt_talk  → 同 /human（计入 hits）
#      POST /is_speaking     → {"code":0,...,"data":true}
#      POST /stop_session    → 删除会话，返回 {"code":0,...}
#      POST /warmup          → {"code":0,...}
#      GET  /health          → {"code":0,...,"data":{"sessions":N}}
#      GET  /sess/{sid}/hits → {"code":0,...,"data":{"hits":H,"port":P}}（测试专用）
#  未识别/缺 sessionid 的会话类请求 → {"code":-1,"msg":"session not found"}
###############################################################################

import sys
import asyncio

from aiohttp import web


def _parse_listenport():
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--listenport' and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith('--listenport='):
            return int(a.split('=', 1)[1])
    return 0


class StubWorker:
    def __init__(self, port):
        self.port = port
        self.sessions = {}   # sid -> hits
        self.seq = 0

    # ── 路由处理 ──────────────────────────────────────────────────────
    async def offer(self, request):
        self.seq += 1
        sid = 'sess-%05d-%03d' % (self.port, self.seq)
        self.sessions[sid] = 0
        # 与真实 /offer 一致：成功响应无 code 包裹，sessionid 在顶层
        return web.json_response({'sdp': 'stub-sdp', 'type': 'offer',
                                  'sessionid': sid})

    async def hit(self, request):
        params = await request.json()
        sid = params.get('sessionid', '')
        if sid not in self.sessions:
            return web.json_response({'code': -1, 'msg': 'session not found'})
        self.sessions[sid] += 1
        return web.json_response({'code': 0, 'msg': 'ok'})

    async def is_speaking(self, request):
        params = await request.json()
        sid = params.get('sessionid', '')
        if sid not in self.sessions:
            return web.json_response({'code': -1, 'msg': 'session not found'})
        return web.json_response({'code': 0, 'msg': 'ok', 'data': True})

    async def stop_session(self, request):
        params = await request.json()
        sid = params.get('sessionid', '')
        if sid in self.sessions:
            del self.sessions[sid]
        return web.json_response({'code': 0, 'msg': 'ok'})

    async def warmup(self, request):
        return web.json_response({'code': 0, 'msg': 'ok'})

    async def health(self, request):
        return web.json_response({'code': 0, 'msg': 'ok',
                                  'data': {'sessions': len(self.sessions)}})

    async def hits(self, request):
        sid = request.match_info['sid']
        if sid not in self.sessions:
            return web.json_response({'code': -1, 'msg': 'session not found'})
        return web.json_response({'code': 0, 'msg': 'ok',
                                  'data': {'hits': self.sessions[sid],
                                           'port': self.port}})

    async def not_found(self, request):
        return web.json_response({'code': -1, 'msg': 'not found'})


def main():
    port = _parse_listenport()
    stub = StubWorker(port)
    app = web.Application()
    app.router.add_post('/offer', stub.offer)
    app.router.add_post('/human', stub.hit)
    app.router.add_post('/interrupt_talk', stub.hit)
    app.router.add_post('/is_speaking', stub.is_speaking)
    app.router.add_post('/stop_session', stub.stop_session)
    app.router.add_post('/warmup', stub.warmup)
    app.router.add_get('/health', stub.health)
    app.router.add_get('/sess/{sid}/hits', stub.hits)
    app.router.add_route('*', '/{tail:.*}', stub.not_found)
    web.run_app(app, host='127.0.0.1', port=port, print=None, access_log=None)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
