# -*- coding: utf-8 -*-
###############################################################################
#  入站鉴权中间件（授权加固 · 批次 4）
#  zl-talking 对来自 zl-ai（主包）及其他调用方的接口做 X-Api-Key 统一鉴权：
#    - 环境变量 ZL_API_KEY 未设置/为空 → 关闭鉴权（默认放行，兼容现有部署）
#    - 请求头 X-Api-Key 常量时间比较（hmac.compare_digest）
#      缺失/不匹配 → 401 {"code":401,"msg":"UNAUTHORIZED"}
#    - 白名单路径（/ /health /config）与非 API 静态资源（web 目录）放行
#  与 zl-ai 能力中心对接：在能力配置的「请求头(Header)」填
#  X-Api-Key: {{secret.xxx}}，出站请求自动携带该头即可通过校验。
###############################################################################

import hmac
import logging
import os

from aiohttp import web

logger = logging.getLogger("zl_auth")

# 未设置 ZL_API_KEY 或为空时关闭鉴权（保持现状行为，避免加固后断链）
_ZL_API_KEY = os.getenv("ZL_API_KEY", "").strip()

# 显式放行的白名单路径（健康检查 / 前端页面初始化配置 / 首页）
_WHITELIST_PATHS = {"/", "/health", "/config"}

# 需鉴权的 API 路径前缀（其余路径视为静态资源，直接放行）
_API_PREFIXES = (
    "/offer",
    "/human",
    "/humanaudio",
    "/set_audiotype",
    "/record",
    "/interrupt_talk",
    "/is_speaking",
    "/stop_session",
    "/warmup",
    "/api/admin",
)


def make_auth_middleware():
    """构造 aiohttp 入站鉴权中间件（注册到 app.middlewares）。"""

    @web.middleware
    async def auth_middleware(request: web.Request, handler):
        # 未配置 ZL_API_KEY 时关闭鉴权
        if not _ZL_API_KEY:
            return await handler(request)

        # CORS 预检请求放行（由 aiohttp_cors 处理）
        if request.method == "OPTIONS":
            return await handler(request)

        path = request.rel_url.path

        # 白名单路径放行
        if path in _WHITELIST_PATHS:
            return await handler(request)

        # 非 API 路径（web/ 静态资源）放行
        if not path.startswith(_API_PREFIXES):
            return await handler(request)

        # X-Api-Key 校验（常量时间比较）
        provided = request.headers.get("X-Api-Key", "")
        if not hmac.compare_digest(provided, _ZL_API_KEY):
            logger.warning("ZLAuth: X-Api-Key 校验失败 path=%s", path)
            return web.json_response(
                {"code": 401, "msg": "UNAUTHORIZED"}, status=401
            )

        return await handler(request)

    return auth_middleware
