# -*- coding: utf-8 -*-
###############################################################################
#  启动授权校验（授权加固 · 批次 4）
#  zl-talking 启动时向 zl-ai 主包请求 license 状态，授权无效则拒绝启动：
#    - GET {ZL_AI_SERVER}/aihuman/license/status
#    - HTTP 200 且响应体 {"code":200,...,"data":{"valid":true}} → 通过
#    - 校验失败重试 2 次（间隔 2s），仍失败 → 进程退出（sys.exit(1)）
#    - 通过 CLI --license_check 或环境变量 ZL_LICENSE_CHECK 开启（默认关闭，
#      避免未接入 zl-ai 的现有部署直接启动失败）
###############################################################################

import asyncio
import sys
import time

import aiohttp

from utils.logger import logger


def _do_check(url: str, timeout: float) -> bool:
    """单次请求 license 状态接口，返回是否有效。"""

    async def _req():
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout)
            ) as resp:
                if resp.status != 200:
                    logger.warning("license check http status=%s", resp.status)
                    return False
                body = await resp.json(content_type=None)
                code = body.get("code")
                data = body.get("data") or {}
                return code == 200 and data.get("valid") is True

    try:
        return asyncio.run(_req())
    except Exception as e:  # 网络异常/超时等均视为校验失败
        logger.warning("license check request failed: %s", e)
        return False


def check_license(server: str, timeout: float = 5.0, retries: int = 2) -> bool:
    """带重试的授权校验：成功返回 True，全部失败返回 False。"""
    url = server.rstrip("/") + "/aihuman/license/status"
    logger.info("license check -> %s", url)
    for attempt in range(retries + 1):
        if _do_check(url, timeout):
            logger.info("license check passed (attempt=%s)", attempt + 1)
            return True
        if attempt < retries:
            time.sleep(2)
    return False


def run_license_check(opt) -> None:
    """由 app.main() 在 parse_args 之后、加载模型之前调用。

    - opt.license_check 为 False（默认）时跳过，保持现有部署行为；
    - 校验失败直接退出进程，阻止 zl-talking 在未授权环境下运行。
    """
    if not getattr(opt, "license_check", False):
        logger.info("license check disabled (--license_check=false)")
        return

    server = getattr(opt, "ZL_AI_SERVER", "") or "http://127.0.0.1:9306"
    timeout = float(getattr(opt, "license_check_timeout", 5.0))
    if not check_license(server, timeout=timeout, retries=2):
        logger.error(
            "license check FAILED -> zl-ai 授权无效或不可达（%s），进程退出。"
            "请确认 zl-ai 已启动且授权有效，或关闭 --license_check 后重试。",
            server,
        )
        sys.exit(1)
