###############################################################################
#  全局会话管理器 (Session Manager)
###############################################################################

import asyncio
import uuid
from typing import Dict, Optional
from utils.logger import logger
from avatars.base_avatar import BaseAvatar

def _rand_session_id() -> str:
    """生成 UUID session ID"""
    return str(uuid.uuid4())

class SessionManager:
    """
    全局数字人会话管理器。
    
    统一管理 avatar_sessions 生命周期，并在脱离 WebRTC 时依然保持服务可用。
    """
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "initialized"):
            self.sessions: Dict[str, BaseAvatar] = {}
            self.build_session_fn = None
            # 会话租约（WS 降级通道用）：被租用的会话在 WebRTC pc 关闭时不销毁
            self.leases: Dict[str, int] = {}
            # pc 关闭时发现有租约 → 记下“待销毁”，等租约全部释放后再补做
            self.pending_destroy: set = set()
            self.initialized = True

    def init_builder(self, build_session_fn):
        """配置用于构建 avatar_session 的工厂函数"""
        self.build_session_fn = build_session_fn
        
    def get_session(self, sessionid: str) -> Optional[BaseAvatar]:
        """获取已存活的会话"""
        return self.sessions.get(sessionid)

    def has_session(self, sessionid: str) -> bool:
        """检查会话是否存在"""
        return sessionid in self.sessions and self.sessions[sessionid] is not None
        
    async def create_session(self, params: dict, sessionid: str = None) -> str:
        """
        在异步环境中创建一个新会话
        如果 sessionid 为 None，则自动生成。
        """
        if self.build_session_fn is None:
            raise Exception("SessionManager builder not initialized")

        if sessionid is None:
            sessionid = _rand_session_id()

        logger.info('Creating sessionid=%s, current session num=%d', sessionid, len(self.sessions))
        # 预先占位防止重复
        self.sessions[sessionid] = None

        try:
            # 在线程池中构建 session（加载模型非常耗时）
            avatar_session = await asyncio.get_event_loop().run_in_executor(
                None, self.build_session_fn, sessionid, params
            )
            self.sessions[sessionid] = avatar_session
            return sessionid
        except BaseException:
            # 构建失败或超时取消时清理占位，避免留下 None 导致后续访问异常。
            # 注意：asyncio.CancelledError（超时取消）继承自 BaseException，
            # 普通 `except Exception` 抓不到，会残留占位导致 sessions 计数只增不减。
            logger.exception('Failed to create session %s', sessionid)
            self.sessions.pop(sessionid, None)
            raise
        
    def add_session(self, sessionid: str, avatar_session: BaseAvatar):
        """同步添加静态或外部管理的会话（供非服务端入口调用）"""
        self.sessions[sessionid] = avatar_session
        
    def stop_session(self, sessionid: str):
        """显式停止会话资源，但不删除引用"""
        avatar_session = self.sessions.get(sessionid)
        if avatar_session is None:
            return
        logger.info(f"Stopping session {sessionid}")
        if hasattr(avatar_session, 'request_stop'):
            avatar_session.request_stop()
        elif hasattr(avatar_session, 'quit_event'):
            avatar_session.quit_event.set()

    def remove_session(self, sessionid: str):
        """销毁会话资源"""
        self.stop_session(sessionid)
        if sessionid in self.sessions:
            logger.info(f"Removing session {sessionid}")
            self.sessions.pop(sessionid, None)

    # ─── 会话租约（WS 降级通道专用）──────────────────────────────────
    def lease_session(self, sessionid: str):
        """标记会话被外部（WS 降级通道）占用，期间 WebRTC pc 关闭不销毁该会话。"""
        self.leases[sessionid] = self.leases.get(sessionid, 0) + 1

    def is_leased(self, sessionid: str) -> bool:
        return self.leases.get(sessionid, 0) > 0

    def mark_pending_destroy(self, sessionid: str):
        """pc 关闭时发现会话仍被租用，记录待销毁，待租约释放后再清理。"""
        self.pending_destroy.add(sessionid)

    def release_session(self, sessionid: str) -> bool:
        """释放一次租约；返回 True 表示此前有被推迟的销毁需要调用方立刻执行。"""
        cnt = self.leases.get(sessionid, 0)
        if cnt > 1:
            self.leases[sessionid] = cnt - 1
            return False
        self.leases.pop(sessionid, None)
        if sessionid in self.pending_destroy:
            self.pending_destroy.discard(sessionid)
            return True
        return False

# 单例抛出
session_manager = SessionManager()
