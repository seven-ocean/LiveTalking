# -*- coding: utf-8 -*-
"""进程级模型切换信号。

请求处理线程里绝不能直接 os._exit：连接会被 reset，上游 relay 拿不到完整
answer，前端表现为「sdp 未设置」或永久加载。这里只抛异常，由 RTCManager
先返回 503，再安排进程退出。
"""


class ModelSwitchRequired(RuntimeError):
    def __init__(self, avatar_id: str):
        super().__init__(avatar_id)
        self.avatar_id = avatar_id
