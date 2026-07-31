import logging
import logging.handlers
import os
import sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

fhandler = logging.FileHandler('zl-talking.log')
fhandler.setFormatter(formatter)
fhandler.setLevel(logging.INFO)
logger.addHandler(fhandler)

shandler = logging.StreamHandler(sys.stdout)
shandler.setFormatter(formatter)
shandler.setLevel(logging.INFO)
logger.addHandler(shandler)


# ─── WebRTC 专用日志 ─────────────────────────────────────────────
# 独立 logger，记录更详细的 ICE / 信令 / 连接状态，便于问题排查

os.makedirs('logs', exist_ok=True)

webrtc_formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - [%(sessionid)s] %(message)s'
)


class _SessionContextFilter(logging.Filter):
    """允许在 extra 中传入 sessionid，缺失时显示为 '-'。"""

    def filter(self, record):
        if not hasattr(record, 'sessionid'):
            record.sessionid = '-'
        return True


webrtc_logger = logging.getLogger('zl-talking.webrtc')
webrtc_logger.setLevel(logging.DEBUG)
webrtc_logger.addFilter(_SessionContextFilter())

# 滚动文件：单个 10MB，保留 5 个备份
webrtc_fhandler = logging.handlers.RotatingFileHandler(
    'logs/zl-talking-webrtc.log',
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    encoding='utf-8',
)
webrtc_fhandler.setFormatter(webrtc_formatter)
webrtc_fhandler.setLevel(logging.DEBUG)
webrtc_logger.addHandler(webrtc_fhandler)

webrtc_shandler = logging.StreamHandler(sys.stdout)
webrtc_shandler.setFormatter(webrtc_formatter)
webrtc_shandler.setLevel(logging.DEBUG)
webrtc_logger.addHandler(webrtc_shandler)


def get_webrtc_logger():
    """返回 WebRTC 专用 logger，调用时可通过 extra={'sessionid': 'xxx'} 传入会话 ID。"""
    return webrtc_logger


def set_webrtc_log_level(level: str):
    """动态调整 WebRTC 日志级别。"""
    level = level.upper()
    if level not in ('DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'CRITICAL'):
        logger.warning('Invalid webrtc log level: %s, fallback to DEBUG', level)
        level = 'DEBUG'
    webrtc_logger.setLevel(getattr(logging, level))
    for h in webrtc_logger.handlers:
        h.setLevel(getattr(logging, level))
