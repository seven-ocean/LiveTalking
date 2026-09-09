###############################################################################
#  配置解析 — CLI 参数 + YAML 配置
###############################################################################

import argparse
import json
import os


def str_to_bool(value):
    """将字符串转换为 bool，支持 True/False/1/0"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    return bool(value)


def str_or_int(value):
    """尝试转换为 int，失败则返回 str"""
    try:
        return int(value)
    except ValueError:
        return value


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="zl-talking Digital Human Server")

    # ─── 音频 ──────────────────────────────────────────────────────────
    parser.add_argument('--fps', type=int, default=25, help="video fps, must be 25")
    parser.add_argument('-l', type=int, default=10)
    parser.add_argument('-m', type=int, default=8)
    parser.add_argument('-r', type=int, default=10)

    # ─── 画面 ──────────────────────────────────────────────────────────
    # parser.add_argument('--W', type=int, default=450, help="GUI width")
    # parser.add_argument('--H', type=int, default=450, help="GUI height")

    # ─── 数字人模型 ────────────────────────────────────────────────────
    parser.add_argument('--model', type=str, default='wav2lip',
                        help="avatar model: musetalk/wav2lip/ultralight")
    parser.add_argument('--avatar_id', type=str, default='wav2lip256_avatar1',
                        help="avatar id in data/avatars")
    parser.add_argument('--batch_size', type=int, default=16, help="infer batch")
    parser.add_argument('--modelres', type=int, default=192)
    parser.add_argument('--modelfile', type=str, default='')
    parser.add_argument('--use_onnx', action='store_true',
                        help="use ONNX model instead of PyTorch (default: False)")
    parser.add_argument('--onnx_model_path', type=str, default='',
                        help="path to ONNX model file (default: auto-select based on --model)")

    # ─── 自定义动作和多形象 ────────────────────────────────────────────
    parser.add_argument('--customvideo_config', type=str, default='',
                        help="custom action json。每项结构: {\"audiotype\": 2, \"imgpath\": \"data/actions/wave\", "
                             "\"audiopath\": \"data/actions/wave.wav\"(可选), \"avatar_id\": \"avatar1\"(可选，"
                             "空/缺省=全局动作，指定=仅该形象可用)}。形象本地 data/avatars/{avatar_id}/actions.json 优先于此配置")

    # ─── TTS ───────────────────────────────────────────────────────────
    parser.add_argument('--tts', type=str, default='edgetts',
                        help="tts plugin: edgetts/gpt-sovits/cosyvoice/fishtts/tencent/doubao/indextts2/azuretts/qwentts")
    parser.add_argument('--REF_FILE', type=str, default="zh-CN-YunxiaNeural",
                        help="参考文件名或语音模型ID")
    parser.add_argument('--REF_TEXT', type=str, default=None)
    parser.add_argument('--TTS_SERVER', type=str, default='http://127.0.0.1:9880')
    parser.add_argument('--COSYVOICE_STREAMING', type=str_to_bool, default=False,
                        help="启用 CosyVoice 流式推理（默认 False，传 True 开启）")
    parser.add_argument('--COSYVOICE_STREAM_CHUNK', type=int, default=9600,
                        help="流式音频块大小（字节），默认 9600 (约 200ms @ 24kHz)")

    # ─── 传输 ─────────────────────────────────────────────────────────
    parser.add_argument('--transport', type=str, default='webrtc',
                        help="output: rtcpush/webrtc/rtmp/virtualcam")
    parser.add_argument('--push_url', type=str,
                        default='http://localhost:1985/rtc/v1/whip/?app=live&stream=livestream')
    parser.add_argument('--max_session', type=int, default=1)
    parser.add_argument('--listenport', type=int, default=8010,
                        help="web listen port")

    # ─── WebRTC / ICE / TURN ───────────────────────────────────────────
    parser.add_argument(
        '--ice_servers',
        type=str,
        default='[{"urls": "stun:stun.freeswitch.org:3478"}]',
        help="JSON 格式的 ICE 服务器列表，可包含 STUN/TURN，例如 "
             '[{"urls":"stun:stun.example.com:3478"},'
             '{"urls":"turn:turn.example.com:3478","username":"u","credential":"p"}]',
    )
    parser.add_argument(
        '--offer_timeout',
        type=float,
        default=30.0,
        help="服务端处理 /offer 的最大超时时间（秒），默认 30",
    )
    parser.add_argument(
        '--ice_gathering_timeout',
        type=float,
        default=5.0,
        help="客户端 ICE 收集超时时间（秒），默认 5。超时后将使用已收集的候选发送 offer",
    )
    parser.add_argument(
        '--webrtc_log_level',
        type=str,
        default='DEBUG',
        choices=['DEBUG', 'INFO', 'WARNING', 'WARN', 'ERROR', 'CRITICAL'],
        help="WebRTC 专用日志级别，默认 DEBUG",
    )

    # ─── 授权加固（批次 4）─────────────────────────────────────────────
    parser.add_argument('--ZL_AI_SERVER', type=str,
                        default=os.getenv('ZL_AI_SERVER', 'http://127.0.0.1:9306'),
                        help="zl-ai 主包地址（启动授权校验接口 /aihuman/license/status）")
    parser.add_argument('--license_check', type=str_to_bool,
                        default=str_to_bool(os.getenv('ZL_LICENSE_CHECK', 'False')),
                        help="启动时校验 zl-ai 授权（True/False），默认 False")
    parser.add_argument('--license_check_timeout', type=float,
                        default=float(os.getenv('ZL_LICENSE_CHECK_TIMEOUT', '5')),
                        help="启动授权校验单次请求超时（秒），默认 5")

    # ─── 一机多卡负载均衡（多卡网关模式参数；--gpu-ids 留空则完全不影响单卡）─────────
    parser.add_argument('--gpu-ids', type=str, default='',
                        help="参与负载均衡的物理卡子集，如 '0,1'；留空=单卡模式（原逻辑不变）")
    parser.add_argument('--gpu-ids-strategy', type=str, default='least_session',
                        choices=['least_session', 'round_robin'],
                        help="选卡策略：least_session（空闲优先，默认）/ round_robin（轮询）")
    parser.add_argument('--lb-worker-base-port', type=int, default=8011,
                        help="多卡模式 worker 起始端口（第 i 张卡 → base_port + i）")
    parser.add_argument('--lb-max-session-per-gpu', type=int, default=2,
                        help="每卡最多同时分配的新会话数（达到上限时选卡跳过；全满则报错）")
    parser.add_argument('--lb-health-interval', type=int, default=5,
                        help="worker 健康/负载轮询间隔（秒）")
    parser.add_argument('--lb-worker-wait-timeout', type=float, default=60.0,
                        help="等待 worker 模型就绪超时（秒）")

    opt = parser.parse_args()

    # ─── 后处理 ────────────────────────────────────────────────────────
    # 解析多卡物理卡子集：'0,1' → [0,1]；留空 → []（单卡模式，不启网关）
    opt.gpu_id_list = [int(x) for x in opt.gpu_ids.split(',') if x.strip()]

    opt.customopt = []
    if opt.customvideo_config:
        with open(opt.customvideo_config, 'r') as f:
            opt.customopt = json.load(f)

    # 解析 ICE 服务器 JSON
    try:
        opt.ice_servers = json.loads(opt.ice_servers)
        if not isinstance(opt.ice_servers, list):
            raise ValueError('ice_servers must be a JSON list')
    except Exception as e:
        print(f"[WARN] 解析 --ice_servers 失败 ({e})，使用默认 STUN 服务器")
        opt.ice_servers = [{"urls": "stun:stun.freeswitch.org:3478"}]

    # 应用 WebRTC 日志级别
    from utils.logger import set_webrtc_log_level
    set_webrtc_log_level(opt.webrtc_log_level)

    return opt


# ---- 调试日志开关 ----
DEBUG_WAV2LIP_FACE_DETECTION = True   # wav2lip genavatar 人脸检测详细日志，定位问题后设为 False
