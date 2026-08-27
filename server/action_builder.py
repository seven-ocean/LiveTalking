# -*- coding: utf-8 -*-
###############################################################################
#  动作片段处理核心（动作编排 · 形象创建操作台）
#  ---------------------------------------------------------------------------
#  职责：
#   - 从主视频按 [start, end] 时间戳用 ffmpeg 提取动作帧序列（%08d.png）
#   - 讲解/思考动作额外提取 16k mono wav 音频（与 base_asr 音频管线一致）
#   - 生成 per-avatar 动作配置 data/avatars/{avatar_id}/actions.json
#   - 生成形象元信息 data/avatars/{avatar_id}/avatar.json
###############################################################################

import json
import os
import shutil
import subprocess
import time

from utils.logger import logger

# 动作名 -> audiotype（与现状 data/custom_config.json 语义保持一致）
AUDIOTYPE_MAP = {"idle": 1, "reply": 2, "think": 3}
# 动作名 -> actions/ 下的子目录名
ACTION_DIRS = {"idle": "idle", "reply": "reply", "think": "think"}
FPS = 25  # 动作帧序列帧率

# 需要提取音频的动作（待机 idle 无音频，静默自动循环）
AUDIO_ACTIONS = {"reply", "think"}


def _run_ffmpeg(cmd, purpose):
    """执行 ffmpeg/ffprobe 命令，失败抛出带 stderr 信息的异常。"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except FileNotFoundError:
        raise RuntimeError(
            f"[action] {purpose} 失败: 未检测到 ffmpeg，请先安装 ffmpeg 并加入 PATH"
        ) from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"[action] {purpose} 超时（600s）") from None
    if result.returncode != 0:
        raise RuntimeError(f"[action] {purpose} 失败: {result.stderr.strip()[:500]}")
    return result


def get_video_duration(video_path):
    """用 ffprobe 读取视频时长（秒）。ffprobe 不可用时返回 None（跳过时长校验）。"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"[action] ffprobe 读取时长失败，跳过时长校验: {e}")
    return None


def extract_frames(video_path, start, end, out_dir):
    """从视频 [start, end] 秒提取帧序列到 out_dir（%08d.png，25fps）。"""
    os.makedirs(out_dir, exist_ok=True)
    # 重新生成前清空旧帧，避免编辑场景残留
    for name in os.listdir(out_dir):
        if name.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(out_dir, name))
    duration = max(end - start, 0.1)
    cmd = ["ffmpeg", "-y",
           "-ss", str(start), "-i", video_path,
           "-t", str(duration),
           "-vf", f"fps={FPS}",
           os.path.join(out_dir, "%08d.png")]
    _run_ffmpeg(cmd, f"提取动作帧 [{start}s~{end}s]")
    frame_count = len([n for n in os.listdir(out_dir) if n.lower().endswith(".png")])
    if frame_count == 0:
        raise RuntimeError(f"[action] 提取动作帧失败: 片段 [{start}s~{end}s] 未提取到任何帧")
    return frame_count


def extract_audio(video_path, start, end, out_wav):
    """从视频 [start, end] 秒提取 16k mono wav（与 base_asr 音频管线一致）。"""
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    duration = max(end - start, 0.1)
    cmd = ["ffmpeg", "-y",
           "-ss", str(start), "-i", video_path,
           "-t", str(duration),
           "-ac", "1", "-ar", "16000",
           out_wav]
    _run_ffmpeg(cmd, f"提取动作音频 [{start}s~{end}s]")


def _validate_actions(actions, duration):
    """校验动作片段参数，返回 {name: (start, end)}；非法项抛异常。"""
    result = {}
    for name, seg in (actions or {}).items():
        if name not in AUDIOTYPE_MAP:
            raise ValueError(f"[action] 未知动作: {name}（仅支持 idle/reply/think）")
        try:
            start = float(seg["start"])
            end = float(seg["end"])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"[action] 动作 {name} 缺少有效的 start/end") from None
        if start < 0:
            raise ValueError(f"[action] 动作 {name} 起点不能为负: {start}")
        if end <= start:
            raise ValueError(f"[action] 动作 {name} 终点必须大于起点: {start} ~ {end}")
        if duration is not None and end > duration:
            raise ValueError(f"[action] 动作 {name} 终点 {end}s 超出视频时长 {duration:.2f}s")
        result[name] = (start, end)
    return result


def build_avatar_actions(avatar_id, video_path, actions, save_path="./data/avatars",
                         progress_callback=None):
    """按标记片段生成形象动作，并写入 actions.json。

    Args:
        avatar_id: 形象 ID
        video_path: 主视频路径
        actions: {"idle": {"start": s, "end": e}, "reply": {...}, "think": {...}}（秒，可部分缺失）
        save_path: 形象根目录（默认 ./data/avatars）
        progress_callback: 可选，接收 0-100 整数
    """
    if not os.path.exists(video_path):
        raise ValueError(f"[action] 视频文件不存在: {video_path}")

    duration = get_video_duration(video_path)
    validated = _validate_actions(actions, duration)

    avatar_dir = os.path.join(save_path, avatar_id)
    actions_dir = os.path.join(avatar_dir, "actions")
    os.makedirs(actions_dir, exist_ok=True)

    items = []
    total = len(validated)
    for i, (name, (start, end)) in enumerate(validated.items()):
        sub_dir = os.path.join(actions_dir, ACTION_DIRS[name])
        rel_path = os.path.join("data", "avatars", avatar_id, "actions", ACTION_DIRS[name])

        logger.info(f"[action] 生成动作 {name}(audiotype={AUDIOTYPE_MAP[name]}): [{start}s~{end}s]")
        extract_frames(video_path, start, end, sub_dir)

        item = {"audiotype": AUDIOTYPE_MAP[name], "imgpath": rel_path}
        if name in AUDIO_ACTIONS:
            wav_name = f"{ACTION_DIRS[name]}.wav"
            wav_path = os.path.join(sub_dir, wav_name)
            extract_audio(video_path, start, end, wav_path)
            item["audiopath"] = os.path.join(rel_path, wav_name)
        items.append(item)

        if progress_callback:
            progress_callback(int(85 + (i + 1) / total * 15))  # 85 -> 100

    actions_file = os.path.join(avatar_dir, "actions.json")
    with open(actions_file, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)
    logger.info(f"[action] 已写入动作配置: {actions_file} ({len(items)} 项)")
    return items


def build_avatar_meta(avatar_id, name, model, actions, save_path="./data/avatars"):
    """写入形象元信息 data/avatars/{avatar_id}/avatar.json。"""
    avatar_dir = os.path.join(save_path, avatar_id)
    os.makedirs(avatar_dir, exist_ok=True)
    meta = {
        "avatar_id": avatar_id,
        "name": name or avatar_id,
        "model": model or "",
        "created_at": time.time(),
        "actions": actions or {},
    }
    meta_file = os.path.join(avatar_dir, "avatar.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
    logger.info(f"[action] 已写入形象元信息: {meta_file}")
    return meta
