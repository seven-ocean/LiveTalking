# 为 wav2lip genavatar 人脸检测添加详细日志

## 现状

执行 `python -m avatars.wav2lip.genavatar` 时，在 `正在检测人脸...` 后卡住，GPU 占用持续 90%+ 长达 1 小时。现有日志仅有 `print('正在检测人脸...')`，无法定位瓶颈。

## 执行链路分析

```
genavatar.py:generate_avatar()
  ├── print('正在检测人脸...')                          ← 最后可见的日志
  ├── detector = face_detection.FaceAlignment(...)    ← 初始化检测器（加载S3FD模型权重）
  └── for each batch:                                  ← 521帧, batch_size=16, 约33批次
       └── detector.get_detections_for_batch(frames)
            └── api.py:FaceAlignment.get_detections_for_batch()
                 └── sfd_detector.py:SFDDetector.detect_from_batch()
                      └── detect.py:batch_detect()      
                           ├── 数据预处理 (归一化 + transpose)
                           ├── net(imgs) 模型推理       ← GPU 推理部分
                           └── CPU 解码循环              ← 遍历6个特征层，解码候选框
```

## 修改计划

### 1. genavatar.py — 添加高层级日志

**文件**: `f:\Projects\AI-Human\zl-talking\avatars\wav2lip\genavatar.py`

在 `generate_avatar()` 函数中 `print('正在检测人脸...')` 前后和批次循环中添加时间日志：

- **检测器初始化前后**：记录开始/完成时间，检测初始化阶段是否耗时（可能因模型下载/加载导致）
- **每个批次的起始/结束**：记录 `[batch X/Y] 帧 i~i+batch_size, 耗时 T 秒`
- **每帧日志**（可选，默认关）：添加 `LOG_EACH_FRAME` 开关，开启后逐帧输出日志

### 2. detect.py — 添加 `batch_detect()` 内部详细日志

**文件**: `f:\Projects\AI-Human\zl-talking\avatars\wav2lip\face_detection\detection\sfd\detect.py`

在 `batch_detect()` 函数中添加：

- **函数入口**：记录 `imgs.shape`（BB, HH, WW）
- **模型推理前后**：记录推理耗时
- **CPU 解码阶段**：记录每层的候选框数量 `len(poss)` 和该层解码耗时
- **函数出口**：记录 `bboxlist.shape` 和函数总耗时

### 3. sfd_detector.py — 添加 `detect_from_batch()` 和 `detect_from_image()` 日志

**文件**: `f:\Projects\AI-Human\zl-talking\avatars\wav2lip\face_detection\detection\sfd\sfd_detector.py`

- `detect_from_batch()` 入口/出口：记录 batch size 和耗时
- `detect_from_image()` 入口/出口（辅助排查单帧情况）

### 4. api.py — 添加 `get_detections_for_batch()` 日志

**文件**: `f:\Projects\AI-Human\zl-talking\avatars\wav2lip\face_detection\api.py`

- `get_detections_for_batch()` 入口：记录 batch 图像数量
- 结果返回前：记录检测到的人脸数

### 5. config.py — 统一日志开关

**文件**: `f:\Projects\AI-Human\zl-talking\config.py`

在文件末尾添加全局调试配置（如果该文件已有配置则追加）：

```python
# ---- 调试日志开关 ----
DEBUG_WAV2LIP_FACE_DETECTION = True   # 设为 False 可关闭详细日志
```

各文件通过 `from config import DEBUG_WAV2LIP_FACE_DETECTION` 引用此开关。

## 执行计划

| 步骤 | 文件 | 改动 |
|------|------|------|
| 1 | config.py | 添加 `DEBUG_WAV2LIP_FACE_DETECTION` 全局开关 |
| 2 | detect.py | `batch_detect()` 添加入口/模型推理/解码循环/出口各阶段日志 |
| 3 | sfd_detector.py | `detect_from_batch()` 添加入口/出口日志 |
| 4 | api.py | `get_detections_for_batch()` 添加入口/出口日志 |
| 5 | genavatar.py | 初始化前后 + 批次循环添加时间日志 |

## 验证

修改后重新运行：
```bash
python -m avatars.wav2lip.genavatar --video_path data/wav2lip256_avatar3.mp4 --img_size 256 --avatar_id wav2lip256_avatar3
```

观察日志输出，确认以下信息：
- 检测器初始化耗时（排查模型加载是否卡住）
- 每个批次的单次耗时（排查是否某批次异常慢）
- 模型推理 vs CPU 解码的耗时比例（定位瓶颈在 GPU 还是 CPU）
- 每层候选框数量（排查是否候选框过多导致解码死循环）
