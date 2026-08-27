# Avatar 生成接口 API 文档

基础路径：`http://<host>:<listenport>`

---

## 1. 创建 Avatar 生成任务

```
POST /api/avatar/task
```

**Content-Type**: `application/json` 或 `multipart/form-data`（上传文件时）

| 参数 | 必填 | 类型 | 默认值 | 说明 |
|------|------|------|--------|------|
| `model` | 是 | string | — | 模型类型：`wav2lip` / `musetalk` |
| `avatar_id` | 是 | string | — | Avatar 唯一标识符 |
| `video_file` | 否 | file | — | 上传的视频文件（multipart），保存到 `./data/tmp/` |
| `video_path` | 条件 | string | — | 视频文件本地路径（未上传 `video_file` 时必填） |
| `img_size` | 否 | int | 256 | 输出图像尺寸 |
| `nosmooth` | 否 | bool | false | 禁用人脸检测平滑 |
| `bbox_shift` | 否 | int | 0 | 人脸框偏移（musetalk） |
| `extra_margin` | 否 | int | 10 | 人脸裁剪额外边距（musetalk） |
| `pads` | 否 | string | "0 10 0 0" | 填充：上 下 左 右（空格分隔） |
| `parsing_mode` | 否 | string | "jaw" | 人脸解析模式（musetalk） |
| `version` | 否 | string | "v15" | MuseTalk 版本：`v1` / `v15` |
| `face_det_batch_size` | 否 | int | 16 | 人脸检测批大小（wav2lip） |
| `task_id` | 否 | string | 自动UUID | 自定义任务ID |
| `notifyurl` | 否 | string | — | 回调URL，任务状态变更时POST通知 |

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "task_id": "uuid-string"
  }
}
```

---

## 2. 查询任务状态

```
GET /api/avatar/task/{task_id}
```

**路径参数**: `task_id` — 创建任务时返回的ID

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "task_id": "uuid-string",
    "model_type": "musetalk",
    "avatar_id": "avatar1",
    "status": "running",
    "progress": 45,
    "error_msg": "",
    "notify_url": "",
    "start_time": 1713916800.0,
    "end_time": null,
    "duration": 30.5
  }
}
```

`status` 取值：`pending` → `running` → `completed` / `failed`

`progress`：0-100 整数

---

## 3. 列出所有任务

```
GET /api/avatar/tasks
```

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "tasks": [
      { "task_id": "...", "status": "completed", "progress": 100 },
      { "task_id": "...", "status": "running", "progress": 30 }
    ]
  }
}
```

按 `start_time` 降序排列。

---

## 4. 删除任务

```
DELETE /api/avatar/task/{task_id}
```

**路径参数**: `task_id`

**响应**:

```json
// 成功
{ "code": 0, "msg": "ok", "data": { "msg": "Task deleted" } }

// 失败 - 任务不存在
{ "code": -1, "msg": "Task not found" }

// 失败 - 任务不可删除
{ "code": -1, "msg": "Task is in running state, cannot delete" }
```

仅 `pending` 状态的任务可删除。

---

## 模型与参数对应关系

| model | 专有参数 | 生成模块 |
|-------|----------|----------|
| `wav2lip` | `face_det_batch_size`, `pads`, `nosmooth`, `img_size` | `avatars/wav2lip/genavatar.py` |
| `musetalk` | `bbox_shift`, `extra_margin`, `parsing_mode`, `version` | `avatars/musetalk/genavatar.py` |

## 生成输出

Avatar 数据保存在 `<data_path>/<avatar_id>/` 目录下，包含：`full_imgs/`、`face_imgs/`、`coords.pkl` 及模型特定文件。生成完成后可直接通过 `--avatar_id <avatar_id>` 启动服务。

---

## 5. 动作编排（形象创建操作台）

配合操作台页面 `/avatar-studio.html` 使用。创建任务时可通过 `name` / `actions` 参数自动生成形象动作配置。

### actions 参数格式（JSON 字符串）

```json
{
  "idle":  { "start": 0,  "end": 3  },
  "reply": { "start": 4,  "end": 12 },
  "think": { "start": 13, "end": 16 }
}
```

- 动作名固定：`idle`=待机(audiotype 1，仅提取帧序列，TTS 静默自动循环)、`reply`=讲解(audiotype 2，帧+16k mono wav)、`think`=思考(audiotype 3，帧+16k mono wav)；
- 时间单位为秒，支持小数；`end` 必须大于 `start` 且不超过视频时长；
- 未标记的动作不生成，运行时回退全局配置（`--customvideo_config` 或 `/offer` 的 `custom_config`）。

生成产物：
- `data/avatars/{avatar_id}/actions/{idle|reply|think}/`：动作帧序列 `00000001.png`（25fps）+ 讲解/思考的 `reply.wav` / `think.wav`（16k mono）；
- `data/avatars/{avatar_id}/actions.json`：运行时 per-avatar 自动加载的动作配置（结构同 `custom_config`）；
- `data/avatars/{avatar_id}/avatar.json`：形象元信息（`avatar_id/name/model/created_at/actions`）。

### 5.1 更新已有形象的动作片段

```
POST /api/avatar/actions
```

**Content-Type**: `application/json` 或 `multipart/form-data`（上传视频时）

| 参数 | 必填 | 类型 | 说明 |
|------|------|------|------|
| `avatar_id` | 是 | string | 形象 ID |
| `video_file` | 否 | file | 上传的视频（存 `./data/tmp/`） |
| `video_path` | 条件 | string | 视频本地路径（未上传 `video_file` 时必填） |
| `actions` | 是 | string | 动作片段标记 JSON 字符串（格式同上） |

同步执行，返回更新后的动作配置项列表。注意：该接口会**按本次传入的 actions 覆盖** `actions.json`（未标记的动作将被移除）。

### 5.2 列出已创建形象

```
GET /api/avatar/list
```

扫描 `data/avatars/*/avatar.json` 返回形象列表（按 `created_at` 降序）。无 `avatar.json` 但存在 `full_imgs/` 的目录（命令行创建的形象）也会列出。

**响应**:

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "avatars": [
      { "avatar_id": "avatar1", "name": "张三", "model": "wav2lip", "created_at": 1713916800.0, "actions": {} }
    ]
  }
}
```

### 5.3 获取形象详情

```
GET /api/avatar/{avatar_id}
```

返回单个形象详情：`avatar.json` 元信息 + `actions.json` 动作配置 + `has_base`（是否已生成基底）。形象不存在返回 404。
