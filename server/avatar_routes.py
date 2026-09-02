import os
import json
import uuid
from aiohttp import web
from server.task_manager import task_manager
from utils.logger import logger

def json_ok(data=None):
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(
        content_type="application/json",
        text=json.dumps(body),
    )

def json_error(msg: str, code: int = -1):
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )

async def create_avatar_task(request):
    """
    POST /api/avatar/task
    Parameters: model, avatar_id, video_file (upload), video_path (local), ...
    """
    try:
        if request.content_type == 'multipart/form-data':
            reader = await request.multipart()
            params = {}
            video_path = None

            while True:
                part = await reader.next()
                if part is None:
                    break

                if part.name == 'video_file':
                    filename = part.filename
                    temp_dir = os.path.abspath('./data/tmp')
                    os.makedirs(temp_dir, exist_ok=True)
                    video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
                    with open(video_path, 'wb') as f:
                        while True:
                            chunk = await part.read_chunk()
                            if not chunk:
                                break
                            f.write(chunk)
                else:
                    value = await part.text()
                    params[part.name] = value

            if video_path:
                params['video_path'] = video_path
        else:
            params = await request.json()

        model_type = params.get('model')
        avatar_id = params.get('avatar_id')

        if not model_type or not avatar_id:
            return json_error("model and avatar_id are required")

        if avatar_id.endswith('_reply'):
            return json_error("avatar_id 不能以 _reply 结尾（该后缀为讲解子形象保留）")

        if 'video_path' not in params:
            return json_error("video_file or video_path is required")

        data_path = './data/avatars'

        video_path = params['video_path']
        if not os.path.isabs(video_path):
            video_path = os.path.join(data_path, video_path)

        save_path = data_path

        task_params = {
            "video_path": video_path,
            "save_path": save_path,
            "img_size": int(params.get('img_size', 256)),
            "nosmooth": params.get('nosmooth', 'false').lower() == 'true' if isinstance(params.get('nosmooth'), str) else params.get('nosmooth', False),
            "bbox_shift": int(params.get('bbox_shift', 0)),
            "extra_margin": int(params.get('extra_margin', 10)),
            "parsing_mode": params.get('parsing_mode', 'jaw'),
            "version": params.get('version', 'v15'),
            "face_det_batch_size": int(params.get('face_det_batch_size', 4))
        }

        # 编辑模式：跳过基座重新生成，只重生成动作帧 + 讲解形象
        if params.get('skip_base') is not None:
            sv = params.get('skip_base')
            task_params['skip_base'] = sv.lower() in ('true', '1', 'yes') if isinstance(sv, str) else bool(sv)

        # 动作编排：形象命名 + 动作片段标记（{"idle": {"start": s, "end": e}, ...}）
        if params.get('name'):
            task_params['name'] = params['name']
        if params.get('actions'):
            task_params['actions'] = params['actions']

        pads_str = params.get('pads', "0 10 0 0")
        if isinstance(pads_str, str):
            task_params['pads'] = [int(x) for x in pads_str.split()]
        else:
            task_params['pads'] = pads_str

        task_id_input = params.get('task_id')
        notify_url = params.get('notifyurl')

        task_id = task_manager.add_task(model_type, avatar_id, task_params, task_id=task_id_input, notify_url=notify_url)
        return json_ok(data={"task_id": task_id})

    except Exception as e:
        logger.exception("create_avatar_task error:")
        return json_error(str(e))

async def get_avatar_task_status(request):
    """
    GET /api/avatar/task/{task_id}
    """
    task_id = request.match_info.get('task_id')
    task = task_manager.get_task(task_id)
    if not task:
        return json_error("Task not found", code=404)

    return json_ok(data=task.to_dict())

async def list_avatar_tasks(request):
    """
    GET /api/avatar/tasks
    """
    tasks = task_manager.list_tasks()
    return json_ok(data={"tasks": tasks})

async def delete_avatar_task(request):
    """
    DELETE /api/avatar/task/{task_id}
    """
    task_id = request.match_info.get('task_id')
    success, msg = task_manager.delete_task(task_id)
    if not success:
        return json_error(msg)
    return json_ok(data={"msg": msg})

def _parse_avatar_id(avatar_id):
    """校验 avatar_id 合法性，防止路径穿越。"""
    if not avatar_id or "/" in avatar_id or "\\" in avatar_id or ".." in avatar_id:
        raise ValueError(f"非法 avatar_id: {avatar_id}")
    return avatar_id

async def _parse_request_parts(request):
    """解析 multipart（含视频上传）或 JSON 请求体，返回 (params, video_path)。"""
    if request.content_type == 'multipart/form-data':
        reader = await request.multipart()
        params = {}
        video_path = None
        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name == 'video_file':
                filename = part.filename
                temp_dir = os.path.abspath('./data/tmp')
                os.makedirs(temp_dir, exist_ok=True)
                video_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{filename}")
                with open(video_path, 'wb') as f:
                    while True:
                        chunk = await part.read_chunk()
                        if not chunk:
                            break
                        f.write(chunk)
            else:
                value = await part.text()
                params[part.name] = value
        if video_path:
            params['video_path'] = video_path
    else:
        params = await request.json()
        video_path = params.get('video_path')
    return params, video_path

async def update_avatar_actions(request):
    """
    POST /api/avatar/actions
    形象已存在时，单独生成/更新动作片段（编辑场景）。
    Body: {avatar_id, video_path(或上传 video_file), actions: {"idle": {"start": s, "end": e}, ...}}
    """
    try:
        params, video_path = await _parse_request_parts(request)
        avatar_id = _parse_avatar_id(params.get('avatar_id'))
        if not avatar_id:
            return json_error("avatar_id is required")
        if 'actions' not in params:
            return json_error("actions is required")

        actions = params['actions']
        if isinstance(actions, str):
            actions = json.loads(actions)

        video_path = params.get('video_path')
        if not video_path:
            return json_error("video_file or video_path is required")
        if not os.path.isabs(video_path):
            video_path = os.path.join('./data/avatars', video_path)

        from server.action_builder import build_avatar_actions
        items = build_avatar_actions(avatar_id, video_path, actions, save_path='./data/avatars')

        # 同步更新形象元信息中的 actions 摘要（avatar.json 存在时）
        meta_file = os.path.join('./data/avatars', avatar_id, 'avatar.json')
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                meta['actions'] = actions
                with open(meta_file, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=4)
            except Exception as e:
                logger.warning(f"[avatar] 更新 {meta_file} 失败: {e}")

        return json_ok(data={"avatar_id": avatar_id, "actions": items})
    except Exception as e:
        logger.exception("update_avatar_actions error:")
        return json_error(str(e))

async def list_avatars(request):
    """
    GET /api/avatar/list
    扫描 data/avatars/*/avatar.json 返回已创建形象列表（按 created_at 降序）。
    无 avatar.json 但存在 full_imgs/ 的目录也列出（命令行创建的形象，容错）。
    """
    try:
        data_root = os.path.abspath('./data/avatars')
        avatars = []
        if os.path.isdir(data_root):
            for name in sorted(os.listdir(data_root)):
                avatar_dir = os.path.join(data_root, name)
                if not os.path.isdir(avatar_dir) or name.startswith('.'):
                    continue
                if name.endswith('_reply'):
                    continue  # 讲解子形象隐藏，父形象以「讲解✓」徽标呈现
                meta_file = os.path.join(avatar_dir, 'avatar.json')
                actions_file = os.path.join(avatar_dir, 'actions.json')
                if os.path.exists(meta_file):
                    try:
                        with open(meta_file, 'r', encoding='utf-8') as f:
                            meta = json.load(f)
                        meta['actions'] = meta.get('actions', {})
                        # has_reply_avatar：有讲解动作即标记（单目录方案下无 _reply 子形象，靠 actions 判定；旧数据兼容 reply_avatar_id）
                        meta['has_reply_avatar'] = bool(meta.get('reply_avatar_id')) or 'reply' in (meta.get('actions') or {})
                        avatars.append(meta)
                        continue
                    except Exception as e:
                        logger.warning(f"[avatar] 读取 {meta_file} 失败: {e}")
                # 容错：命令行创建的形象没有 avatar.json
                if os.path.isdir(os.path.join(avatar_dir, 'full_imgs')):
                    created_at = 0.0
                    try:
                        created_at = os.path.getmtime(avatar_dir)
                    except OSError:
                        pass
                    actions = {}
                    if os.path.exists(actions_file):
                        try:
                            with open(actions_file, 'r', encoding='utf-8') as f:
                                actions = json.load(f)
                        except Exception:
                            actions = {}
                    avatars.append({
                        "avatar_id": name,
                        "name": name,
                        "model": "",
                        "created_at": created_at,
                        "actions": actions,
                        "has_reply_avatar": any(item.get('audiotype') == 2 for item in actions) or os.path.isdir(
                            os.path.join(data_root, f"{name}_reply")),
                    })
        avatars.sort(key=lambda x: x.get('created_at', 0), reverse=True)
        return json_ok(data={"avatars": avatars})
    except Exception as e:
        logger.exception("list_avatars error:")
        return json_error(str(e))

async def get_avatar_detail(request):
    """
    GET /api/avatar/{avatar_id}
    返回单个形象详情（avatar.json + actions.json）。
    """
    try:
        avatar_id = _parse_avatar_id(request.match_info.get('avatar_id'))
        avatar_dir = os.path.join(os.path.abspath('./data/avatars'), avatar_id)
        if not os.path.isdir(avatar_dir):
            return json_error("Avatar not found", code=404)

        meta = {"avatar_id": avatar_id, "name": avatar_id, "model": "", "created_at": None, "actions": {}}
        meta_file = os.path.join(avatar_dir, 'avatar.json')
        if os.path.exists(meta_file):
            try:
                with open(meta_file, 'r', encoding='utf-8') as f:
                    meta.update(json.load(f))
            except Exception as e:
                logger.warning(f"[avatar] 读取 {meta_file} 失败: {e}")

        actions = []
        actions_file = os.path.join(avatar_dir, 'actions.json')
        if os.path.exists(actions_file):
            try:
                with open(actions_file, 'r', encoding='utf-8') as f:
                    actions = json.load(f)
            except Exception as e:
                logger.warning(f"[avatar] 读取 {actions_file} 失败: {e}")
        meta['actions'] = actions
        meta['has_base'] = os.path.isdir(os.path.join(avatar_dir, 'full_imgs'))
        # actions 为 actions.json 列表（含 audiotype），讲解=2；旧数据兼容 reply_avatar_id / _reply 目录
        meta['has_reply_avatar'] = bool(meta.get('reply_avatar_id')) or any(
            item.get('audiotype') == 2 for item in actions) or os.path.isdir(
            os.path.join(os.path.dirname(avatar_dir), f"{avatar_id}_reply"))
        return json_ok(data=meta)
    except Exception as e:
        logger.exception("get_avatar_detail error:")
        return json_error(str(e))

def setup_avatar_routes(app):
    app.router.add_post("/api/avatar/task", create_avatar_task)
    app.router.add_get("/api/avatar/task/{task_id}", get_avatar_task_status)
    app.router.add_delete("/api/avatar/task/{task_id}", delete_avatar_task)
    app.router.add_get("/api/avatar/tasks", list_avatar_tasks)
    app.router.add_post("/api/avatar/actions", update_avatar_actions)
    app.router.add_get("/api/avatar/list", list_avatars)
    app.router.add_get("/api/avatar/{avatar_id}", get_avatar_detail)
