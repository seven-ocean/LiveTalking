#!/usr/bin/env python3
"""按官方 MuseTalk v1.5 的 prepare_material 重新生成形象素材（不覆盖原形象）。

与官方 `/data/MuseTalk/scripts/realtime_inference.py` 的差异说明：
- 输入用现成 full_imgs（原始视频已不在），等价于官方 video2imgs；
- bbox 用官方 mmpose(DWPose) 68 点面部关键点 + S3FD 检测兜底（与官方 preprocessing.py 一致），
  替换了我们原来 genavatar 里的 face_alignment(FAN)；
- latents/mask 与官方同参数：v1.5 extra_margin=10、FaceParsing(cheek=90)、mode='jaw'；
- 不做官方那种 frame_list[::-1] 翻倍存储（运行时 mirror_index 与翻倍索引等价，避免资产翻倍）。

用法（在 zl-talking 目录）：
  ../runtime/venv/talking/bin/python tests/rebuild_avatar_official.py \
      --src musetalk_avater_test_1 --dst musetalk_avater_test_1_v15o [--video /path/to/source.mp4]
"""
import argparse
import json
import os
import pickle
import shutil
import sys

import cv2
import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

# mmpose/mmcv 在可选环境里（mmpose 是源码 .pth，这里手动补路径）；
# tomli 等缺包由 conda 底座补齐。追加在 talking venv 之后，不覆盖现有依赖。
# 注意：talking venv 里装了同名 pip 包 face_detection（无 detection 子模块），
# 而 api.py 内部按顶层包名 __import__，必须把 vendored 目录放到 sys.path 最前面。
sys.path.insert(0, os.path.join(ROOT, 'avatars', 'musetalk', 'utils'))
for _p in (
    os.path.abspath(os.path.join(ROOT, '..', 'runtime', 'venv', 'optional',
                                 'lib', 'python3.10', 'site-packages')),
    os.path.abspath(os.path.join(ROOT, '..', 'runtime', 'optional', 'mmpose')),
    '/opt/conda/lib/python3.10/site-packages',
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)

from mmpose.apis import inference_topdown, init_model  # noqa: E402
from mmpose.structures import merge_data_samples  # noqa: E402

from avatars.musetalk.models.vae import VAE  # noqa: E402
from avatars.musetalk.utils.blending import get_image_prepare_material  # noqa: E402
from avatars.musetalk.utils.face_detection import FaceAlignment, LandmarksType  # noqa: E402
from avatars.musetalk.utils.face_parsing import FaceParsing  # noqa: E402

DWPOSE_CFG = 'avatars/musetalk/utils/dwpose/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
DWPOSE_CKPT = 'models/musetalk/dwpose/dw-ll_ucoco_384.pth'
VAE_DIR = 'models/musetalk/vae'
COORD_PLACEHOLDER = (0.0, 0.0, 0.0, 0.0)


def read_imgs(img_list):
    return [cv2.imread(p) for p in img_list]


def official_bboxes(frames, bbox_shift=0):
    """官方 preprocessing.get_landmark_and_bbox：mmpose 68 点 + S3FD 兜底。"""
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    pose = init_model(DWPOSE_CFG, DWPOSE_CKPT, device=device)
    fa = FaceAlignment(LandmarksType._2D, flip_input=False, device=device)

    coords = []
    for frame in tqdm(frames, desc='landmarks'):
        results = inference_topdown(pose, np.asarray(frame))
        results = merge_data_samples(results)
        keypoints = results.pred_instances.keypoints
        face_land_mark = keypoints[0][23:91].astype(np.int32)

        bbox = fa.get_detections_for_batch(np.asarray([frame]))
        f = bbox[0]
        if f is None:
            coords.append(list(COORD_PLACEHOLDER))
            continue
        half_face_coord = face_land_mark[29]
        if bbox_shift:
            half_face_coord[1] = bbox_shift + half_face_coord[1]
        half_face_dist = np.max(face_land_mark[:, 1]) - half_face_coord[1]
        upper_bond = max(0, half_face_coord[1] - half_face_dist)
        f_landmark = (int(np.min(face_land_mark[:, 0])), int(upper_bond),
                      int(np.max(face_land_mark[:, 0])), int(np.max(face_land_mark[:, 1])))
        x1, y1, x2, y2 = f_landmark
        coords.append(list(f_landmark) if (y2 - y1 > 0 and x2 - x1 > 0 and x1 >= 0) else list(f))
    return coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='musetalk_avater_test_1')
    ap.add_argument('--dst', default='musetalk_avater_test_1_v15o')
    ap.add_argument('--extra-margin', type=int, default=10)
    ap.add_argument('--bbox-shift', type=int, default=0)
    ap.add_argument('--video', default='', help='原始视频路径；提供后直接从视频抽帧（官方 video2imgs）')
    args = ap.parse_args()

    src = os.path.join('data', 'avatars', args.src)
    dst = os.path.join('data', 'avatars', args.dst)
    if os.path.exists(dst):
        shutil.rmtree(dst)
    for sub in ('full_imgs', 'mask'):
        os.makedirs(os.path.join(dst, sub), exist_ok=True)

    if args.video:
        cap = cv2.VideoCapture(args.video)
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        print(f'输入帧 {len(frames)} 张（来自视频 {args.video}，官方 video2imgs）')
    else:
        img_paths = sorted(
            [p for p in (os.path.join(src, 'full_imgs', n)
                         for n in os.listdir(os.path.join(src, 'full_imgs')))
             if p.lower().endswith(('.png', '.jpg', '.jpeg'))],
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        frames = read_imgs(img_paths)
        print(f'输入帧 {len(frames)} 张（来自 {src}/full_imgs）')

    coords = official_bboxes(frames, args.bbox_shift)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(model_path=VAE_DIR)
    vae.vae = vae.vae.half().to(device)
    fp = FaceParsing(left_cheek_width=90, right_cheek_width=90)

    latents, mask_coords = [], []
    for idx, (frame, bbox) in enumerate(tqdm(list(zip(frames, coords)), desc='latents+mask')):
        if tuple(bbox) == COORD_PLACEHOLDER:
            continue
        x1, y1, x2, y2 = bbox
        y2 = min(y2 + args.extra_margin, frame.shape[0])
        coords[idx] = [x1, y1, x2, y2]
        crop = frame[y1:y2, x1:x2]
        resized = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        latents.append(vae.get_latents_for_unet(resized))

        mask, crop_box = get_image_prepare_material(frame, [x1, y1, x2, y2], fp=fp, mode='jaw')
        cv2.imwrite(os.path.join(dst, 'mask', f'{idx:08d}.png'), mask)
        mask_coords.append(crop_box)

    for i, frame in enumerate(frames):
        cv2.imwrite(os.path.join(dst, 'full_imgs', f'{i:08d}.png'), frame)

    with open(os.path.join(dst, 'coords.pkl'), 'wb') as fh:
        pickle.dump(coords, fh)
    with open(os.path.join(dst, 'mask_coords.pkl'), 'wb') as fh:
        pickle.dump(mask_coords, fh)
    torch.save(latents, os.path.join(dst, 'latents.pt'))

    old_meta = {}
    meta_path = os.path.join(src, 'avatar.json')
    if os.path.isfile(meta_path):
        old_meta = json.load(open(meta_path, encoding='utf-8'))
    name = old_meta.get('name') or args.dst
    json.dump({'avatar_id': args.dst, 'name': f'{name}（官方预处理）',
               'model': 'musetalk', 'base_source': 'full'},
              open(os.path.join(dst, 'avatar.json'), 'w', encoding='utf-8'),
              ensure_ascii=False)
    json.dump({'avatar_id': args.dst, 'video_path': args.video or f'{src}/full_imgs',
               'bbox_shift': args.bbox_shift, 'version': 'v15', 'extra_margin': args.extra_margin},
              open(os.path.join(dst, 'avator_info.json'), 'w', encoding='utf-8'))

    # 与旧资产做 bbox 对比，便于判断“脸变形是否来自裁剪框差异”
    try:
        old_coords = pickle.load(open(os.path.join(src, 'coords.pkl'), 'rb'))
        n = min(len(old_coords), len(coords))
        d = np.abs(np.asarray(old_coords[:n], dtype=np.float32) -
                   np.asarray(coords[:n], dtype=np.float32))
        print(f'bbox 对比：{n} 帧，平均 |dx1,dy1,dx2,dy2| = {d.mean(axis=0).round(1).tolist()}，'
              f'最大 = {d.max(axis=0).tolist()}')
        w_old = np.asarray(old_coords[:n])[:, 2] - np.asarray(old_coords[:n])[:, 0]
        w_new = np.asarray(coords[:n])[:, 2] - np.asarray(coords[:n])[:, 0]
        h_old = np.asarray(old_coords[:n])[:, 3] - np.asarray(old_coords[:n])[:, 1]
        h_new = np.asarray(coords[:n])[:, 3] - np.asarray(coords[:n])[:, 1]
        print(f'裁剪框尺寸：旧 {w_old.mean():.1f}x{h_old.mean():.1f} → '
              f'新 {w_new.mean():.1f}x{h_new.mean():.1f}')
    except Exception as exc:
        print('bbox 对比失败:', exc)

    print(f'新形象已生成：{dst}（{len(latents)} 帧 latents）')
    return 0


if __name__ == '__main__':
    sys.exit(main())
