import os
# Windows 上 torch.compile 需要 triton，会直接抛错；关键点模型要用到它，这里提前关掉
os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

from os import listdir, path
import math
import numpy as np
import scipy, cv2, os, sys, argparse
import json, subprocess, random, string
import time
from tqdm import tqdm
from glob import glob
import torch
import pickle
from avatars.wav2lip import face_detection
from config import DEBUG_WAV2LIP_FACE_DETECTION


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def osmakedirs(path_list):
    for path in path_list:
        os.makedirs(path) if not os.path.exists(path) else None

def video2imgs(vid_path, save_path, ext = '.png',cut_frame = 10000000):
    cap = cv2.VideoCapture(vid_path)
    count = 0
    while True:
        if count > cut_frame:
            break
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "zl-talking", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (128,128,128), 1)
            cv2.imwrite(f"{save_path}/{count:08d}.png", frame)
            count += 1
        else:
            break

def read_imgs(img_list):
    frames = []
    print('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def get_face_landmarks(frames, batch_size=16):
	"""用 68 点关键点逐帧定位五官，失败返回 None。

	这里让关键点模型自己找人脸：只传 Wav2Lip 的人脸框时，关键点会按错误的
	裁剪比例映射回来，嘴部框会撑满整张脸，反而不可用。整帧自动检测慢一点，
	但坐标是准的。

	Wav2Lip 只在人脸端正且下半脸位置稳定时才贴得准，所以这里同时服务两件事：
	1) 用左右眼估头部倾斜角 roll，构建期转正裁剪、推理期反向贴回；
	2) 用嘴部关键点记录嘴的位置，推理期只把「嘴」这一小块贴回，
	   避免大范围融合把露出的牙齿贴到嘴唇上方。
	"""
	try:
		os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')  # Windows 无 triton，禁用 torch.compile
		import face_alignment
		from face_alignment import LandmarksType
	except Exception as e:
		print(f'  [genavatar] 关键点模型不可用，跳过偏头/嘴部对齐: {e}')
		return None

	fa = face_alignment.FaceAlignment(LandmarksType.TWO_D, device=device, flip_input=False)
	all_marks = []
	for start in range(0, len(frames), batch_size):
		chunk = frames[start:start + batch_size]
		rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in chunk])
		# 批量检测接口按 0~255 的图像做检测，传 0~1 会导致“未检测到人脸”
		rgb = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2)))
		try:
			marks = fa.get_landmarks_from_batch(rgb)
		except Exception as e:
			print(f'  [genavatar] 关键点检测失败，按正脸处理: {e}')
			marks = [None] * len(chunk)
		# 一张图可能检出多张脸（接口此时返回列表），取最大那张；单张脸时直接是 68×2 数组
		picked = []
		for lm in marks:
			if lm is None or len(lm) == 0:
				picked.append(None)
				continue
			if isinstance(lm, np.ndarray):
				picked.append(lm)
				continue
			cands = [np.asarray(m, dtype=np.float32) for m in lm]
			picked.append(max(cands, key=lambda m: (m[:, 0].max() - m[:, 0].min()) * (m[:, 1].max() - m[:, 1].min())))
		all_marks.extend(picked)
	return all_marks


def head_roll_from_landmarks(landmarks):
	"""左右眼中点连线相对水平线的夹角（度），失败按 0 处理。"""
	if landmarks is None:
		return 0.0
	lm = np.asarray(landmarks, dtype=np.float32)
	left = lm[36:42].mean(axis=0)
	right = lm[42:48].mean(axis=0)
	return math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))


def map_mouth_box(landmarks, matrix, img_size):
	"""把嘴部 68 点框（48~67）映射到端正裁剪空间，返回归一化 (x1,y1,x2,y2)。"""
	if landmarks is None:
		return None
	pts = np.asarray(landmarks, dtype=np.float32)[48:68]
	pts = np.hstack([pts, np.ones((len(pts), 1), np.float32)]) @ matrix.T
	return [float(pts[:, 0].min()) / img_size, float(pts[:, 1].min()) / img_size,
			float(pts[:, 0].max()) / img_size, float(pts[:, 1].max()) / img_size]


def build_alignment_matrix(box, roll_deg, img_size):
	"""生成 原帧 -> 端正的 img_size×img_size 人脸裁剪 的仿射矩阵。"""
	x1, y1, x2, y2 = [float(v) for v in box]
	cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
	sx = img_size / max(1.0, x2 - x1)
	sy = img_size / max(1.0, y2 - y1)
	# 先按 x/y 分别缩放再旋转，旋转角要用缩放后的眼线角度，否则眼线不会真的水平
	rad = math.radians(roll_deg)
	theta = -math.atan2(sy * math.sin(rad), sx * math.cos(rad))
	c, s = math.cos(theta), math.sin(theta)
	linear = np.array([[c, -s], [s, c]]) @ np.diag([sx, sy])
	offset = np.array([img_size / 2.0, img_size / 2.0]) - linear @ np.array([cx, cy])
	return np.hstack([linear, offset.reshape(2, 1)]).astype(np.float32)


def generate_avatar(video_path, avatar_id, save_path='./data/avatars', img_size=96, pads=[0, 10, 0, 0], nosmooth=False, face_det_batch_size=16, progress_callback=None, align=True):
    """
    生成avatar的核心逻辑

    Args:
        video_path: 输入视频路径
        avatar_id: Avatar ID
        save_path: 保存根路径
        img_size: 缩放后的图像大小
        pads: 人脸框填充 [top, bottom, left, right]
        nosmooth: 是否禁用平滑（开启 align 时始终使用逐帧精确人脸框）
        face_det_batch_size: 人脸检测批处理大小
        progress_callback: 进度回调函数，接收 0-100 的整数
        align: 是否按关键点做头部倾斜对齐（摇头/偏头形象需要）
    """
    avatar_path = os.path.join(save_path, avatar_id)
    full_imgs_path = os.path.join(avatar_path, "full_imgs")
    face_imgs_path = os.path.join(avatar_path, "face_imgs")
    coords_path = os.path.join(avatar_path, "coords.pkl")
    align_path = os.path.join(avatar_path, "align.pkl")
    mouth_path = os.path.join(avatar_path, "mouth.pkl")

    osmakedirs([avatar_path, full_imgs_path, face_imgs_path])

    if progress_callback: progress_callback(5)

    print(f"正在处理视频: {video_path}")
    video2imgs(video_path, full_imgs_path, ext='png')

    if progress_callback: progress_callback(20)

    input_img_list = sorted(glob(os.path.join(full_imgs_path, '*.[jpJP][pnPN]*[gG]')))
    frames = read_imgs(input_img_list)

    if progress_callback: progress_callback(40)

    print('正在检测人脸...')
    _t_init_start = time.perf_counter()
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
                                            flip_input=False, device=device)
    _t_init_end = time.perf_counter()
    if DEBUG_WAV2LIP_FACE_DETECTION:
        print(f"  [genavatar] FaceAlignment init time: {_t_init_end - _t_init_start:.3f}s")

    batch_size = face_det_batch_size
    predictions = []
    total_batches = (len(frames) + batch_size - 1) // batch_size
    batch_idx = 0

    while 1:
        predictions = []
        try:
            for i in range(0, len(frames), batch_size):
                batch_idx += 1
                _t_batch_start = time.perf_counter()
                batch_frames = np.array(frames[i:i + batch_size])
                if DEBUG_WAV2LIP_FACE_DETECTION:
                    print(f"  [genavatar] batch {batch_idx}/{total_batches}: frames {i}~{min(i + batch_size, len(frames)) - 1} ({batch_frames.shape[0]} frames, {batch_frames.shape[1]}x{batch_frames.shape[2]})")
                detections = detector.get_detections_for_batch(batch_frames)
                predictions.extend(detections)
                _t_batch_end = time.perf_counter()
                if DEBUG_WAV2LIP_FACE_DETECTION:
                    batch_face_count = sum(1 for d in detections if d is not None)
                    print(f"  [genavatar] batch {batch_idx} done: {batch_face_count}/{batch_frames.shape[0]} faces, time={_t_batch_end - _t_batch_start:.3f}s")
                if progress_callback:
                    progress = 40 + int((i + batch_size) / len(frames) * 40)
                    progress_callback(min(progress, 80))
        except RuntimeError:
            if batch_size == 1:
                raise RuntimeError('Image too big to run face detection on GPU.')
            batch_size //= 2
            total_batches = (len(frames) + batch_size - 1) // batch_size
            batch_idx = 0
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = pads
    for rect, image in zip(predictions, frames):
        if rect is None:
            rect = [0, 0, image.shape[1], image.shape[0]]

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if align:
        # 逐帧精确人脸框：5 帧均值在摇头时会滞后，导致贴回的脸/嘴位置偏移
        nosmooth = True
    if not nosmooth:
        boxes = get_smoothened_boxes(boxes, T=5)

    landmarks = None
    if align:
        print('正在估计头部姿态与嘴部位置...')
        landmarks = get_face_landmarks(frames, batch_size=face_det_batch_size)
        if landmarks is None:
            align = False
        elif progress_callback:
            progress_callback(83)

    if progress_callback: progress_callback(85)

    coord_list = []
    align_list = []
    mouth_list = []
    print(f"正在保存人脸图片和坐标...")
    for idx, (rect, frame) in enumerate(zip(boxes, frames)):
        x1, y1, x2, y2 = int(rect[0]), int(rect[1]), int(rect[2]), int(rect[3])
        if align:
            roll = head_roll_from_landmarks(landmarks[idx])
            matrix = build_alignment_matrix((x1, y1, x2, y2), roll, img_size)
            resized_crop_frame = cv2.warpAffine(
                frame, matrix, (img_size, img_size),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            align_list.append(cv2.invertAffineTransform(matrix).astype(np.float32))
            mouth_list.append(map_mouth_box(landmarks[idx], matrix, img_size))
        else:
            face_frame = frame[y1:y2, x1:x2]
            resized_crop_frame = cv2.resize(face_frame, (img_size, img_size))
        cv2.imwrite(f"{face_imgs_path}/{idx:08d}.png", resized_crop_frame)
        coord_list.append((y1, y2, x1, x2))

        if progress_callback:
            progress = 85 + int((idx + 1) / len(boxes) * 15)
            progress_callback(progress)

    print(f"写入数据到坐标文件: {coords_path}")
    with open(coords_path, 'wb') as f:
        pickle.dump(coord_list, f)

    if align:
        print(f"写入头部对齐矩阵: {align_path}")
        with open(align_path, 'wb') as f:
            pickle.dump(align_list, f)
        print(f"写入嘴部位置: {mouth_path}")
        with open(mouth_path, 'wb') as f:
            pickle.dump(mouth_list, f)
    else:
        # 避免旧的 align.pkl / mouth.pkl 与新生成的 face_imgs 不匹配
        for p in (align_path, mouth_path):
            if os.path.exists(p):
                os.remove(p)

    del detector
    if progress_callback: progress_callback(100)
    print("Avatar 生成完成！")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')
    parser.add_argument('--img_size', default=96, type=int)
    parser.add_argument('--avatar_id', default='wav2lip_avatar1', type=str)
    parser.add_argument('--save_path', default='data/avatars', type=str)
    parser.add_argument('--video_path', default='', type=str)
    parser.add_argument('--nosmooth', default=False, action='store_true',
                        help='Prevent smoothing face detections over a short temporal window')
    parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0],
                        help='Padding (top, bottom, left, right). Please adjust to include chin at least')
    parser.add_argument('--face_det_batch_size', type=int,
                        help='Batch size for face detection', default=16)
    args = parser.parse_args()

    generate_avatar(
        video_path=args.video_path,
        avatar_id=args.avatar_id,
        save_path=args.save_path,
        img_size=args.img_size,
        pads=args.pads,
        nosmooth=args.nosmooth,
        face_det_batch_size=args.face_det_batch_size
    )
