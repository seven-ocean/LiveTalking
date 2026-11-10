# coding: utf-8
"""离线音频驱动口型：形象图片/视频 + 一段 wav -> 口型对齐的视频（Wav2Lip 256）。

与 LivePortrait 的区别：LivePortrait 用固定动作模板，口型跟音频无关；
这里用 Wav2Lip，逐帧用音频 mel 预测嘴部，真正做到「声音驱动口型」。
传 --video 时会在「待机视频」上换嘴，保留转头/眨眼等微动，画面不会像照片。

用法（在 zl-talking 目录下执行）：
    python offline_lipsync.py --image assets/streamer3d.jpg --audio speech.wav --out out.mp4
    python offline_lipsync.py --video idle.mp4            --audio speech.wav --out out.mp4
"""

from __future__ import annotations

import argparse
import math
import subprocess
from pathlib import Path

import cv2
import imageio.v2 as imageio
import imageio_ffmpeg
import librosa
import numpy as np
import torch

from avatars.wav2lip import face_detection
from avatars.wav2lip.audio import melspectrogram
from avatars.wav2lip.models import Wav2Lip

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FPS = 25
MEL_STEP = 16


def load_model(path: Path) -> Wav2Lip:
    model = Wav2Lip()
    state = torch.load(str(path), map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
    return model.to(DEVICE).eval()


class BoxDetector:
    def __init__(self, pads=(0, 10, 0, 0)):
        self.det = face_detection.FaceAlignment(
            face_detection.LandmarksType._2D, flip_input=False, device=DEVICE
        )
        self.pads = pads
        self.last = None

    def __call__(self, img_bgr: np.ndarray) -> tuple[int, int, int, int]:
        """返回 (top, bottom, left, right)，检测失败时沿用上一帧的框。"""
        h, w = img_bgr.shape[:2]
        boxes = self.det.get_detections_for_batch(np.array([img_bgr]))
        if not boxes or boxes[0] is None:
            if self.last is not None:
                return self.last
            side = int(min(w, h) * 0.62)
            cx, cy = w // 2, int(h * 0.32)
            return max(0, cy - side // 2), min(h, cy + side // 2), max(0, cx - side // 2), min(w, cx + side // 2)
        x1, y1, x2, y2 = boxes[0]
        pady1, pady2, padx1, padx2 = self.pads
        box = (max(0, y1 - pady1), min(h, y2 + pady2), max(0, x1 - padx1), min(w, x2 + padx2))
        self.last = box
        return box


def mel_chunks(mel: np.ndarray, n_frames: int) -> np.ndarray:
    idx_mult = 80.0 / FPS
    out = []
    for i in range(n_frames):
        s = int(i * idx_mult)
        s = min(s, max(0, mel.shape[1] - MEL_STEP))
        out.append(mel[:, s:s + MEL_STEP].reshape(80, MEL_STEP, 1))
    return np.ascontiguousarray(np.transpose(np.asarray(out), (0, 3, 1, 2)))


def read_frames(path: Path) -> list[bytes]:
    """源画面按 JPEG 存内存，600 帧也才几十 MB，避免整段 BGR 撑爆内存。"""
    out = []
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
        img = cv2.imread(str(path))
        return [cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 96])[1].tobytes()]
    for f in imageio.get_reader(str(path)):
        out.append(cv2.imencode(".jpg", cv2.cvtColor(f, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])[1].tobytes())
    return out


def decode(buf: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--image", type=Path)
    src.add_argument("--video", type=Path)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weight", type=Path, default=Path("models/wav2lip/wav2lip256.pth"))
    ap.add_argument("--img-size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=48)
    args = ap.parse_args()

    src = read_frames(args.video or args.image)
    if not src:
        raise SystemExit(f"读取失败: {args.video or args.image}")
    h, w = decode(src[0]).shape[:2]

    wav, _ = librosa.load(str(args.audio), sr=16000, mono=True)
    duration = len(wav) / 16000.0
    n_frames = max(1, math.ceil(duration * FPS))
    mel = melspectrogram(wav)
    print(f"音频 {duration:.2f}s -> {n_frames} 帧 @ {FPS}fps，源画面 {len(src)} 帧 {w}x{h}")

    detector = BoxDetector()
    faces, boxes = [], []
    for i in range(n_frames):
        frame = decode(src[i % len(src)])  # 待机视频比音频短时循环
        box = detector(frame)
        y1, y2, x1, x2 = box
        boxes.append(box)
        faces.append(cv2.resize(frame[y1:y2, x1:x2], (args.img_size, args.img_size)))
    print(f"人脸框首帧 ({boxes[0][2]},{boxes[0][0]})-({boxes[0][3]},{boxes[0][1]})")

    audio_in = torch.FloatTensor(mel_chunks(mel, n_frames)).to(DEVICE)
    masked = [f.copy() for f in faces]
    for m in masked:
        m[args.img_size // 2:] = 0
    pair = np.concatenate((np.stack(masked), np.stack(faces)), axis=3) / 255.0
    pair = torch.FloatTensor(np.transpose(pair, (0, 3, 1, 2))).to(DEVICE)

    model = load_model(args.weight)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    silent = args.out.with_name(args.out.stem + "_silent.mp4")
    writer = imageio.get_writer(silent, fps=FPS, macro_block_size=2)
    written = 0
    try:
        with torch.no_grad():
            for s in range(0, n_frames, args.batch):
                e = min(s + args.batch, n_frames)
                pred = model(audio_in[s:e], pair[s:e]).cpu().numpy().transpose(0, 2, 3, 1)
                pred = np.clip(pred * 255.0, 0, 255).astype(np.uint8)
                for k, p in zip(range(s, e), pred):
                    frame = decode(src[k % len(src)]).copy()
                    y1, y2, x1, x2 = boxes[k]
                    frame[y1:y2, x1:x2] = cv2.resize(p, (x2 - x1, y2 - y1))
                    writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    written += 1
                print(f"  {e}/{n_frames}", end="\r")
    finally:
        writer.close()
    print(f"\n已写入 {written} 帧")

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(silent), "-i", str(args.audio),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-c:a", "aac", "-b:a", "128k",
         "-shortest", str(args.out)],
        check=True,
    )
    silent.unlink(missing_ok=True)
    print("成片:", args.out, args.out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
