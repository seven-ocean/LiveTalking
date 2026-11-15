#!/usr/bin/env python3
"""MuseTalk 嘴部抖动离线探针。

用固定音频走一遍现有的 MuseTalk 推理链（与线上同一套代码路径），
在嘴部 mask/bbox 区域统计「相邻帧像素差」和「二阶差分（高频抖动）」，
并按音频能量分档输出；同时记录 latent 触边（clamp）比例。

配合环境变量做 A/B：
  MUSETALK_MOUTH_SMOOTH=0.4      # 帧间 latent EMA
  MUSETALK_LATENT_CLAMP=0        # 关闭 ±3.5 硬裁剪
  MUSETALK_FEAT_WIN=2,2          # 上游 MuseTalk 对称特征窗
  MUSETALK_VAE_FP32=1            # VAE fp32 解码
  MUSETALK_MASK_BLUR=3           # mask 羽化
  MUSETALK_DEBUG_LATENT=1        # 记录触边比例

用法（在 zl-talking 目录下）：
  ../runtime/venv/talking/bin/python tests/mouth_jitter_probe.py \
      --avatar musetalk_avater_test_1 --wav ../zl-SoVITS/ref_audio/Keira.wav \
      --batches 12 --json /tmp/jitter_baseline.json
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)  # 模型/形象路径都是相对 zl-talking 目录

from avatars.audio_features.whisper import WhisperASR, _LazyFeat  # noqa: E402
from avatars.musetalk_avatar import MuseReal, load_avatar, load_model  # noqa: E402
from utils.image import mirror_index  # noqa: E402


def load_wav_16k(path):
    import resampy
    import soundfile as sf

    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        data = resampy.resample(x=data, sr_orig=sr, sr_new=16000).astype(np.float32)
    return data


def main():
    parser = argparse.ArgumentParser(description='MuseTalk 嘴部抖动探针')
    parser.add_argument('--avatar', default='musetalk_avater_test_1')
    parser.add_argument('--wav', required=True)
    parser.add_argument('--batches', type=int, default=12, help='跑多少个 640ms 批次')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--fps', type=int, default=25)
    parser.add_argument('--left', type=int, default=10)
    parser.add_argument('--right', type=int, default=10)
    parser.add_argument('--json', default='')
    args = parser.parse_args()

    opt = argparse.Namespace(
        fps=args.fps, batch_size=args.batch_size, l=args.left, r=args.right,
        avatar_id=args.avatar, model='musetalk', modelres=192,
    )

    print(f'[probe] avatar={args.avatar} wav={args.wav} batches={args.batches}')
    print('[probe] knobs: ' + ', '.join(
        f'{k}={os.getenv(k, "(default)")}'
        for k in ('MUSETALK_MOUTH_SMOOTH', 'MUSETALK_LATENT_CLAMP',
                  'MUSETALK_FEAT_WIN', 'MUSETALK_VAE_FP32', 'MUSETALK_MASK_BLUR')
    ))

    t0 = time.time()
    model = load_model()
    avatar = load_avatar(args.avatar)
    print(f'[probe] 模型+形象加载完成 {time.time() - t0:.1f}s')

    probe = MuseReal.__new__(MuseReal)  # 复用生产方法，跳过 __init__ 里的 ASR/warm_up
    (probe.vae, probe.unet, probe.pe,
     probe.timesteps, probe.audio_processor) = model
    (probe.frame_list_cycle, probe.mask_list_cycle, probe.coord_list_cycle,
     probe.mask_coords_list_cycle, probe.input_latent_list_cycle) = avatar
    probe.batch_size = args.batch_size
    probe.opt = opt

    asr = WhisperASR(opt, None, model[4])

    audio = load_wav_16k(args.wav)
    chunk = 320  # 16k * 20ms
    chunks = [audio[i:i + chunk] for i in range(0, len(audio) - chunk, chunk)]
    print(f'[probe] 音频 {len(audio) / 16000:.2f}s，{len(chunks)} 个 20ms chunk')

    frames_window = []
    prev_probe_frame = None
    rows = []
    idx = 0
    start = time.time()

    with torch.no_grad():
        for b in range(args.batches):
            batch_chunks = chunks[b * args.batch_size * 2:(b + 1) * args.batch_size * 2]
            if len(batch_chunks) < args.batch_size * 2:
                batch_chunks += [np.zeros(chunk, dtype=np.float32)] * (
                    args.batch_size * 2 - len(batch_chunks))
            rms = float(np.sqrt(np.mean(np.concatenate(batch_chunks) ** 2)) + 1e-9)

            frames_window += batch_chunks
            if len(frames_window) <= args.left + args.right:
                continue
            inputs = np.concatenate(frames_window)
            feats = asr.resolve_feat_batch(_LazyFeat(inputs))
            frames_window = frames_window[-(args.left + args.right):]
            if len(feats) != args.batch_size:
                print(f'[probe] batch {b} 特征数量异常: {len(feats)}，跳过')
                continue

            t_infer = time.time()
            pred = probe.inference_batch(idx, feats)
            infer_ms = (time.time() - t_infer) * 1000
            clamp_ratio = getattr(probe, '_last_clamp_ratio', None)

            diffs, sharp = [], []
            for res_frame in pred:
                frame = probe.paste_back_frame(res_frame, mirror_index(
                    len(probe.frame_list_cycle), idx))
                x1, y1, x2, y2 = probe.coord_list_cycle[mirror_index(
                    len(probe.coord_list_cycle), idx)]
                # bbox 会随源视频帧轻微变化，统一缩放到 128x128 再做帧间比较
                mouth = cv2.resize(frame[y1:y2, x1:x2], (128, 128)).astype(np.float32)
                sharp.append(float(cv2.Laplacian(mouth.astype(np.uint8), cv2.CV_64F).var()))
                if prev_probe_frame is not None:
                    diffs.append(float(np.abs(mouth - prev_probe_frame).mean()))
                prev_probe_frame = mouth
                idx += 1
            rows.append({
                'batch': b, 'rms': round(rms, 5),
                'mouth_delta': round(float(np.mean(diffs)) if diffs else 0.0, 4),
                'mouth_sharpness': round(float(np.mean(sharp)) if sharp else 0.0, 1),
                'infer_ms': round(infer_ms, 1),
                'clamp_ratio': clamp_ratio,
            })
            print(f'[probe] batch {b:02d} rms={rms:.4f} mouth_delta='
                  f'{rows[-1]["mouth_delta"]:.3f} infer={infer_ms:.0f}ms '
                  f'clamp={clamp_ratio}')

    if not rows:
        print('[probe] 没有有效批次')
        return 1

    rms_sorted = sorted(r['rms'] for r in rows)
    cut_low = rms_sorted[len(rms_sorted) // 3]
    cut_high = rms_sorted[2 * len(rms_sorted) // 3]
    buckets = {'low': [], 'mid': [], 'high': []}
    for r in rows:
        key = 'low' if r['rms'] <= cut_low else ('high' if r['rms'] > cut_high else 'mid')
        buckets[key].append(r['mouth_delta'])
    summary = {
        'avatar': args.avatar,
        'wav': args.wav,
        'batches': len(rows),
        'overall_mouth_delta': round(float(np.mean([r['mouth_delta'] for r in rows])), 4),
        'buckets': {k: round(float(np.mean(v)), 4) if v else None for k, v in buckets.items()},
        'avg_mouth_sharpness': round(float(np.mean([r['mouth_sharpness'] for r in rows])), 1),
        'avg_infer_ms': round(float(np.mean([r['infer_ms'] for r in rows])), 1),
        'avg_clamp_ratio': (round(float(np.mean([r['clamp_ratio'] for r in rows
                                                 if r['clamp_ratio'] is not None])), 4)
                            if any(r['clamp_ratio'] is not None for r in rows) else None),
        'knobs': {k: os.getenv(k) for k in (
            'MUSETALK_MOUTH_SMOOTH', 'MUSETALK_LATENT_CLAMP', 'MUSETALK_FEAT_WIN',
            'MUSETALK_VAE_FP32', 'MUSETALK_MASK_BLUR')},
        'elapsed_s': round(time.time() - start, 1),
    }
    print('[probe] summary: ' + json.dumps(summary, ensure_ascii=False))
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump({'summary': summary, 'rows': rows}, fh, ensure_ascii=False, indent=2)
        print(f'[probe] 明细已写入 {args.json}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
