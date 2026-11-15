#!/usr/bin/env python3
"""MuseTalk 口型条件对比探针。

把同一段音频分别用：
  1) 官方算法（/data/MuseTalk 的 AudioProcessor.get_whisper_chunk 等价实现）
  2) 本仓库的新整句实现（WhisperASR + MUSETALK_FEAT_MODE=utterance）
  3) 旧的 1.04s 滑窗实现（legacy）
计算每帧口型条件向量 (50, 384)，输出相对官方的逐帧 L2 差异。

官方算法：整句 whisper 特征 → 按真实时长(50fps)截断 → 前补 4 帧零、后补 12 帧零
→ 帧 f 取 padded[2f : 2f+10] → reshape(50, 384)。

用法（在 zl-talking 目录下）：
  ../runtime/venv/talking/bin/python tests/audio_feature_probe.py \
      --wav ../zl-SoVITS/ref_audio/Keira.wav --frames 120
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)

from avatars.audio_features.whisper import WhisperASR  # noqa: E402
from avatars.musetalk.whisper.audio2feature import Audio2Feature  # noqa: E402

UTT_WIN = 10
PAD_L = 4
PAD_R = 12


def load_wav_16k(path):
    import resampy
    import soundfile as sf

    data, sr = sf.read(path, dtype='float32')
    if data.ndim > 1:
        data = data[:, 0]
    if sr != 16000:
        data = resampy.resample(x=data, sr_orig=sr, sr_new=16000).astype(np.float32)
    return data


def official_chunks(audio, audio_processor, num_frames):
    """官方 get_whisper_chunk 的等价实现。"""
    feat = audio_processor.audio2feat(audio)  # (1500, 5, 384)
    actual = max(1, int(len(audio) / 16000 * 50))
    feat = feat[:actual]
    pad_l = np.zeros((PAD_L,) + feat.shape[1:], dtype=feat.dtype)
    pad_r = np.zeros((PAD_R,) + feat.shape[1:], dtype=feat.dtype)
    padded = np.concatenate([pad_l, feat, pad_r], axis=0)
    chunks = []
    for f in range(num_frames):
        clip = padded[2 * f: 2 * f + UTT_WIN]
        if clip.shape[0] < UTT_WIN:
            pad = np.zeros((UTT_WIN - clip.shape[0],) + feat.shape[1:], dtype=feat.dtype)
            clip = np.concatenate([clip, pad], axis=0)
        chunks.append(clip.reshape(-1, 384).astype(np.float32))
        if 2 * f + UTT_WIN > padded.shape[0]:
            break
    return chunks


def legacy_chunks(audio, audio_processor, batch_size, num_frames,
                  feat_left=0, feat_right=5, stride=10):
    """旧实现：每 640ms 用滑窗重算 whisper，再按 [left,right] 切片。"""
    chunk = 320
    pieces = [audio[i:i + chunk] for i in range(0, len(audio) - chunk, chunk)]
    frames, out = [], []
    for b in range((num_frames + batch_size - 1) // batch_size):
        batch = pieces[b * batch_size * 2:(b + 1) * batch_size * 2]
        if len(batch) < batch_size * 2:
            batch += [np.zeros(chunk, dtype=np.float32)] * (batch_size * 2 - len(batch))
        frames += batch
        if len(frames) <= stride * 2:
            continue
        feat = audio_processor.audio2feat(np.concatenate(frames))
        feat = feat[:max(1, int(len(np.concatenate(frames)) / 16000 * 50))]
        for i in range(batch_size):
            center = (i + stride // 2) * 2
            left = center - feat_left * 2
            right = center + feat_right * 2
            idx = [min(max(k, 0), len(feat) - 1) for k in range(left, right)]
            out.append(feat[idx].reshape(-1, 384).astype(np.float32))
        frames = frames[-(stride * 2):]
    return out


def stats(name, ref, other):
    n = min(len(ref), len(other))
    diffs, rels = [], []
    for i in range(n):
        d = float(np.linalg.norm(ref[i] - other[i]))
        base = float(np.linalg.norm(ref[i])) + 1e-6
        diffs.append(d)
        rels.append(d / base)
    print(f"{name}: frames={n} L2 均值={np.mean(diffs):.4f} 最大={np.max(diffs):.4f} "
          f"相对误差 均值={np.mean(rels) * 100:.3f}% 最大={np.max(rels) * 100:.3f}%")
    return diffs, rels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav', required=True)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--json', default='')
    args = parser.parse_args()

    audio = load_wav_16k(args.wav)
    num_frames = min(args.frames, int(len(audio) / 16000 * 25))
    print(f"音频 {len(audio) / 16000:.2f}s，对比 {num_frames} 帧")

    ap = Audio2Feature(model_path='./models/musetalk/whisper')
    ref = official_chunks(audio, ap, num_frames)

    opt = argparse.Namespace(fps=25, batch_size=args.batch_size, l=10, r=10,
                             avatar_id='probe', model='musetalk')
    asr = WhisperASR(opt, None, ap)
    utt_id = 'probe-1'
    asr.set_utterance(utt_id, audio)
    new = []
    for f in range(min(num_frames, len(ref))):
        feat = asr._frame_feat(utt_id, f)
        new.append(np.asarray(feat, dtype=np.float32))

    d_new, rel_new = stats('新整句实现 vs 官方', ref, new)
    legacy = legacy_chunks(audio, ap, args.batch_size, min(num_frames, len(ref)))
    d_old, rel_old = stats('旧滑窗实现 vs 官方', ref, legacy)

    # 新实现与官方只应存在 GPU fp16 的非确定性误差（通常 <1%），旧实现是量级差异
    ok = np.max(rel_new) < 0.02 and np.mean(rel_old) > np.mean(rel_new) * 5
    print(f"结论：{'PASS' if ok else 'CHECK'} 新实现最大相对误差={max(rel_new) * 100:.3f}%，"
          f"旧实现平均相对误差={np.mean(rel_old) * 100:.2f}%")
    if args.json:
        import json
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump({'new': d_new, 'legacy': d_old,
                       'rel_new': rel_new, 'rel_old': rel_old}, fh)
        print(f"明细写入 {args.json}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
