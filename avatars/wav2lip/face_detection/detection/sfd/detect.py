import torch
import torch.nn.functional as F

import os
import sys
import cv2
import random
import datetime
import math
import argparse
import time
import numpy as np

import scipy.io as sio
import zipfile
from .net_s3fd import s3fd
from .bbox import *
from config import DEBUG_WAV2LIP_FACE_DETECTION


def detect(net, img, device):
    img = img - np.array([104, 117, 123])
    img = img.transpose(2, 0, 1)
    img = img.reshape((1,) + img.shape)

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    img = torch.from_numpy(img).float().to(device)
    BB, CC, HH, WW = img.size()
    with torch.no_grad():
        olist = net(img)

    bboxlist = []
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    olist = [oelem.data.cpu() for oelem in olist]
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2**(i + 2)    # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, 5))

    return bboxlist

def batch_detect(net, imgs, device):
    _t_start = time.perf_counter()
    BB, HH, WW, CC = imgs.shape
    if DEBUG_WAV2LIP_FACE_DETECTION:
        print(f"  [batch_detect] enter: batch={BB}, res={WW}x{HH}, channels={CC}")

    # ---- step 1: numpy 预处理 (归一化 + transpose) ----
    imgs = imgs - np.array([104, 117, 123])
    imgs = imgs.transpose(0, 3, 1, 2)
    _t_prep = time.perf_counter()
    if DEBUG_WAV2LIP_FACE_DETECTION:
        print(f"  [batch_detect] step1 preprocess done: {_t_prep - _t_start:.3f}s")

    # ---- step 2: 传输到 GPU ----
    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True
    imgs = torch.from_numpy(imgs).float().to(device)
    BB, CC, HH, WW = imgs.size()
    _t_gpu = time.perf_counter()
    if DEBUG_WAV2LIP_FACE_DETECTION:
        print(f"  [batch_detect] step2 gpu transfer done ({BB}x{CC}x{HH}x{WW}): {_t_gpu - _t_prep:.3f}s")

    # ---- step 3: 模型推理 (GPU) ----
    if DEBUG_WAV2LIP_FACE_DETECTION:
        print(f"  [batch_detect] step3 model inference starting...")
    _t_infer = time.perf_counter()
    with torch.no_grad():
        olist = net(imgs)
    _t_infer_end = time.perf_counter()
    infer_time = _t_infer_end - _t_infer
    if DEBUG_WAV2LIP_FACE_DETECTION:
        print(f"  [batch_detect] step3 model inference done: {infer_time:.3f}s")

    # ---- step 4: softmax + 回传 CPU ----
    _t_softmax_start = time.perf_counter()
    for i in range(len(olist) // 2):
        olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    olist = [oelem.data.cpu() for oelem in olist]
    _t_softmax_end = time.perf_counter()
    if DEBUG_WAV2LIP_FACE_DETECTION:
        print(f"  [batch_detect] step4 softmax+cpu transfer done: {_t_softmax_end - _t_softmax_start:.3f}s")

    # ---- step 5: 解码候选框 (CPU) ----
    _t_decode_start = time.perf_counter()
    bboxlist = []
    total_layers = len(olist) // 2
    for i in range(total_layers):
        _t_layer_start = time.perf_counter()
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2**(i + 2)    # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        layer_candidates = 0
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[:, 1, hindex, windex]
            loc = oreg[:, :, hindex, windex].contiguous().view(BB, 1, 4)
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]]).view(1, 1, 4)
            variances = [0.1, 0.2]
            box = batch_decode(loc, priors, variances)
            box = box[:, 0] * 1.0
            bboxlist.append(torch.cat([box, score.unsqueeze(1)], 1).cpu().numpy())
            layer_candidates += 1
        _t_layer_end = time.perf_counter()
        if DEBUG_WAV2LIP_FACE_DETECTION:
            print(f"  [batch_detect] step5 layer {i}/{total_layers} (stride={stride}, feat={FH}x{FW}): candidates={layer_candidates}, decode_time={_t_layer_end - _t_layer_start:.3f}s")

    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist):
        bboxlist = np.zeros((1, BB, 5))

    _t_end = time.perf_counter()
    if DEBUG_WAV2LIP_FACE_DETECTION:
        print(f"  [batch_detect] step5 decode total: {_t_end - _t_decode_start:.3f}s | overall: {_t_end - _t_start:.3f}s | bbox_shape={bboxlist.shape}")

    return bboxlist

def flip_detect(net, img, device):
    img = cv2.flip(img, 1)
    b = detect(net, img, device)

    bboxlist = np.zeros(b.shape)
    bboxlist[:, 0] = img.shape[1] - b[:, 2]
    bboxlist[:, 1] = b[:, 1]
    bboxlist[:, 2] = img.shape[1] - b[:, 0]
    bboxlist[:, 3] = b[:, 3]
    bboxlist[:, 4] = b[:, 4]
    return bboxlist


def pts_to_bb(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return np.array([min_x, min_y, max_x, max_y])
