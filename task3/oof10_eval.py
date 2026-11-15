# -*- coding: utf-8 -*-
"""v10 10折 合并 OOF 评估: 每图用其折外模型推理 -> 80 图汇总 RMSE
并与 5 折版本(35.01)对比; 同时输出多模型集成(全10模型)的 OOF 上界参考."""
import math
import numpy as np
import torch
from PIL import Image

from train_fovea import (OUT, load_records, get_canvas, set_seed, to_tensor,
                         CANVAS, INPUT, UNIT_PX, Transpose, Resample)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = load_records()

import torch.nn as nn
import torchvision.models as tvm
from train_fovea import build_model

K = 10
idx = np.arange(len(records))
rng = np.random.RandomState(42)
rng.shuffle(idx)

models = {}
for k in range(1, K + 1):
    m = build_model(device)
    m.load_state_dict(torch.load(OUT / f"fold{k}.pth", map_location=device))
    m.eval()
    models[k] = m


@torch.no_grad()
def infer(ms, canvas_img):
    preds = []
    for img in (canvas_img, canvas_img.transpose(Transpose.FLIP_LEFT_RIGHT)):
        t = to_tensor(img.resize((INPUT, INPUT), Resample.BILINEAR)).unsqueeze(0).to(device)
        for m in ms:
            preds.append(m(t)[0].cpu().numpy())
    us = np.mean(preds, axis=0)   # 注意: 翻转求平均应先成对处理, 此处近似(与训练评估同口径)
    # 正确处理: 翻转 x 取负后平均
    return us


@torch.no_grad()
def infer2(ms, canvas_img):
    """与 train_fovea.predict_test 同口径: 每模型原图+翻转成对平均"""
    acc = []
    for m in ms:
        p = []
        for img in (canvas_img, canvas_img.transpose(Transpose.FLIP_LEFT_RIGHT)):
            t = to_tensor(img.resize((INPUT, INPUT), Resample.BILINEAR)).unsqueeze(0).to(device)
            p.append(m(t)[0].cpu().numpy())
        u = np.array([(p[0][0] - p[1][0]) / 2, (p[0][1] + p[1][1]) / 2])
        acc.append(u)
    return np.mean(acc, axis=0)


exs, eys = [], []
exs_ens, eys_ens = [], []
all_models = list(models.values())
for f, i in enumerate(idx):
    k = f % K + 1                      # val fold = 该图所属折
    r = records[i]
    arr, s = get_canvas(r["img_path"])
    ci = Image.fromarray(arr)
    # 折外单模型
    u = infer2([models[k]], ci)
    x = (u[0] * UNIT_PX + INPUT / 2) * CANVAS / INPUT
    y = (u[1] * UNIT_PX + INPUT / 2) * CANVAS / INPUT
    exs.append(x / s - r["x"]); eys.append(y / s - r["y"])
    # 全10模型集成(含见过的模型, 测试口径参考)
    u2 = infer2(all_models, ci)
    x2 = (u2[0] * UNIT_PX + INPUT / 2) * CANVAS / INPUT
    y2 = (u2[1] * UNIT_PX + INPUT / 2) * CANVAS / INPUT
    exs_ens.append(x2 / s - r["x"]); eys_ens.append(y2 / s - r["y"])

for tag, ex, ey in (("折外单模型", exs, eys), ("10模型集成(参考,含训练内)", exs_ens, eys_ens)):
    ex, ey = np.array(ex), np.array(ey)
    eu = math.sqrt((ex ** 2 + ey ** 2).mean())
    kg = math.sqrt(((ex ** 2).mean() + (ey ** 2).mean()) / 2)
    print(f"{tag}: 欧氏 {eu:.2f}px | X/Y口径 {kg:.2f}px")
