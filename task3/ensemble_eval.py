# -*- coding: utf-8 -*-
"""多尺度/多架构集成评估 (零训练):
候选模型组:
  A: resnet34@384  (fold2-5, fold1已覆盖 -> 4折)
  B: resnet50@512  (fold1-5)
  C: A+B 融合
输出各组合的 OOF 欧氏/X-Y RMSE, 选最优组合用于测试推理.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image

from train_fovea import (OUT, load_records, get_canvas, set_seed, to_tensor,
                         CANVAS, Transpose, Resample)
from stage2_fovea import fold_splits

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = load_records()
splits = fold_splits(len(records))

NEW_INPUT, NEW_UNIT = 512, (512 / 2) / 4.0
OLD_INPUT, OLD_UNIT = 384, (384 / 2) / 4.0


def build_old():
    m = tvm.resnet34(weights=None); m.fc = nn.Linear(512, 2); return m


def build_new():
    m = tvm.resnet50(weights=None); m.fc = nn.Linear(2048, 2); return m


@torch.no_grad()
def infer(model, canvas_img, inp, unit):
    preds = []
    for img in (canvas_img, canvas_img.transpose(Transpose.FLIP_LEFT_RIGHT)):
        t = to_tensor(img.resize((inp, inp), Resample.BILINEAR)).unsqueeze(0).to(device)
        preds.append(model(t)[0].cpu().numpy())
    u_o, u_f = preds
    u = np.array([(u_o[0] - u_f[0]) / 2, (u_o[1] + u_f[1]) / 2])
    return (u[0] * unit + inp / 2) * CANVAS / inp, (u[1] * unit + inp / 2) * CANVAS / inp


def kag_rmse(exs, eys):
    return math.sqrt(((exs ** 2).mean() + (eys ** 2).mean()) / 2), math.sqrt((exs ** 2 + eys ** 2).mean())


# 检查各 fold 权重架构 (按文件大小)
import os
is50 = {k: os.path.getsize(OUT / f"fold{k}.pth") > 9e7 for k in range(1, 6)}
print("fold 架构(50?):", is50)

# ---- OOF 推理 ----
oofA, oofB = {}, {}     # A: 34@384, B: 50@512
for f, val_idx in enumerate(splits):
    k = f + 1
    if is50[k]:
        m = build_new(); m.load_state_dict(torch.load(OUT / f"fold{k}.pth", map_location=device))
        m = m.to(device).eval()
        for i in val_idx:
            arr, s = get_canvas(records[i]["img_path"])
            cx, cy = infer(m, Image.fromarray(arr), NEW_INPUT, NEW_UNIT)
            oofB[records[i]["id"]] = (cx / s, cy / s)
    else:
        m = build_old(); m.load_state_dict(torch.load(OUT / f"fold{k}.pth", map_location=device))
        m = m.to(device).eval()
        for i in val_idx:
            arr, s = get_canvas(records[i]["img_path"])
            cx, cy = infer(m, Image.fromarray(arr), OLD_INPUT, OLD_UNIT)
            oofA[records[i]["id"]] = (cx / s, cy / s)

haveA = len(oofA) == len(records)


def score(oof_map, ids=None):
    ids = ids or [r["id"] for r in records]
    exs = np.array([oof_map[i][0] - next(r["x"] for r in records if r["id"] == i) for i in ids])
    eys = np.array([oof_map[i][1] - next(r["y"] for r in records if r["id"] == i) for i in ids])
    return kag_rmse(exs, eys), len(ids)


if haveA:
    (kagA, euA), nA = score(oofA)
    print(f"A resnet34@384 ({nA}图): 欧氏 {euA:.2f} | X/Y {kagA:.2f}")
(kagB, euB), nB = score(oofB)
print(f"B resnet50@512 ({nB}图): 欧氏 {euB:.2f} | X/Y {kagB:.2f}")

if haveA:
    # C: 交集上融合权重扫描
    common = [r["id"] for r in records if r["id"] in oofA and r["id"] in oofB]
    gt = {r["id"]: (r["x"], r["y"]) for r in records}
    print(f"\n融合扫描 (共同 {len(common)} 图):")
    best = None
    for w in np.arange(0, 1.01, 0.1):
        exs, eys = [], []
        for i in common:
            x = w * oofA[i][0] + (1 - w) * oofB[i][0]
            y = w * oofA[i][1] + (1 - w) * oofB[i][1]
            exs.append(x - gt[i][0]); eys.append(y - gt[i][1])
        kag, eu = kag_rmse(np.array(exs), np.array(eys))
        print(f"  w(A)={w:.1f}  欧氏 {eu:6.2f} | X/Y {kag:6.2f}")
        if best is None or kag < best[0]:
            best = (kag, w)
    print(f"最优 w(A)={best[1]:.1f}, X/Y {best[0]:.2f}")
