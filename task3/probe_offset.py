# -*- coding: utf-8 -*-
"""验证 '热图模型把输出钉死在 ROI 中心' 假设:
对 GT 为中心的 ROI 施加人工偏移 (dx=0,40,80,...,240), 看模型输出跟随多少.
同时记录热图峰值响应, 判断能否用峰值置信度挑选最佳 ROI 中心.
"""
import math
import numpy as np
import torch
from PIL import Image

from train_fovea import OUT, load_records, get_canvas, set_seed
from stage2_fovea import fold_splits
from stage2_heat import build_heatnet, soft_argmax, roi_variants, ROI, S2_INPUT
from train_fovea import to_tensor

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = load_records()
splits = fold_splits(len(records))
fold_of = {}
for k, val_idx in enumerate(splits):
    for i in val_idx:
        fold_of[records[i]["id"]] = k

models = {}
for k in range(5):
    m = build_heatnet(device)
    m.load_state_dict(torch.load(OUT / f"hfold{k + 1}.pth", map_location=device))
    m.eval()
    models[k] = m


@torch.no_grad()
def probe(model, arr, cx, cy):
    """返回 (精修坐标, 无TTA单视角热图峰值logit)"""
    img, mx, my = next(iter(roi_variants(arr, cx, cy, augment=False)))
    t = to_tensor(img).unsqueeze(0).to(device)
    hm = model(t)
    peak = float(hm.max())
    xy = soft_argmax(hm)[0].cpu().numpy()
    return mx(xy[0]), my(xy[1]), peak


# 偏移响应曲线: 对若干图, ROI中心 = GT + dx 方向偏移
offsets = [0, 40, 80, 120, 160, 200, 240]
print(f"{'offset':>7} {'跟随率':>7} {'err_px':>7} {'峰值':>7} (跟随率=输出偏移/中心偏移, 1为完全跟随)")
sample = [r for r in records if r["id"] in (1, 5, 10, 20, 30, 40, 50, 60, 70, 80)]
for off in offsets:
    follows, errs, peaks = [], [], []
    for r in sample:
        arr, s = get_canvas(r["img_path"])
        gx, gy = r["x"] * s, r["y"] * s
        cx, cy = gx + off, gy           # 中心向 +x 偏移
        m = models[fold_of[r["id"]]]
        px, py, peak = probe(m, arr, cx, cy)
        errs.append(math.hypot(px - gx, py - gy))
        follows.append((px - (cx - off)) / off if off else 0.0)
        peaks.append(peak)
    print(f"{off:>7} {np.mean(follows):7.2f} {np.mean(errs):7.1f} {np.mean(peaks):7.2f}")
