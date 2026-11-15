# -*- coding: utf-8 -*-
"""阶段2上限探测 + 失败图片检查:
1. 以 GT 为中心裁 ROI 喂给热图模型 -> 精修上限(排除粗定位误差干扰)
2. 最差图片(49/43/34/80/2/30)的 GT 位置、图像尺寸、亮度特征
3. R/G/B 通道在 GT 处的局部对比度 (黄斑在哪个通道最明显)
"""
import math
import numpy as np
import torch
from PIL import Image

from train_fovea import (OUT, load_records, get_canvas, set_seed)
from stage2_fovea import fold_splits
from stage2_heat import (build_heatnet, refine_heat, ROI)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = load_records()
splits = fold_splits(len(records))
# id -> fold
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

# ---- 1. GT为中心的精修上限 ----
errs = []
for r in records:
    arr, s = get_canvas(r["img_path"])
    cx, cy = r["x"] * s, r["y"] * s
    k = fold_of[r["id"]]
    gx, gy = refine_heat(models[k], arr, cx, cy, device, augment=True)
    e = math.hypot(gx - cx, gy - cy)
    errs.append((e, r["id"]))
es = np.array([e for e, _ in errs])
print(f"== 阶段2上限 (GT为中心ROI, 折外模型) ==")
print(f"mean {es.mean():.1f}px | RMSE {math.sqrt((es**2).mean()):.2f}px | median {np.median(es):.1f}px | max {es.max():.1f}px")

# ---- 2. 最差图检查 ----
worst = [49, 43, 34, 80, 2, 30]
gt_map = {r["id"]: r for r in records}
print("\n== 失败图片属性 ==")
print(f"{'id':>4} {'WxH':>12} {'GT':>14} {'亮度':>6} {'视野占比':>10}")
for wid in worst:
    r = gt_map[wid]
    img = Image.open(r["img_path"])
    W, H = img.size
    g = np.asarray(img.convert("L"), dtype=np.float32)
    # GT 处 100px 半径圆 vs 图像整体
    yy, xx = np.ogrid[:H, :W]
    m_patch = (xx - r["x"]) ** 2 + (yy - r["y"]) ** 2 <= 100 ** 2
    lum_patch = g[m_patch].mean()
    print(f"{wid:>4} {W:>5}x{H:<6} ({r['x']:>5.0f},{r['y']:>5.0f}) {lum_patch:6.1f} "
          f"{(r['x'] / W):>5.2f},{(r['y'] / H):.2f}")

# ---- 3. 通道对比度: GT中心patch vs 周围环 ----
print("\n== GT处通道暗度对比 (patch 60px vs 环 100-160px, 越负越暗) ==")
for ch, name in enumerate(["R", "G", "B"]):
    diffs = []
    for r in records:
        img = np.asarray(Image.open(r["img_path"]).convert("RGB"), dtype=np.float32)
        H, W = img.shape[:2]
        yy, xx = np.ogrid[:H, :W]
        d2 = (xx - r["x"]) ** 2 + (yy - r["y"]) ** 2
        inner = img[d2 <= 30 ** 2, ch].mean()
        annulus = img[(d2 >= 80 ** 2) & (d2 <= 140 ** 2), ch].mean()
        diffs.append(inner - annulus)
    diffs = np.array(diffs)
    print(f"{name}: mean {diffs.mean():6.2f}  std {diffs.std():5.2f}  "
          f"(信号强度 |mean|/std = {abs(diffs.mean()) / diffs.std():.2f})")
