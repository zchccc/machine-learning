# -*- coding: utf-8 -*-
"""逐图 OOF 误差诊断 (零训练, 复用已有权重):
1. 阶段1 OOF 粗定位逐图误差
2. 阶段2 热图精修后逐图误差
3. 找出失败模式: 粗定位脱靶? 精修无改善? 特定图像属性?
"""
import math
import numpy as np
import torch
import torchvision.models as tvm
import torch.nn as nn

from PIL import Image
from train_fovea import (OUT, load_records, get_canvas, set_seed)
from stage2_fovea import fold_splits, s1_infer
from stage2_heat import (build_heatnet, refine_heat, evaluate_full, ROI)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

records = load_records()
splits = fold_splits(len(records))

# ---- 阶段1 OOF ----
oof = {}
for f, val_idx in enumerate(splits):
    m = tvm.resnet34(); m.fc = nn.Linear(512, 2)
    m.load_state_dict(torch.load(OUT / f"fold{f + 1}.pth", map_location=device))
    m = m.to(device).eval()
    for i in val_idx:
        arr, s = get_canvas(records[i]["img_path"])
        cx, cy = s1_infer([m], Image.fromarray(arr))
        oof[records[i]["id"]] = (float(cx), float(cy), s)

# ---- 阶段2 OOF (各折模型对自己折外) ----
per_img_s2 = {}
for k in range(5):
    model = build_heatnet(device)
    model.load_state_dict(torch.load(OUT / f"hfold{k + 1}.pth", map_location=device))
    model.eval()
    _, _, refs = evaluate_full(model, [records[i] for i in splits[k]], oof, device)
    per_img_s2.update(refs)

# ---- 汇总 ----
rows = []
for r in records:
    cx, cy, s = oof[r["id"]]
    e1 = math.hypot(cx / s - r["x"], cy / s - r["y"])
    x2, y2 = per_img_s2[r["id"]]
    e2 = math.hypot(x2 - r["x"], y2 - r["y"])
    # 粗定位误差是否超出 ROI 半径 (脱靶)
    offcenter = math.hypot(cx - r["x"] * s, cy - r["y"] * s)
    rows.append({"id": r["id"], "e1": e1, "e2": e2, "off_roi": offcenter > ROI / 2,
                 "gain": e1 - e2})

e1s = np.array([r["e1"] for r in rows]); e2s = np.array([r["e2"] for r in rows])
print(f"阶段1 OOF:  mean {e1s.mean():.1f}  RMSE {math.sqrt((e1s**2).mean()):.2f}  median {np.median(e1s):.1f}")
print(f"阶段2 OOF:  mean {e2s.mean():.1f}  RMSE {math.sqrt((e2s**2).mean()):.2f}  median {np.median(e2s):.1f}")
print(f"改善>10px: {sum(1 for r in rows if r['gain'] > 10)} | 恶化>10px: {sum(1 for r in rows if r['gain'] < -10)}")
print(f"粗定位脱靶(>ROI/2={ROI//2}px): {sum(1 for r in rows if r['off_roi'])}")
rows.sort(key=lambda r: -r["e2"])
print(f"\n{'id':>6} {'e1_px':>7} {'e2_px':>7} {'gain':>7}  脱靶")
for r in rows[:20]:
    print(f"{r['id']:>6} {r['e1']:7.1f} {r['e2']:7.1f} {r['gain']:7.1f}  {'!!' if r['off_roi'] else ''}")
