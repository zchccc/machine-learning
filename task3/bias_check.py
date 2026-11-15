# -*- coding: utf-8 -*-
"""OOF 误差偏差-方差分解: 若存在系统性偏移向量, 减去它可降低 RMSE (零成本).
用旧 resnet34@384 fold2-5 + OOF 逻辑重算全部 80 图 OOF.
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

# 旧架构 resnet34@384 (硬编码, 不依赖被改的 build_model)
OLD_INPUT = 384
OLD_UNIT = (OLD_INPUT / 2) / 4.0


def build_old():
    m = tvm.resnet34(weights=None)
    m.fc = nn.Linear(512, 2)
    return m


@torch.no_grad()
def old_infer(model, canvas_img):
    preds = []
    for img in (canvas_img, canvas_img.transpose(Transpose.FLIP_LEFT_RIGHT)):
        t = to_tensor(img.resize((OLD_INPUT, OLD_INPUT), Resample.BILINEAR)).unsqueeze(0).to(device)
        preds.append(model(t)[0].cpu().numpy())
    u_o, u_f = preds
    u = np.array([(u_o[0] - u_f[0]) / 2, (u_o[1] + u_f[1]) / 2])
    x_in = u[0] * OLD_UNIT + OLD_INPUT / 2
    y_in = u[1] * OLD_UNIT + OLD_INPUT / 2
    return x_in * CANVAS / OLD_INPUT, y_in * CANVAS / OLD_INPUT


# fold1.pth 已是 resnet50, 无法用于旧架构; fold2-5 仍可用, 折1 用折2模型代(近似)
oof = {}
for f, val_idx in enumerate(splits):
    k = f + 1 if f + 1 >= 2 else 2        # fold1 用 fold2 权重(近似)
    m = build_old()
    m.load_state_dict(torch.load(OUT / f"fold{k}.pth", map_location=device))
    m = m.to(device).eval()
    for i in val_idx:
        arr, s = get_canvas(records[i]["img_path"])
        cx, cy = old_infer(m, Image.fromarray(arr))
        oof[records[i]["id"]] = (float(cx), float(cy), s)

exs, eys = [], []
for r in records:
    cx, cy, s = oof[r["id"]]
    exs.append(cx / s - r["x"]); eys.append(cy / s - r["y"])
exs, eys = np.array(exs), np.array(eys)
bx, by = exs.mean(), eys.mean()
rmse_eu = math.sqrt((exs ** 2 + eys ** 2).mean())
rmse_kag = math.sqrt(((exs ** 2).mean() + (eys ** 2).mean()) / 2)
rmse_eu_c = math.sqrt(((exs - bx) ** 2 + (eys - by) ** 2).mean())
rmse_kag_c = math.sqrt((((exs - bx) ** 2).mean() + ((eys - by) ** 2).mean()) / 2)
print(f"OOF(旧34@384): 欧氏 {rmse_eu:.2f} -> 去偏差 {rmse_eu_c:.2f} | "
      f"X/Y {rmse_kag:.2f} -> 去偏差 {rmse_kag_c:.2f}")
print(f"系统偏差向量: ({bx:.1f}, {by:.1f}) px  (|b|={math.hypot(bx, by):.1f})")
print(f"误差分位数: ex p10/p50/p90 = {np.percentile(exs, 10):.0f}/{np.percentile(exs, 50):.0f}/{np.percentile(exs, 90):.0f}")
print(f"            ey p10/p50/p90 = {np.percentile(eys, 10):.0f}/{np.percentile(eys, 50):.0f}/{np.percentile(eys, 90):.0f}")
