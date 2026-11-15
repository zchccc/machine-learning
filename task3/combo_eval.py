# -*- coding: utf-8 -*-
"""组合评估: v14 各折 + v17-mixup fold2, 并检查难样本误差变化"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image

from train_fovea import (load_records, get_canvas, set_seed, to_tensor,
                         CANVAS, INPUT, UNIT_PX, Transpose, Resample)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = load_records()
K = 5
idx = np.arange(len(records))
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_folds = [idx[k::K] for k in range(K)]


def build():
    m = tvm.resnet34(weights=None); m.fc = nn.Linear(512, 2); return m


@torch.no_grad()
def infer_one(model, canvas_img):
    preds = []
    for img in (canvas_img, canvas_img.transpose(Transpose.FLIP_LEFT_RIGHT)):
        t = to_tensor(img.resize((INPUT, INPUT), Resample.BILINEAR)).unsqueeze(0).to(device)
        preds.append(model(t)[0].cpu().numpy())
    u_o, u_f = preds
    ux = (u_o[0] + (-u_f[0])) / 2
    uy = (u_o[1] + u_f[1]) / 2
    return (ux * UNIT_PX + INPUT / 2) * CANVAS / INPUT, (uy * UNIT_PX + INPUT / 2) * CANVAS / INPUT


# 每折选择更优的权重: v14 vs v17mixup
CHOICE = {1: "backup_v14", 2: "output_v17", 3: "backup_v14",
          4: "backup_v14", 5: "backup_v14"}

exs, eys = [], []
hard_err = {}
for k in range(1, K + 1):
    m = build()
    m.load_state_dict(torch.load(f"{CHOICE[k]}/fold{k}.pth", map_location=device))
    m = m.to(device).eval()
    for i in val_folds[k - 1]:
        r = records[i]
        arr, s = get_canvas(r["img_path"])
        cx, cy = infer_one(m, Image.fromarray(arr))
        x, y = cx / s, cy / s
        exs.append(x - r["x"]); eys.append(y - r["y"])
        hard_err[r["id"]] = ((x - r["x"]) ** 2 + (y - r["y"]) ** 2) ** 0.5

exs, eys = np.array(exs), np.array(eys)
eu = float(np.sqrt((exs ** 2 + eys ** 2).mean()))
kg = float(np.sqrt(((exs ** 2).mean() + (eys ** 2).mean()) / 2))
print(f"v14+v17mixup(fold2) 组合 OOF: 欧氏 {eu:.2f}px | X/Y口径 {kg:.2f}px")
print("(对照: v14 全折 35.87/25.37, 15模型融合 36.36/25.71)")

print("\n难样本误差 (v14基线 -> 组合):")
base = {49: 162, 43: 101, 72: 81, 80: 69, 14: 69, 2: 64}
for h, b in base.items():
    print(f"  图{h:03d}: {b} -> {hard_err[h]:.0f}")
