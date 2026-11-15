# -*- coding: utf-8 -*-
"""诊断: 各折 OOF 误差最大的样本; 离群图(黄斑在左, X<900)的预测误差"""
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


models = []
for k in range(1, K + 1):
    m = build()
    m.load_state_dict(torch.load(f"backup_v14/fold{k}.pth", map_location=device))
    models.append(m.to(device).eval())

print("离群图归属与误差:")
errs = []
for k, m in enumerate(models):
    for i in val_folds[k]:
        r = records[i]
        arr, s = get_canvas(r["img_path"])
        cx, cy = infer_one(m, Image.fromarray(arr))
        x, y = cx / s, cy / s
        e = ((x - r["x"]) ** 2 + (y - r["y"]) ** 2) ** 0.5
        errs.append((e, r["id"], k + 1, r["x"], x))

errs.sort(reverse=True)
print("误差 Top-10:")
for e, rid, fk, gx, px in errs[:10]:
    print(f"  图{rid:03d} fold{fk}: GT=({gx:.0f},?) pred_x={px:.0f} err={e:.1f}px")
top10_sq = [e ** 2 for e, *_ in errs[:10]]
total_sq = sum(e ** 2 for e, *_ in errs)
print(f"Top-10 贡献 MSE 占比: {sum(top10_sq) / total_sq * 100:.1f}%  (共{len(errs)}图)")
outliers = [t for t in errs if t[3] < 900]
print(f"\nX<900 的图({len(outliers)}张):")
for e, rid, fk, gx, px in outliers:
    print(f"  图{rid:03d} fold{fk}: GT_x={gx:.0f} pred_x={px:.0f} err={e:.1f}px")
