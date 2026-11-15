# -*- coding: utf-8 -*-
"""检查 15 个模型(seed42/777/2024 x 5折)在难样本(049/043/072/080/014/002)上的
逐模型预测: 是否存在能定位对的模型(-> 加权融合可修)还是全错(-> 必须换路线)"""
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

SEED_DIRS = {"s42": "backup_v14", "s777": "output_s777", "s2024": "output_s2024"}
HARD = [49, 43, 72, 80, 14, 2]


def build():
    m = tvm.resnet34(weights=None); m.fc = nn.Linear(512, 2); return m


@torch.no_grad()
def infer_one(model, canvas_img):
    """翻转TTA, 返回画布(1024)坐标"""
    preds = []
    for img in (canvas_img, canvas_img.transpose(Transpose.FLIP_LEFT_RIGHT)):
        t = to_tensor(img.resize((INPUT, INPUT), Resample.BILINEAR)).unsqueeze(0).to(device)
        preds.append(model(t)[0].cpu().numpy())
    u_o, u_f = preds
    ux = (u_o[0] + (-u_f[0])) / 2
    uy = (u_o[1] + u_f[1]) / 2
    return (ux * UNIT_PX + INPUT / 2) * CANVAS / INPUT, (uy * UNIT_PX + INPUT / 2) * CANVAS / INPUT


# 加载 15 个模型, 记录 {图id: [(seed, fold, x, y, err), ...]}
per_img = {h: [] for h in HARD}
for tag, d in SEED_DIRS.items():
    for k in range(1, K + 1):
        m = build()
        m.load_state_dict(torch.load(f"{d}/fold{k}.pth", map_location=device))
        m = m.to(device).eval()
        for i in val_folds[k - 1]:
            r = records[i]
            if r["id"] not in per_img:
                continue
            arr, s = get_canvas(r["img_path"])
            cx, cy = infer_one(m, Image.fromarray(arr))
            x, y = cx / s, cy / s
            e = ((x - r["x"]) ** 2 + (y - r["y"]) ** 2) ** 0.5
            per_img[r["id"]].append((tag, k, x, y, e))

for h in HARD:
    r = next(rc for rc in records if rc["id"] == h)
    rows = per_img[h]
    xs = [t[2] for t in rows]; ys = [t[3] for t in rows]
    print(f"\n图{h:03d} GT=({r['x']:.0f},{r['y']:.0f})  15模型: x∈[{min(xs):.0f},{max(xs):.0f}] "
          f"y∈[{min(ys):.0f},{max(ys):.0f}]")
    ok = [t for t in rows if t[4] < 50]
    print(f"  误差<50px 的模型: {len(ok)}/15")
    for tag, k, x, y, e in sorted(rows, key=lambda t: t[4]):
        flag = " <-- 准" if e < 50 else ""
        print(f"    {tag} f{k}: ({x:7.0f},{y:7.0f}) err={e:5.0f}{flag}")
