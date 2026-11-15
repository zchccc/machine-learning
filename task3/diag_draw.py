# -*- coding: utf-8 -*-
"""把 GT(红) 与预测(蓝) 画到原图上, 裁剪黄斑区域保存, 供人工检查"""
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image, ImageDraw

from train_fovea import (load_records, get_canvas, set_seed, to_tensor,
                         CANVAS, INPUT, UNIT_PX, Transpose, Resample)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = {r["id"]: r for r in load_records()}
K = 5
idx = np.arange(80)
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


# {图id: 折号} (由 val_folds 推出, recs 与训练时同序)
recs = load_records()
img_fold = {}
for k in range(K):
    for i in val_folds[k]:
        img_fold[recs[i]["id"]] = k + 1

targets = [49, 43, 72, 80, 2]
for tid in targets:
    fk = img_fold[tid]
    m = build()
    m.load_state_dict(torch.load(f"backup_v14/fold{fk}.pth", map_location=device))
    m = m.to(device).eval()
    r = records[tid]
    arr, s = get_canvas(r["img_path"])
    cx, cy = infer_one(m, Image.fromarray(arr))
    px, py = cx / s, cy / s
    img = Image.open(r["img_path"]).convert("RGB")
    W, H = img.size
    sc = min(1.0, 900 / max(W, H))
    if sc < 1:
        img = img.resize((int(W * sc), int(H * sc)), Resample.BILINEAR)
    dr = ImageDraw.Draw(img)
    gx, gy = r["x"] * sc, r["y"] * sc
    qx, qy = px * sc, py * sc
    for rad, col, thick in [(28, "red", 6), (14, "red", 4), (28, "blue", 6), (14, "blue", 4)]:
        pass
    # GT 红, 预测蓝
    for rr, col in [(30, "#ff0000"), (16, "#ff0000")]:
        dr.ellipse([gx - rr, gy - rr, gx + rr, gy + rr], outline=col, width=6)
    for rr, col in [(30, "#0066ff"), (16, "#0066ff")]:
        dr.ellipse([qx - rr, qy - rr, qx + rr, qy + rr], outline=col, width=6)
    img.save(f"diag_{tid:03d}.png")
    print(f"图{tid:03d} fold{fk}: GT=({r['x']:.0f},{r['y']:.0f}) pred=({px:.0f},{py:.0f}) err={((px-r['x'])**2+(py-r['y'])**2)**.5:.0f}px -> diag_{tid:03d}.png")
