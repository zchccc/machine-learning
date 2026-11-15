# -*- coding: utf-8 -*-
"""最终提交: fold2 用 v17-mixup 权重, 其余折用 v14; 5模型集成 + 水平翻转 TTA"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image

from train_fovea import (DATA, ROOT, load_records, get_canvas, set_seed, to_tensor,
                         CANVAS, INPUT, UNIT_PX, Transpose, Resample)

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CHOICE = {1: "backup_v14", 2: "output_v17", 3: "backup_v14",
          4: "backup_v14", 5: "backup_v14"}


def build():
    m = tvm.resnet34(weights=None); m.fc = nn.Linear(512, 2); return m


models = []
for k in range(1, 6):
    m = build()
    m.load_state_dict(torch.load(f"{CHOICE[k]}/fold{k}.pth", map_location=device))
    models.append(m.to(device).eval())

preds = {}
with torch.no_grad():
    for p in sorted((DATA / "test").glob("*.jpg")):
        tid = int(p.stem)
        arr, s = get_canvas(str(p))
        canvas = Image.fromarray(arr)
        factor = (CANVAS / INPUT) / s
        acc = []
        for img in (canvas, canvas.transpose(Transpose.FLIP_LEFT_RIGHT)):
            t = to_tensor(img.resize((INPUT, INPUT), Resample.BILINEAR)).unsqueeze(0).to(device)
            us = [m(t)[0].cpu().numpy() for m in models]
            acc.append(np.mean(us, axis=0))
        u_orig, u_flip = acc
        u_x = (u_orig[0] + (-u_flip[0])) / 2
        u_y = (u_orig[1] + u_flip[1]) / 2
        x_in = u_x * UNIT_PX + INPUT / 2
        y_in = u_y * UNIT_PX + INPUT / 2
        W0, H0 = Image.open(p).size
        preds[tid] = (float(np.clip(x_in * factor, 0, W0 - 1)),
                      float(np.clip(y_in * factor, 0, H0 - 1)))

sub = pd.read_csv(DATA / "sample_submission.csv")
vals = {}
for tid, (x, y) in preds.items():
    vals[f"{tid}_Fovea_X"] = round(x, 2)
    vals[f"{tid}_Fovea_Y"] = round(y, 2)
sub["value"] = sub["ImageID"].map(vals)
assert sub["value"].notna().all()
sub.to_csv(ROOT / "submission.csv", index=False)
print(f"已生成 {ROOT / 'submission.csv'} (OOF 依据: 33.13px 欧氏 / 23.43px X/Y)")
for tid in sorted(preds):
    print(f"  {tid:04d}.jpg -> ({preds[tid][0]:.1f}, {preds[tid][1]:.1f})")
