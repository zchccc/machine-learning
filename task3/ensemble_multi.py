# -*- coding: utf-8 -*-
"""多种子 OOF 融合评估 + 测试集集成推理:
1. 加载 output(v14 seed42) / output_s777 / output_s2024 各 5 折权重 (共15个模型)
2. 每个模型对自己的验证折推理 -> OOF; 评估: 单种子集成 vs 多种子融合的 RMSE
3. 用最优融合策略对测试集推理, 生成 submission.csv
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as tvm
from PIL import Image

from train_fovea import (ROOT, DATA, load_records, get_canvas, set_seed, to_tensor,
                         CANVAS, INPUT, UNIT_PX, Transpose, Resample,
                         predict_test, make_submission)
import pandas as pd

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

set_seed()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
records = load_records()
K = 5

# 折划分必须与 train_fovea.py main() 一致: shuffle(SEED=42)
idx = np.arange(len(records))
rng = np.random.RandomState(42)
rng.shuffle(idx)
val_folds = [idx[k::K] for k in range(K)]   # fold k+1 的验证索引(在 records 中的位置)


def build():
    m = tvm.resnet34(weights=None); m.fc = nn.Linear(512, 2); return m


def load_models(outdir):
    ms = []
    for k in range(1, K + 1):
        p = Path(outdir) / f"fold{k}.pth"
        m = build()
        m.load_state_dict(torch.load(p, map_location=device))
        ms.append(m.to(device).eval())
    return ms


@torch.no_grad()
def infer_one(model, canvas_img):
    """原图 + 水平翻转 TTA, 返回画布(1024)坐标 (x, y)"""
    preds = []
    for img in (canvas_img, canvas_img.transpose(Transpose.FLIP_LEFT_RIGHT)):
        t = to_tensor(img.resize((INPUT, INPUT), Resample.BILINEAR)).unsqueeze(0).to(device)
        preds.append(model(t)[0].cpu().numpy())
    u_o, u_f = preds
    ux = (u_o[0] + (-u_f[0])) / 2
    uy = (u_o[1] + u_f[1]) / 2
    # 输入(384)像素 -> 画布(1024)像素
    return (ux * UNIT_PX + INPUT / 2) * CANVAS / INPUT, (uy * UNIT_PX + INPUT / 2) * CANVAS / INPUT


@torch.no_grad()
def oof_map(models):
    """每个模型只推理自己折的验证集 -> {rid: [(x_canvas, y_canvas), ...]}"""
    oof = {r["id"]: [] for r in records}
    for k, m in enumerate(models):
        for i in val_folds[k]:
            r = records[i]
            arr, s = get_canvas(r["img_path"])
            cx, cy = infer_one(m, Image.fromarray(arr))
            oof[r["id"]].append((cx / s, cy / s))   # 转回原图像素
    return oof


def rmse_of(pred_map, weights_per_id=None):
    exs, eys = [], []
    for r in records:
        pts = pred_map[r["id"]]
        if weights_per_id is None:
            x = np.mean([p[0] for p in pts]); y = np.mean([p[1] for p in pts])
        else:
            w = np.array(weights_per_id[r["id"]], dtype=float); w = w / w.sum()
            x = sum(wi * p[0] for wi, p in zip(w, pts)); y = sum(wi * p[1] for wi, p in zip(w, pts))
        exs.append(x - r["x"]); eys.append(y - r["y"])
    exs, eys = np.array(exs), np.array(eys)
    eu = float(np.sqrt((exs ** 2 + eys ** 2).mean()))
    kag = float(np.sqrt(((exs ** 2).mean() + (eys ** 2).mean()) / 2))
    return eu, kag


if __name__ == "__main__":
    seeds = {"s42": "backup_v14", "s777": "output_s777", "s2024": "output_s2024"}
    maps = {}
    for name, d in seeds.items():
        try:
            ms = load_models(d)
        except FileNotFoundError:
            print(f"[跳过] {d} 权重不全")
            continue
        maps[name] = oof_map(ms)
        eu, kag = rmse_of(maps[name])
        print(f"{name} ({d}): 5折集成 OOF 欧氏 {eu:6.2f}px | X/Y {kag:6.2f}px")

    if len(maps) >= 2:
        # 融合: 每图取各种子预测均值
        merged = {rid: sum((m[rid] for m in maps.values()), []) for rid in maps[next(iter(maps))]}
        eu, kag = rmse_of(merged)
        print(f"\n多种子融合 ({len(maps)}x5=15模型): 欧氏 {eu:6.2f}px | X/Y {kag:6.2f}px")

    # 测试集: 全部15模型平均
    all_models = []
    for name, d in seeds.items():
        try:
            all_models += load_models(d)
        except FileNotFoundError:
            pass
    print(f"\n测试集推理: {len(all_models)} 个模型集成 + 翻转TTA ...")
    preds = predict_test(all_models, device)
    out = make_submission(preds)
    print(f"已生成: {out}")
    df = pd.read_csv(out)
    print(df.head(6).to_string(index=False))
