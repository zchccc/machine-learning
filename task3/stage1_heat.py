# -*- coding: utf-8 -*-
"""
黄斑中心凹定位 - 阶段1: 全图级联热图 (v2)
================================================
直接在全图上用热图+soft-argmax 回归中心凹(无需两阶段):
  - 画布 1024 -> 输入 512x512, 热图 64x64, 高斯 sigma=2.0
  - ResNet34 (stride8) + 卷积头, 焦点加权MSE, soft-argmax 亚像素提取
  - 5折CV + 测试集 5模型集成 + 水平翻转TTA

用法:
  py stage1_heat.py [--epochs 120] [--folds 5]
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm

from train_fovea import (DATA, OUT, ROOT, CANVAS, SEED,
                         load_records, get_canvas, photometric, to_tensor,
                         rand_crop_box, set_seed, Transpose, Resample)
from stage2_fovea import fold_splits

INP = 512
HMG = 64
SIG = 2.0
BB_LR, HD_LR = 3e-4, 1.5e-3


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)
        except Exception:
            m = tvm.resnet34(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        return self.head(self.stem(x))


def gauss(cx, cy, size=HMG, sigma=SIG):
    g = torch.arange(size, dtype=torch.float32) + 0.5
    xs = g.view(1, size, 1).expand(1, size, size) * (INP / size)
    ys = g.view(1, 1, size).expand(1, size, size) * (INP / size)
    return torch.exp(-(((xs - cx) ** 2) + ((ys - cy) ** 2)) / (2 * (sigma * INP / size) ** 2)).clamp(0, 1)


def soft_argmax(hm):
    b = hm.shape[0]
    p = torch.softmax(hm.view(b, -1) * 4.0, dim=1)
    g = torch.arange(HMG, device=hm.device, dtype=torch.float32) + 0.5
    xs = g.view(1, HMG, 1).expand(1, HMG, HMG).reshape(1, -1)
    ys = g.view(1, 1, HMG).expand(1, HMG, HMG).reshape(1, -1)
    return torch.stack([(p * xs).sum(1) * (INP / HMG), (p * ys).sum(1) * (INP / HMG)], dim=1)


class DS(Dataset):
    def __init__(self, records, train=True):
        self.r, self.train = records, train

    def __len__(self):
        return len(self.r)

    def __getitem__(self, k):
        r = self.r[k]
        arr, s = get_canvas(r["img_path"])
        canvas = Image.fromarray(arr)
        x, y = r["x"] * s, r["y"] * s
        w = CANVAS
        if self.train:
            canvas = photometric(canvas)
            if random.random() < 0.5:
                canvas = canvas.transpose(Transpose.FLIP_LEFT_RIGHT)
                x = CANVAS - x
            th = math.radians(random.uniform(-12, 12))
            sc = random.uniform(0.9, 1.1)
            tx, ty = random.uniform(-20, 20), random.uniform(-20, 20)
            cos, sin = math.cos(th) * sc, math.sin(th) * sc
            Fm = np.array([[cos, -sin, 512 - 512 * cos + 512 * sin + tx],
                           [sin, cos, 512 - 512 * sin - 512 * cos + ty], [0, 0, 1]])
            p = Fm @ np.array([x, y, 1.0])
            if 40 <= p[0] <= CANVAS - 40 and 40 <= p[1] <= CANVAS - 40:
                inv = np.linalg.inv(Fm)
                a = tuple(inv[0, :2].tolist() + [inv[0, 2]] + inv[1, :2].tolist() + [inv[1, 2]])
                canvas = canvas.transform((CANVAS, CANVAS), 0, a, resample=Resample.BILINEAR)
                x, y = float(p[0]), float(p[1])
            i, j, h, w = rand_crop_box()
            canvas = canvas.crop((j, i, j + w, i + h)).resize((INP, INP), Resample.BILINEAR)
            x = (x - j) * INP / w
            y = (y - i) * INP / h
            if random.random() < 0.3:
                canvas = canvas.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.2)))
        else:
            canvas = canvas.resize((INP, INP), Resample.BILINEAR)
            x = x * INP / CANVAS
            y = y * INP / CANVAS
        img = to_tensor(canvas)
        heat = gauss(x, y)[0]
        factor = (CANVAS / INP) / s if not self.train else (w / INP) / s
        return img, heat, torch.tensor([factor], dtype=torch.float32)


def hm_loss(pred, target):
    w = 1.0 + 8.0 * target
    return (w * (pred - target) ** 2).mean()


def eval_model(model, loader, device):
    """热图路径的坐标误差(px原图)"""
    model.eval()
    P, T, Ff = [], [], []
    with torch.no_grad():
        for img, heat, factor in loader:
            xy = soft_argmax(model(img.to(device))).cpu()
            P.append(xy)
            T.append((heat.nonzero(as_tuple=False).float().mean(0) + 0.5).flip(-1) * (INP / HMG))  # 近似
            Ff.append(factor)
    # 用精确 GT 重新计算(从数据集): 直接返回 None
    return None


@torch.no_grad()
def eval_exact(model, records, device):
    """无增强推理 + 翻转TTA -> 原图像素误差"""
    model.eval()
    ex, ey = [], []
    for r in records:
        arr, s = get_canvas(r["img_path"])
        xys = []
        for img, fm, fx in variants(Image.fromarray(arr)):
            t = to_tensor(img).unsqueeze(0).to(device)
            xy = soft_argmax(model(t))[0].cpu().numpy()
            xys.append(fx(xy))
        x = np.mean([a[0] for a in xys]); y = np.mean([a[1] for a in xys])
        ex.append(x / s - r["x"]); ey.append(y / s - r["y"])
    ex, ey = np.array(ex), np.array(ey)
    eu = float(np.sqrt((ex ** 2 + ey ** 2).mean()))
    kg = float(np.sqrt(((ex ** 2).mean() + (ey ** 2).mean()) / 2))
    return eu, kg, (ex, ey)


def variants(canvas):
    """无随机增强: 原图 + 水平翻转 TTA"""
    img = canvas.resize((INP, INP), Resample.BILINEAR)
    yield img, None, (lambda xy: (xy[0] * CANVAS / INP, xy[1] * CANVAS / INP))
    f = img.transpose(Transpose.FLIP_LEFT_RIGHT)
    yield f, None, (lambda xy: ((INP - xy[0]) * CANVAS / INP, xy[1] * CANVAS / INP))


def train_fold(tr, va, epochs, device, fid):
    torch.manual_seed(SEED + 300 + fid)
    random.seed(SEED + 300 + fid)
    model = Net().to(device)
    loader = DataLoader(DS(tr, True), batch_size=8, shuffle=True, pin_memory=True)
    opt = torch.optim.AdamW([
        {"params": [p for p in model.stem.parameters()], "lr": BB_LR},
        {"params": [p for p in model.head.parameters()], "lr": HD_LR},
    ], weight_decay=1e-4)
    total = epochs * len(loader); warm = 2 * len(loader)
    scaler = torch.amp.GradScaler("cuda")
    best = {"eu": float("inf")}
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for it, (img, heat, _) in enumerate(loader):
            step = ep * len(loader) + it
            lr = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            opt.param_groups[0]["lr"] = BB_LR * lr
            opt.param_groups[1]["lr"] = HD_LR * lr
            img, heat = img.to(device), heat.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = hm_loss(model(img), heat)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        eu, kg, _ = eval_exact(model, va, device)
        tag = ""
        if eu < best["eu"]:
            best = {"eu": eu, "kg": kg, "ep": ep, "state": copy.deepcopy(model.state_dict())}
            tag = "  <- best"
        if (ep + 1) % 5 == 0 or tag:
            print(f"  f{fid} ep{ep + 1:03d} loss {tot / max(nb, 1):.4f} 欧氏 {eu:6.2f} | X/Y {kg:6.2f}{tag}",
                  flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    set_seed()
    device = torch.device("cuda")
    print(f"设备: {device}")
    torch.backends.cudnn.benchmark = True
    records = load_records()
    splits = fold_splits(len(records))
    models, eus, kgs = [], [], []
    t0 = time.time()
    if not args.skip_train:
        for k in range(args.folds):
            va_idx = splits[k]
            tr_idx = np.setdiff1d(np.arange(len(records)), va_idx)
            print(f"===== Heat1 Fold {k + 1}/{args.folds} =====")
            best = train_fold([records[i] for i in tr_idx], [records[i] for i in va_idx],
                              args.epochs, device, k + 1)
            m = Net().to(device); m.load_state_dict(best["state"]); m.eval()
            models.append(m); eus.append(best["eu"]); kgs.append(best["kg"])
            torch.save(best["state"], OUT / f"h1fold{k + 1}.pth")
            print(f"Heat1 Fold {k + 1} 最佳 ep{best['ep'] + 1}: 欧氏 {best['eu']:.2f} | X/Y {best['kg']:.2f}",
                  flush=True)
        print("\n各折欧氏: " + ", ".join(f"{r:.2f}" for r in eus))
        print(f"平均欧氏 {np.mean(eus):.2f} ± {np.std(eus):.2f} | 平均X/Y口径 {np.mean(kgs):.2f} px")
        (OUT / "cv_summary_h1.json").write_text(json.dumps(
            {"fold_eu": [round(r, 3) for r in eus], "fold_xy": [round(r, 3) for r in kgs],
             "mean_eu": round(float(np.mean(eus)), 3), "mean_xy": round(float(np.mean(kgs)), 3)},
            ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        for k in range(args.folds):
            m = Net().to(device)
            m.load_state_dict(torch.load(OUT / f"h1fold{k + 1}.pth", map_location=device))
            m.eval(); models.append(m)

    # 测试集
    print("\n测试集推理...")
    preds = {}
    with torch.no_grad():
        for p in sorted((DATA / "test").glob("*.jpg")):
            tid = int(p.stem)
            arr, s = get_canvas(str(p))
            xs, ys = [], []
            for m in models:
                for img, _, fx in variants(Image.fromarray(arr)):
                    t = to_tensor(img).unsqueeze(0).to(device)
                    xy = soft_argmax(m(t))[0].cpu().numpy()
                    gx, gy = fx(xy)
                    xs.append(gx); ys.append(gy)
            W0, H0 = Image.open(p).size
            preds[tid] = (float(np.clip(np.mean(xs) / s, 0, W0 - 1)),
                          float(np.clip(np.mean(ys) / s, 0, H0 - 1)))
    sub = pd.read_csv(DATA / "sample_submission.csv")
    vals = {}
    for tid, (x, y) in preds.items():
        vals[f"{tid}_Fovea_X"] = round(x, 2)
        vals[f"{tid}_Fovea_Y"] = round(y, 2)
    sub["value"] = sub["ImageID"].map(vals)
    sub.to_csv(ROOT / "submission_h1.csv", index=False)
    print(f"已生成 {ROOT / 'submission_h1.csv'}")
    for tid in sorted(preds):
        print(f"  {tid:04d}.jpg -> ({preds[tid][0]:.1f}, {preds[tid][1]:.1f})")


if __name__ == "__main__":
    main()
