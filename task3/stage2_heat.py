# -*- coding: utf-8 -*-
"""
黄斑中心凹定位 - 阶段2b: ROI 热图定位 (heatmap + soft-argmax)
============================================================
依赖阶段1 (output/fold{k}.pth) 的 OOF 粗定位。

v4 关键修复 (2026-10-25 诊断):
  v1-v3 的纯热图 MSE 存在平凡解 —— 模型输出全零/平坦热图时 loss 已很低,
  soft-argmax 退化为"返回 ROI 中心" (实测跟随率 1.01), 阶段2从未真正定位.
  v4 在热图 MSE 之外加入 DSNT 式直接坐标监督:
      loss = SmoothL1(E[z] , GT) + w_var * Var(z) + w_hm * 热图MSE
  其中 z = softmax(热图logits), E[z] 为期望坐标 —— 唯一低 loss 解是正确定位.

用法:
  py stage2_heat.py                     # 完整 (5折 x 80轮)
  py stage2_heat.py --epochs 2 --folds 2
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
                         set_seed, Transpose, Resample)
from stage2_fovea import (S1_INPUT, KAG,
                          fold_splits, s1_infer, JITTER)

# --- v6 配置: 迭代精修 (小ROI高分辨率, 推理时多轮收敛) ---
ROI = 320          # 画布坐标裁剪边长
S2_INPUT = 384     # 网络输入 (超采样 1.2x, 保留黄斑细节)
HM = 48            # 热图边长 (384/8)
SIGMA = 1.2        # 高斯 sigma (热图像素)
ITER_REFINE = 4    # 推理时迭代精修次数
# --- v3: 校准(平滑/锐化/对比度) 三视角 TTA ---
from PIL import ImageFilter as _IF
from PIL import ImageEnhance as _IE
BACKBONE_LR = 3e-4
HEAD_LR = 1.5e-3


def gauss_heatmap(cx, cy, size=HM, sigma=SIGMA, device="cpu"):
    """(cx, cy) 为 ROI 输入坐标(0..S2_INPUT) -> HMxHM 高斯热图张量"""
    xs = torch.arange(size, dtype=torch.float32, device=device).view(1, size, 1).expand(1, size, size) + 0.5
    ys = torch.arange(size, dtype=torch.float32, device=device).view(1, 1, size).expand(1, size, size) + 0.5
    scale = S2_INPUT / size
    h = torch.exp(-(((xs * scale - cx) ** 2) + ((ys * scale - cy) ** 2)) / (2 * (sigma * scale) ** 2))
    return h.clamp(0, 1)


class HeatNet(nn.Module):
    """ResNet34 前半段(stride 8) + 轻量卷积热图头"""

    def __init__(self):
        super().__init__()
        try:
            m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)
        except Exception:
            m = tvm.resnet34(weights=None)
        self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool,
                                  m.layer1, m.layer2)          # stride 8
        self.head = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1),
        )

    def forward(self, x):
        f = self.stem(x)
        return self.head(f)                                    # (B,1,HM,HM) logits


def soft_argmax(hm):
    """热图 -> 输入坐标 (0..S2_INPUT)。hm: (B,1,HM,HM)"""
    b = hm.shape[0]
    p = hm.view(b, -1)
    p = torch.softmax(p * 4.0, dim=1)                          # 锐化后归一化
    grid = torch.arange(HM, device=hm.device, dtype=torch.float32) + 0.5
    xs = grid.view(1, HM, 1).expand(1, HM, HM).reshape(1, -1)
    ys = grid.view(1, 1, HM).expand(1, HM, HM).reshape(1, -1)
    gx = (p * xs).sum(1) * (S2_INPUT / HM)
    gy = (p * ys).sum(1) * (S2_INPUT / HM)
    return torch.stack([gx, gy], dim=1)


def hm_loss(pred, target):
    """焦点式加权 MSE: 前景权重更高"""
    w = 1.0 + 8.0 * target
    return (w * (pred - target) ** 2).mean()


def coord_loss(hm, gt_xy):
    """v7: DSNT 式直接坐标监督 (未锐化 softmax 的期望坐标).
    梯度处处非零且平滑 —— 峰值初始在错误位置时也能被拉回 GT,
    与锐化版(梯度只在峰值附近)不同, 适合从随机初始化学习定位.
    hm: (B,1,HM,HM) logits, gt_xy: (B,2) GT 输入坐标(0..S2_INPUT)"""
    b = hm.shape[0]
    p = torch.softmax(hm.view(b, -1), dim=1)
    grid = torch.arange(HM, device=hm.device, dtype=torch.float32) + 0.5
    xs = grid.view(1, HM, 1).expand(1, HM, HM).reshape(1, -1)
    ys = grid.view(1, 1, HM).expand(1, HM, HM).reshape(1, -1)
    ex = (p * xs).sum(1) * (S2_INPUT / HM)
    ey = (p * ys).sum(1) * (S2_INPUT / HM)
    l1 = F.smooth_l1_loss(torch.stack([ex, ey], 1), gt_xy, reduction="none").sum(1)
    vx = (p * xs ** 2).sum(1) * (S2_INPUT / HM) ** 2 - ex ** 2
    vy = (p * ys ** 2).sum(1) * (S2_INPUT / HM) ** 2 - ey ** 2
    var = torch.sqrt(torch.clamp(vx, min=0) + torch.clamp(vy, min=0))
    return (l1 + 0.01 * var).mean()


class HeatRoiDataset(Dataset):
    def __init__(self, records, oof, train=True, err_pool=None):
        self.records, self.oof, self.train = records, oof, train
        self.err_pool = err_pool      # 训练时自举采样的 OOF 误差向量池 [(ex, ey)]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, k):
        r = self.records[k]
        arr, s = get_canvas(r["img_path"])
        cx, cy, _ = self.oof[r["id"]]
        if self.train:
            if self.err_pool is not None and len(self.err_pool) > 0:
                ex, ey = self.err_pool[random.randrange(len(self.err_pool))]
                cx += ex; cy += ey        # 自举: 模拟真实粗定位误差(含长尾)
            cx += random.gauss(0, JITTER)
            cy += random.gauss(0, JITTER)
            # v6: 迭代精修式训练 —— 偏移尺度模拟推理链: 初始粗定位误差(长尾~150px)
            # 随迭代轮数衰减(每轮约减半), 让模型学会"把中心拉回 GT"的局部修正
            if random.random() < 0.5:
                steps = random.randint(1, ITER_REFINE)
                rad = max(5.0, 150.0 * (0.5 ** (steps - 1)))    # 150 -> 75 -> 37 -> 18
                ang = random.uniform(0, 2 * math.pi)
                cx += math.cos(ang) * random.uniform(0, rad)
                cy += math.sin(ang) * random.uniform(0, rad)
        x0 = int(min(max(round(cx - ROI / 2), 0), CANVAS - ROI))
        y0 = int(min(max(round(cy - ROI / 2), 0), CANVAS - ROI))
        roi = Image.fromarray(arr[y0:y0 + ROI, x0:x0 + ROI])
        gx, gy = r["x"] * s - x0, r["y"] * s - y0
        flip = self.train and random.random() < 0.5
        if flip:
            roi = roi.transpose(Transpose.FLIP_LEFT_RIGHT)
            gx = ROI - gx
        if self.train:
            roi = photometric(roi)
            if random.random() < 0.3:
                roi = roi.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        img = to_tensor(roi.resize((S2_INPUT, S2_INPUT), Resample.BILINEAR))
        # GT 输入坐标
        gx_in, gy_in = gx * S2_INPUT / ROI, gy * S2_INPUT / ROI
        heat = gauss_heatmap(gx_in, gy_in)[0]
        factor = (ROI / S2_INPUT) / s
        return img, heat, torch.tensor([gx_in, gy_in], dtype=torch.float32), \
            torch.tensor([x0, y0], dtype=torch.float32)


def pix_err_px(pred_xy_in, gt_xy_in, factors):
    """输入坐标误差 -> 原图像素误差, 返回欧氏RMSE与X/Y平均RMSE"""
    d = (pred_xy_in - gt_xy_in) * factors[:, None]
    ex, ey = d[:, 0], d[:, 1]
    return (float(torch.sqrt((ex ** 2 + ey ** 2).mean())),
            float(torch.sqrt(((ex ** 2).mean() + (ey ** 2).mean()) / 2)))


def build_heatnet(device):
    return HeatNet().to(device)


@torch.no_grad()
def probe_shift(model, records, device, n=8, off=60.0):
    """v7 选型守卫: 合成偏移探测 —— ROI 中心 = GT + 固定偏移, 看精修能拉回多少.
    "猜中心"的退化模型此指标恒等于 off, 真定位模型显著小于 off."""
    errs = []
    for r in records[:n]:
        arr, s = get_canvas(r["img_path"])
        gx, gy = r["x"] * s, r["y"] * s
        for dx, dy in ((off, 0), (-off, 0), (0, off), (0, -off)):
            px, py = refine_heat(model, arr, gx + dx, gy + dy, device,
                                 augment=False, iters=1)
            errs.append(math.hypot(px - gx, py - gy))
    return float(np.mean(errs))


def train_heat_fold(train_recs, val_recs, oof, epochs, device, fold_id):
    torch.manual_seed(SEED + 200 + fold_id)
    random.seed(SEED + 200 + fold_id)
    model = build_heatnet(device)
    # 自举误差池: 用训练折内 OOF 粗定位误差(画布px)模拟真实粗定位误差分布(含长尾)
    err_pool = []
    for r in train_recs:
        cx, cy, s = oof[r["id"]]
        err_pool.append((cx - r["x"] * s, cy - r["y"] * s))
    tr = DataLoader(HeatRoiDataset(train_recs, oof, True, err_pool), batch_size=12,
                    shuffle=True, pin_memory=True)
    va = DataLoader(HeatRoiDataset(val_recs, oof, False), batch_size=12, shuffle=False)
    head_ids = set(map(id, model.head.parameters()))
    groups = [
        {"params": [p for p in model.stem.parameters()], "lr": BACKBONE_LR},
        {"params": [p for p in model.head.parameters()], "lr": HEAD_LR},
    ]
    assert id(model.head[0].weight) in head_ids
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    total = epochs * len(tr); warm = 2 * len(tr)
    scaler = torch.amp.GradScaler("cuda")
    best = {"rmse": float("inf")}
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for it, (img, heat, gtxy, _rest) in enumerate(tr):
            step = ep * len(tr) + it
            lr = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            opt.param_groups[0]["lr"] = BACKBONE_LR * lr
            opt.param_groups[1]["lr"] = HEAD_LR * lr
            img, heat, gtxy = img.to(device), heat.to(device), gtxy.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                pred = model(img)
                # v4: 热图MSE(形状) + DSNT坐标监督(位置) —— 消除"返回中心"平凡解
                loss = hm_loss(pred, heat) + 4.0 * coord_loss(pred, gtxy)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        # 验证: 完整两阶段流水线误差(含 OOF 粗定位误差) + 合成偏移探测(选型守卫)
        rmse_eu, rmse_kag, _ = evaluate_full(model, val_recs, oof, device)
        shift_err = probe_shift(model, val_recs, device)
        tag = ""
        # v7: 守卫条件 —— 仅当精修真正起效(shift < off*0.9)时才允许以 RMSE 选型,
        # 防止欠训练期的"猜中心"解因 RMSE 偶然更低而被保存
        if shift_err < 54.0 and rmse_eu < best["rmse"]:
            best = {"rmse": rmse_eu, "kag": rmse_kag, "epoch": ep,
                    "state": copy.deepcopy(model.state_dict())}
            tag = "  <- best"
        print(f"  fold{fold_id} ep{ep + 1:03d}/{epochs} loss {tot / max(nb, 1):.4f} "
              f"val_RMSE 欧氏 {rmse_eu:6.2f} | X/Y {rmse_kag:6.2f} "
              f"| shift {shift_err:5.1f}{tag}", flush=True)
    return best


@torch.no_grad()
def evaluate_full(model, records, oof, device, augment=True):
    """完整两阶段评估: OOF粗定位中心 -> 热图精修 -> 原图像素误差
    返回 (rmse_eu, rmse_kag, {id: refined_canvas_xy})"""
    model.eval()
    refs = {}
    for r in records:
        arr, s = get_canvas(r["img_path"])
        cx, cy, _ = oof[r["id"]]
        gx, gy = refine_heat(model, arr, cx, cy, device, augment=augment,
                             iters=ITER_REFINE)
        refs[r["id"]] = (gx / s, gy / s)          # 原图坐标
    ex = np.array([refs[r["id"]][0] - r["x"] for r in records])
    ey = np.array([refs[r["id"]][1] - r["y"] for r in records])
    rmse_eu = float(np.sqrt((ex ** 2 + ey ** 2).mean()))
    rmse_kag = float(np.sqrt(((ex ** 2).mean() + (ey ** 2).mean()) / 2))
    return rmse_eu, rmse_kag, refs


@torch.no_grad()
def refine_heat(model, arr, cx, cy, device, augment=True, iters=1):
    """以(cx,cy)裁ROI -> 热图TTA(原图+水平翻转) -> 画布坐标
    v6: 迭代精修 —— 每轮以当前估计重新裁 ROI, 逐步收敛到黄斑"""
    for _ in range(iters):
        xs = []
        for img, sx, sy in roi_variants(arr, cx, cy, augment):
            t = to_tensor(img).unsqueeze(0).to(device)
            xy = soft_argmax(model(t))[0].cpu().numpy()
            gx = sx(xy[0]); gy = sy(xy[1])
            xs.append((gx, gy))
        cx = float(np.mean([a for a, _ in xs]))
        cy = float(np.mean([b for _, b in xs]))
    return cx, cy


def roi_variants(arr, cx, cy, augment=True):
    """yield (PIL输入图, x映射, y映射): 输入坐标 -> 画布坐标
    v3: 原图 + 翻转 + 平滑 + 锐化 + 对比度 五视角 TTA"""
    x0 = int(min(max(round(cx - ROI / 2), 0), CANVAS - ROI))
    y0 = int(min(max(round(cy - ROI / 2), 0), CANVAS - ROI))
    roi = Image.fromarray(arr[y0:y0 + ROI, x0:x0 + ROI]).resize((S2_INPUT, S2_INPUT), Resample.BILINEAR)
    mx = lambda ix: x0 + ix * ROI / S2_INPUT
    my = lambda iy: y0 + iy * ROI / S2_INPUT
    mfx = lambda ix: x0 + (S2_INPUT - ix) * ROI / S2_INPUT
    yield roi, mx, my
    if augment:
        f = roi.transpose(Transpose.FLIP_LEFT_RIGHT)
        yield f, mfx, my
        sm = roi.filter(_IF.GaussianBlur(radius=1.5))
        yield sm, mx, my
        smf = sm.transpose(Transpose.FLIP_LEFT_RIGHT)
        yield smf, mfx, my
        sh = _IE.Contrast(roi).enhance(1.3)
        yield sh, mx, my
        shf = sh.transpose(Transpose.FLIP_LEFT_RIGHT)
        yield shf, mfx, my


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--folds", type=int, default=KAG)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    torch.backends.cudnn.benchmark = True

    records = load_records()
    # 阶段1 OOF 粗定位
    oof = {}
    splits = fold_splits(len(records))
    print("计算阶段1 OOF 粗定位...")
    for f, val_idx in enumerate(splits):
        m = tvm.resnet34()
        m.fc = nn.Linear(512, 2)
        m.load_state_dict(torch.load(OUT / f"fold{f + 1}.pth", map_location=device))
        m = m.to(device).eval()
        for i in val_idx:
            arr, s = get_canvas(records[i]["img_path"])
            cx, cy = s1_infer([m], Image.fromarray(arr))
            oof[records[i]["id"]] = (float(cx), float(cy), s)
    errs = [math.hypot(oof[r["id"]][0] - r["x"] * oof[r["id"]][2],
                       oof[r["id"]][1] - r["y"] * oof[r["id"]][2]) / oof[r["id"]][2]
            for r in records]
    print(f"  OOF 粗定位: mean {np.mean(errs):.1f}px, RMSE {math.sqrt(np.mean(np.square(errs))):.2f}px")

    fold_models, folds_eu, folds_kag = [], [], []
    t0 = time.time()
    if not args.skip_train:
        for k in range(args.folds):
            val_idx = splits[k]
            tr_idx = np.setdiff1d(np.arange(len(records)), val_idx)
            print(f"\n===== Heat Fold {k + 1}/{args.folds} (train {len(tr_idx)} / val {len(val_idx)}) =====")
            best = train_heat_fold([records[i] for i in tr_idx], [records[i] for i in val_idx],
                                   oof, args.epochs, device, k + 1)
            model = build_heatnet(device)
            model.load_state_dict(best["state"])
            fold_models.append(model)
            folds_eu.append(best["rmse"]); folds_kag.append(best["kag"])
            torch.save(best["state"], OUT / f"hfold{k + 1}.pth")
            print(f"Heat Fold {k + 1} 最佳: ep{best['epoch'] + 1} 欧氏 {best['rmse']:.2f}px | "
                  f"X/Y {best['kag']:.2f}px", flush=True)
        print("\n================ 两阶段(热图)流水线 CV ================")
        print("各折欧氏 RMSE(px): " + ", ".join(f"{r:.2f}" for r in folds_eu))
        print(f"平均欧氏 RMSE = {np.mean(folds_eu):.2f} ± {np.std(folds_eu):.2f} px")
        print(f"平均X/Y口径 RMSE = {np.mean(folds_kag):.2f} px  (目标 < 20)")
        (OUT / "cv_summary_heat.json").write_text(json.dumps({
            "fold_rmse_euclid_px": [round(r, 3) for r in folds_eu],
            "fold_rmse_xy_px": [round(r, 3) for r in folds_kag],
            "mean_euclid": round(float(np.mean(folds_eu)), 3),
            "mean_xy": round(float(np.mean(folds_kag)), 3),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        for k in range(args.folds):
            model = build_heatnet(device)
            model.load_state_dict(torch.load(OUT / f"hfold{k + 1}.pth", map_location=device))
            model.eval()
            fold_models.append(model)

    # ---------- 测试集推理 ----------
    print("\n测试集两阶段推理 (阶段1集成 -> 热图集成)...")
    s1_models = []
    for k in range(KAG):
        m = tvm.resnet34(); m.fc = nn.Linear(512, 2)
        m.load_state_dict(torch.load(OUT / f"fold{k + 1}.pth", map_location=device))
        s1_models.append(m.to(device).eval())

    # 在OOF上搜索 融合权重 alpha: final = alpha*stage2 + (1-alpha)*stage1
    # (用各折热图模型的 OOF 精修结果)
    alphas = np.arange(0.0, 1.01, 0.1)
    blend_scores = []
    per_img_s1 = {r["id"]: (oof[r["id"]][0] / oof[r["id"]][2], oof[r["id"]][1] / oof[r["id"]][2])
                  for r in records}
    per_img_s2 = {}
    for k in range(args.folds):
        val_idx = splits[k]
        _, _, refs = evaluate_full(fold_models[k], [records[i] for i in val_idx], oof, device)
        per_img_s2.update(refs)
    for a in alphas:
        exs, eys = [], []
        for r in records:
            if r["id"] not in per_img_s2:
                continue
            x1, y1 = per_img_s1[r["id"]]
            x2, y2 = per_img_s2[r["id"]]
            x = a * x2 + (1 - a) * x1
            y = a * y2 + (1 - a) * y1
            exs.append(x - r["x"]); eys.append(y - r["y"])
        exs, eys = np.array(exs), np.array(eys)
        kag = math.sqrt(((exs ** 2).mean() + (eys ** 2).mean()) / 2)
        eu = math.sqrt((exs ** 2 + eys ** 2).mean())
        blend_scores.append((a, eu, kag))
        print(f"  alpha={a:.1f}  欧氏 {eu:6.2f}px  X/Y {kag:6.2f}px")
    best_a, best_eu, best_kag = min(blend_scores, key=lambda t: t[2])
    print(f"最优融合权重 alpha={best_a:.1f}: 欧氏 {best_eu:.2f}px, X/Y口径 {best_kag:.2f}px")

    preds = {}
    for p in sorted((DATA / "test").glob("*.jpg")):
        tid = int(p.stem)
        arr, s = get_canvas(str(p))
        cx, cy = s1_infer(s1_models, Image.fromarray(arr))
        # 热图集成精修 (v6 迭代)
        xs, ys = [], []
        for m in fold_models:
            gx, gy = refine_heat(m, arr, cx, cy, device, iters=ITER_REFINE)
            xs.append(gx); ys.append(gy)
        gx2, gy2 = float(np.mean(xs)), float(np.mean(ys))
        x1, y1 = cx / s, cy / s
        x = best_a * gx2 / s + (1 - best_a) * x1
        y = best_a * gy2 / s + (1 - best_a) * y1
        W0, H0 = Image.open(p).size
        preds[tid] = (float(np.clip(x, 0, W0 - 1)), float(np.clip(y, 0, H0 - 1)))

    sub = pd.read_csv(DATA / "sample_submission.csv")
    vals = {}
    for tid, (x, y) in preds.items():
        vals[f"{tid}_Fovea_X"] = round(x, 2)
        vals[f"{tid}_Fovea_Y"] = round(y, 2)
    sub["value"] = sub["ImageID"].map(vals)
    assert sub["value"].notna().all()
    sub.to_csv(ROOT / "submission.csv", index=False)
    print(f"\n已生成提交文件: {ROOT / 'submission.csv'}")
    for tid in sorted(preds):
        x, y = preds[tid]
        print(f"  {tid:04d}.jpg -> Fovea ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    main()
