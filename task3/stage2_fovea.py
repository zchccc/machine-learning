# -*- coding: utf-8 -*-
"""
黄斑中心凹定位 - 阶段2: ROI 精细回归
====================================
依赖阶段1 (train_fovea.py) 已训练好的 output/fold{k}.pth 粗定位模型。

流程:
  1. 用阶段1 各折模型得到 80 张训练图的 OOF(折外)粗定位坐标 (模拟真实粗定位误差)
  2. 以粗定位为中心裁 448x448 ROI(画布坐标, 原生分辨率), 缩放到 320x320 输入,
     回归中心凹相对裁剪中心的偏移 (高分辨率 -> 高精度)
  3. 5 折交叉验证评估两阶段完整流水线 RMSE
  4. 测试集: 阶段1 5模型集成+翻转TTA 粗定位 -> 阶段2 5模型集成+翻转TTA 精修
     -> 更新 submission.csv

用法:
  py stage2_fovea.py                 # 完整训练(5折, 每折50轮)
  py stage2_fovea.py --epochs 2 --folds 2   # 冒烟测试
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
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from train_fovea import (DATA, OUT, ROOT, CANVAS, SEED, TGT_SCALE, UNIT_PX,
                         load_records, get_canvas, photometric, to_tensor,
                         build_model, set_seed, Transpose, Resample)

# 阶段1 配置(与 train_fovea 保持一致, 用于加载粗定位模型)
S1_INPUT = 384
# 阶段2 配置
ROI = 384          # 画布坐标下的裁剪边长
S2_INPUT = 384     # 阶段2 网络输入 (1:1, 无缩放损失)
OFF_SCALE = 1.0    # 偏移目标: o_input_px / (S2_INPUT/2)
JITTER = 20.0      # 训练时在 OOF 粗定位中心上叠加的高斯抖动(px, 画布坐标)
BIG_SHIFT = 0.35   # v8: 训练时大偏移采样半径(占ROI比例), 强制 GT 遍历 ROI
ITER_REFINE = 2    # v15: 4轮迭代精修易漂移, 降为2轮

KAG = 5            # 折数(与阶段1一致以复现相同划分)


def fold_splits(n, k=KAG, seed=SEED):
    idx = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    return [idx[f::k] for f in range(k)]


@torch.no_grad()
def s1_infer(models, canvas_img, device=None):
    """阶段1: 画布 PIL -> 5模型集成+水平翻转TTA -> (x, y) 画布坐标"""
    if device is None:
        device = next(models[0].parameters()).device
    preds = []
    for img in (canvas_img, canvas_img.transpose(Transpose.FLIP_LEFT_RIGHT)):
        t = to_tensor(img.resize((S1_INPUT, S1_INPUT), Resample.BILINEAR)).unsqueeze(0).to(device)
        us = [m(t)[0].cpu().numpy() for m in models]
        preds.append(np.mean(us, axis=0))
    u_o, u_f = preds
    u = np.array([(u_o[0] - u_f[0]) / 2, (u_o[1] + u_f[1]) / 2])
    x_in = u[0] * UNIT_PX + S1_INPUT / 2
    y_in = u[1] * UNIT_PX + S1_INPUT / 2
    return x_in * CANVAS / S1_INPUT, y_in * CANVAS / S1_INPUT


S1_DIR = ROOT / "backup_v14"   # v15: 阶段1权重目录(全局回归最优版, OOF 35.87px)


@torch.no_grad()
def compute_oof_coarse(records, device):
    """用阶段1折模型对每张训练图用未见过的折模型推理 -> OOF 粗定位(画布坐标)"""
    splits = fold_splits(len(records))
    oof = {}
    print("计算阶段1 OOF 粗定位 (模拟真实粗定位误差)...")
    for f, val_idx in enumerate(splits):
        model = build_model(device)
        model.load_state_dict(torch.load(S1_DIR / f"fold{f + 1}.pth", map_location=device))
        model.eval()
        for i in val_idx:
            arr, s = get_canvas(records[i]["img_path"])
            cx, cy = s1_infer([model], Image.fromarray(arr))
            oof[records[i]["id"]] = (float(cx), float(cy), s)
    # 粗定位质量(供参考)
    errs = []
    for r in records:
        cx, cy, s = oof[r["id"]]
        errs.append(math.hypot(cx - r["x"] * s, cy - r["y"] * s) / s)
    errs = np.array(errs)
    print(f"  OOF 粗定位误差(px, 原图): mean {errs.mean():.1f}, median {np.median(errs):.1f}, "
          f"p90 {np.percentile(errs, 90):.1f}, max {errs.max():.1f}")
    rmse = math.sqrt((errs ** 2).mean())
    print(f"  OOF 粗定位欧氏 RMSE = {rmse:.2f} px")
    return oof


class RoiDataset(Dataset):
    """以 coarse 中心裁 ROI, 回归中心凹偏移"""

    def __init__(self, records, oof, train=True):
        self.records = records
        self.oof = oof
        self.train = train

    def __len__(self):
        return len(self.records)

    def _crop(self, arr, cx, cy):
        x0 = int(min(max(round(cx - ROI / 2), 0), CANVAS - ROI))
        y0 = int(min(max(round(cy - ROI / 2), 0), CANVAS - ROI))
        return x0, y0

    def __getitem__(self, k):
        r = self.records[k]
        arr, s = get_canvas(r["img_path"])
        cx, cy, _ = self.oof[r["id"]]
        if self.train:
            cx += random.gauss(0, JITTER)
            cy += random.gauss(0, JITTER)
            # v8: 大偏移采样 —— 50% 概率把中心偏移到 ROI 内任意位置,
            # GT 相对中心遍布全 ROI, 强制模型在 ROI 内真正定位(而非猜中心)
            if random.random() < 0.5:
                ang = random.uniform(0, 2 * math.pi)
                rad = ROI * BIG_SHIFT * math.sqrt(random.uniform(0.1, 1.0))
                cx += math.cos(ang) * rad
                cy += math.sin(ang) * rad
        x0, y0 = self._crop(arr, cx, cy)
        roi = Image.fromarray(arr[y0:y0 + ROI, x0:x0 + ROI])
        gx, gy = r["x"] * s - x0, r["y"] * s - y0      # GT 相对裁剪左上角 (ROI 像素)
        flip = self.train and random.random() < 0.5
        if flip:
            roi = roi.transpose(Transpose.FLIP_LEFT_RIGHT)
            gx = ROI - gx
        if self.train:
            roi = photometric(roi)
        inp = roi.resize((S2_INPUT, S2_INPUT), Resample.BILINEAR)
        img = to_tensor(inp)
        # 归一化偏移目标(相对裁剪中心, 输入像素)
        ox = (gx - ROI / 2) * S2_INPUT / ROI
        oy = (gy - ROI / 2) * S2_INPUT / ROI
        tgt = torch.tensor([ox / (S2_INPUT / 2) * OFF_SCALE,
                            oy / (S2_INPUT / 2) * OFF_SCALE], dtype=torch.float32)
        factor = (ROI / S2_INPUT) / s   # 输入像素 -> 原图像素
        return img, tgt, torch.tensor([factor], dtype=torch.float32), \
            torch.tensor([x0, y0], dtype=torch.float32), torch.tensor([flip], dtype=torch.float32)


def offset_pixel_rmse(pred, tgt, factors):
    d = (pred - tgt) / OFF_SCALE * (S2_INPUT / 2) * factors
    ex, ey = d[:, 0], d[:, 1]
    rmse_eu = float(torch.sqrt((ex ** 2 + ey ** 2).mean()))
    rmse_kag = float(torch.sqrt(((ex ** 2).mean() + (ey ** 2).mean()) / 2))
    return rmse_eu, rmse_kag


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ts, fs = [], [], []
    for img, tgt, f, _, _ in loader:
        out = model(img.to(device, non_blocking=True))
        ps.append(out.cpu()); ts.append(tgt); fs.append(f)
    return offset_pixel_rmse(torch.cat(ps), torch.cat(ts), torch.cat(fs))


@torch.no_grad()
def evaluate_pipeline(model, val_recs, oof, device):
    """v8: 完整两阶段评估 —— OOF粗定位 -> 单模型迭代精修 -> 原图像素误差
    (验证集用单模型, 测试集用5模型集成, 略保守但相对排序一致)"""
    model.eval()
    exs, eys = [], []
    for r in val_recs:
        arr, s = get_canvas(r["img_path"])
        cx, cy, _ = oof[r["id"]]
        gx, gy = refine_point([model], arr, cx, cy, device)
        exs.append(gx / s - r["x"]); eys.append(gy / s - r["y"])
    exs, eys = np.array(exs), np.array(eys)
    rmse_eu = float(np.sqrt((exs ** 2 + eys ** 2).mean()))
    rmse_kag = float(np.sqrt(((exs ** 2).mean() + (eys ** 2).mean()) / 2))
    return rmse_eu, rmse_kag


def train_stage2_fold(train_recs, val_recs, oof, epochs, device, fold_id):
    torch.manual_seed(SEED + 100 + fold_id)
    random.seed(SEED + 100 + fold_id)
    model = build_model(device)
    tr = DataLoader(RoiDataset(train_recs, oof, True), batch_size=12, shuffle=True, pin_memory=True)
    va = DataLoader(RoiDataset(val_recs, oof, False), batch_size=12, shuffle=False)
    fc = list(model.fc.parameters()); ids = {id(p) for p in fc}
    base = [p for p in model.parameters() if id(p) not in ids]
    opt = torch.optim.AdamW([{"params": base, "lr": 3e-4}, {"params": fc, "lr": 1.5e-3}],
                            weight_decay=1e-4)
    total = epochs * len(tr); warm = 2 * len(tr)
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.SmoothL1Loss()
    best = {"rmse": float("inf")}
    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for it, (img, tgt, *_ ) in enumerate(tr):
            step = ep * len(tr) + it
            lr = min(1.0, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * step / total))
            opt.param_groups[0]["lr"] = 3e-4 * lr
            opt.param_groups[1]["lr"] = 1.5e-3 * lr
            img, tgt = img.to(device), tgt.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = crit(model(img), tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += loss.item(); nb += 1
        # v8: 完整两阶段(迭代精修)流水线误差, 选型与测试推理口径一致
        rmse_eu, rmse_kag = evaluate_pipeline(model, val_recs, oof, device)
        tag = ""
        if rmse_eu < best["rmse"]:
            best = {"rmse": rmse_eu, "kag": rmse_kag, "epoch": ep,
                    "state": copy.deepcopy(model.state_dict())}
            tag = "  <- best"
        print(f"  fold{fold_id} ep{ep + 1:03d}/{epochs} loss {tot / max(nb, 1):.4f} "
              f"val_RMSE(欧氏) {rmse_eu:6.2f}px  (X/Y平均 {rmse_kag:6.2f}px){tag}", flush=True)
    return best


@torch.no_grad()
def refine_point(models, arr, cx, cy, device, iters=ITER_REFINE):
    """阶段2精修: 以(cx,cy)为中心裁ROI, 多模型+翻转TTA -> 画布坐标
    v8: 迭代精修 —— 每轮以当前估计重新裁 ROI 中心, 逐步收敛"""
    for _ in range(iters):
        x0 = int(min(max(round(cx - ROI / 2), 0), CANVAS - ROI))
        y0 = int(min(max(round(cy - ROI / 2), 0), CANVAS - ROI))
        roi = Image.fromarray(arr[y0:y0 + ROI, x0:x0 + ROI])
        offs = []
        for img in (roi, roi.transpose(Transpose.FLIP_LEFT_RIGHT)):
            t = to_tensor(img.resize((S2_INPUT, S2_INPUT), Resample.BILINEAR)).unsqueeze(0).to(device)
            us = [m(t)[0].cpu().numpy() for m in models]
            offs.append(np.mean(us, axis=0))
        o, of = offs
        ox = (o[0] - of[0]) / 2 * OFF_SCALE * (S2_INPUT / 2)      # 输入像素
        oy = (o[1] + of[1]) / 2 * OFF_SCALE * (S2_INPUT / 2)
        cx = x0 + ROI / 2 + ox * ROI / S2_INPUT
        cy = y0 + ROI / 2 + oy * ROI / S2_INPUT
    return float(cx), float(cy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--folds", type=int, default=KAG)
    ap.add_argument("--skip-train", action="store_true", help="只用已保存的阶段2权重做推理")
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    torch.backends.cudnn.benchmark = True

    records = load_records()
    oof = compute_oof_coarse(records, device)

    splits = fold_splits(len(records))
    fold_models, folds_rmse = [], []
    t0 = time.time()
    if not args.skip_train:
        for k in range(args.folds):
            val_idx = splits[k]
            tr_idx = np.setdiff1d(np.arange(len(records)), val_idx)
            print(f"\n===== Stage2 Fold {k + 1}/{args.folds} (train {len(tr_idx)} / val {len(val_idx)}) =====")
            best = train_stage2_fold([records[i] for i in tr_idx], [records[i] for i in val_idx],
                                     oof, args.epochs, device, k + 1)
            model = build_model(device)
            model.load_state_dict(best["state"])
            fold_models.append(model)
            folds_rmse.append(best["rmse"])
            torch.save(best["state"], OUT / f"s2fold{k + 1}.pth")
            print(f"Stage2 Fold {k + 1} 最佳: ep{best['epoch'] + 1} RMSE(欧氏)={best['rmse']:.2f}px "
                  f"(X/Y平均={best['kag']:.2f}px)", flush=True)
        mean_r, std_r = float(np.mean(folds_rmse)), float(np.std(folds_rmse))
        print("\n================ 两阶段流水线交叉验证 ================")
        print("各折 RMSE(px): " + ", ".join(f"{r:.2f}" for r in folds_rmse))
        print(f"平均 RMSE = {mean_r:.2f} ± {std_r:.2f} px   (目标 < 20 px: {'达成' if mean_r < 20 else '未达成'})")
        print(f"耗时 {time.time() - t0:.0f}s")
        (OUT / "cv_summary_stage2.json").write_text(json.dumps({
            "fold_rmse_euclid_px": [round(r, 3) for r in folds_rmse],
            "mean": round(mean_r, 3), "std": round(std_r, 3),
            "epochs": args.epochs, "folds": args.folds,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        for k in range(args.folds):
            model = build_model(device)
            model.load_state_dict(torch.load(OUT / f"s2fold{k + 1}.pth", map_location=device))
            model.eval()
            fold_models.append(model)

    # ---------- 测试集两阶段推理 ----------
    print("\n测试集两阶段推理...")
    s1_models = []
    for k in range(KAG):
        m = build_model(device)
        m.load_state_dict(torch.load(S1_DIR / f"fold{k + 1}.pth", map_location=device))
        m.eval()
        s1_models.append(m)
    for m in fold_models:
        m.eval()

    preds = {}
    for p in sorted((DATA / "test").glob("*.jpg")):
        tid = int(p.stem)
        arr, s = get_canvas(str(p))
        cx, cy = s1_infer(s1_models, Image.fromarray(arr))          # 阶段1 粗定位
        gx, gy = refine_point(fold_models, arr, cx, cy, device)     # 阶段2 精修
        W0, H0 = Image.open(p).size
        preds[tid] = (float(np.clip(gx / s, 0, W0 - 1)), float(np.clip(gy / s, 0, H0 - 1)))

    sub = pd.read_csv(DATA / "sample_submission.csv")
    vals = {}
    for tid, (x, y) in preds.items():
        vals[f"{tid}_Fovea_X"] = round(x, 2)
        vals[f"{tid}_Fovea_Y"] = round(y, 2)
    sub["value"] = sub["ImageID"].map(vals)
    assert sub["value"].notna().all()
    sub.to_csv(ROOT / "submission.csv", index=False)
    print(f"已更新提交文件: {ROOT / 'submission.csv'}")
    for tid in sorted(preds):
        x, y = preds[tid]
        print(f"  {tid:04d}.jpg -> Fovea ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    main()
