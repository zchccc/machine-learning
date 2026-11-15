# -*- coding: utf-8 -*-
"""
黄斑中心凹(Fovea)自动定位程序
================================
数据(相对本文件):
  detection/train/                     80 张训练眼底图 (0001-0080.jpg)
  detection/fovea_localization_train_GT.csv  训练集中心凹坐标 GT (原图像素坐标)
  detection/test/                      20 张测试眼底图 (0081-0100.jpg)
  detection/sample_submission.csv      提交格式

方法:
  1. 预处理: 等比缩放+填充到 1024x1024 画布; 取绿通道复制为 3 通道(眼底血管/黄斑在绿通道最清晰)
  2. 模型: ImageNet 预训练 ResNet-34, 回归中心凹的归一化坐标
  3. 训练: 随机水平翻转/旋转/平移/缩放/亮度对比度gamma抖动 + smooth L1 + AdamW + 余弦退火 + AMP
  4. 评估: 5 折交叉验证, 指标为原图像素坐标系下的欧氏距离 RMSE (目标 < 20 px)
  5. 推理: 5 个折模型集成 + 水平翻转 TTA, 生成 submission.csv

用法:
  py train_fovea.py            # 完整训练(5折, 每折60轮)
  py train_fovea.py --epochs 2 --folds 2   # 快速冒烟测试
"""
import argparse
import copy
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm

# ---------------- 全局配置 ----------------
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "detection"
OUT = ROOT / "output"
CANVAS = 1024          # 预处理统一画布尺寸
INPUT = 384            # 网络输入尺寸 (v9 大模型实验失败, 回归最优配置 384)
def set_input(n):
    """v16: 支持命令行修改输入尺寸(全局量由模块内函数引用)"""
    global INPUT, UNIT_PX
    INPUT = int(n)
    UNIT_PX = (INPUT / 2) / TGT_SCALE
SEED = 42

# 坐标回归目标: u = (x_in - INPUT/2) / (INPUT/2) * TGT_SCALE
TGT_SCALE = 4.0        # 目标值放大倍数, 使回归范围约 [-2, 3], 优化更稳定
UNIT_PX = (INPUT / 2) / TGT_SCALE   # 1 个目标单位对应的输入像素数 (=48)
BACKBONE = "resnet34"  # v10: 恢复 resnet34 (80 图小数据最优; resnet50@512 过拟合至 49px)
K = 5                  # v11: 恢复 5 折 (10 折实验劣化: 单折验证集仅 8 图, 选型噪声大)

IMNET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMNET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# Pillow 版本兼容
Resample = getattr(Image, "Resampling", Image)
Transpose = getattr(Image, "Transpose", Image)
AFFINE_T = getattr(getattr(Image, "Transform", Image), "AFFINE", 0)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------- 数据准备 ----------------
def load_records():
    gt = pd.read_csv(DATA / "fovea_localization_train_GT.csv")
    records = []
    for _, r in gt.iterrows():
        idx = int(r["data"])
        records.append({
            "img_path": str(DATA / "train" / f"{idx:04d}.jpg"),
            "id": idx,
            "x": float(r["Fovea_X"]),
            "y": float(r["Fovea_Y"]),
        })
    return records


_canvas_cache = {}   # 路径 -> (np.uint8 canvas, 缩放系数 s)


def get_canvas(path):
    """解码一次并缓存: 等比缩放(长边->CANVAS)后贴到 CANVASxCANVAS 黑底画布"""
    if path in _canvas_cache:
        return _canvas_cache[path]
    img = Image.open(path).convert("RGB")
    W, H = img.size
    s = CANVAS / max(W, H)
    img = img.resize((max(1, round(W * s)), max(1, round(H * s))), Resample.BILINEAR)
    canvas = Image.new("RGB", (CANVAS, CANVAS))
    canvas.paste(img, (0, 0))
    arr = np.asarray(canvas, dtype=np.uint8)
    _canvas_cache[path] = (arr, s)
    return arr, s


def photometric(img):
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
    g = random.uniform(0.8, 1.25)
    if abs(g - 1.0) > 1e-3:
        lut = [round(((i / 255.0) ** (1.0 / g)) * 255) for i in range(256)]
        img = img.point(lut * 3)
    return img


def rand_crop_box():
    """等效 torchvision RandomResizedCrop: scale (0.8,1.0), ratio (0.85,1.18)"""
    area = CANVAS * CANVAS
    for _ in range(10):
        ta = area * random.uniform(0.8, 1.0)
        ar = math.exp(random.uniform(math.log(0.85), math.log(1.18)))
        w = int(round(math.sqrt(ta * ar)))
        h = int(round(math.sqrt(ta / ar)))
        if 0 < w <= CANVAS and 0 < h <= CANVAS:
            i = random.randint(0, CANVAS - h)
            j = random.randint(0, CANVAS - w)
            return i, j, h, w
    return 0, 0, CANVAS, CANVAS


def to_tensor(canvas_img):
    """绿通道 -> 3通道 -> ImageNet 归一化"""
    arr = np.asarray(canvas_img, dtype=np.uint8)
    g = arr[..., 1]
    t = torch.from_numpy(np.stack([g, g, g], axis=-1)).permute(2, 0, 1).float().div_(255.0)
    return (t - IMNET_MEAN) / IMNET_STD


class FoveaDataset(Dataset):
    def __init__(self, records, train=True):
        self.records = records
        self.train = train

    def __len__(self):
        return len(self.records)

    def __getitem__(self, k):
        r = self.records[k]
        arr, s = get_canvas(r["img_path"])
        canvas = Image.fromarray(arr)
        x, y = r["x"] * s, r["y"] * s     # GT 转到画布坐标
        crop_w = CANVAS                    # 记录裁剪宽度, 用于坐标换算

        if self.train:
            canvas = photometric(canvas)
            # 随机水平翻转 (黄斑左右眼对称)
            if random.random() < 0.5:
                canvas = canvas.transpose(Transpose.FLIP_LEFT_RIGHT)
                x = CANVAS - x
            # 随机旋转/缩放/平移(绕画布中心的仿射)
            th = math.radians(random.uniform(-12, 12))
            sc = random.uniform(0.9, 1.1)
            tx, ty = random.uniform(-20, 20), random.uniform(-20, 20)
            cos, sin = math.cos(th) * sc, math.sin(th) * sc
            Fm = np.array([
                [cos, -sin, 512 - 512 * cos + 512 * sin + tx],
                [sin, cos, 512 - 512 * sin - 512 * cos + ty],
                [0.0, 0.0, 1.0],
            ])
            p = Fm @ np.array([x, y, 1.0])
            if 40 <= p[0] <= CANVAS - 40 and 40 <= p[1] <= CANVAS - 40:
                inv = np.linalg.inv(Fm)
                a = tuple(inv[0, :2].tolist() + [inv[0, 2]] + inv[1, :2].tolist() + [inv[1, 2]])
                canvas = canvas.transform((CANVAS, CANVAS), AFFINE_T, a, resample=Resample.BILINEAR)
                x, y = float(p[0]), float(p[1])
            # 随机裁剪并 resize 到网络输入
            i, j, h, w = rand_crop_box()
            canvas = canvas.crop((j, i, j + w, i + h)).resize((INPUT, INPUT), Resample.BILINEAR)
            x = (x - j) * INPUT / w
            y = (y - i) * INPUT / h
            crop_w = w
        else:
            canvas = canvas.resize((INPUT, INPUT), Resample.BILINEAR)
            x = x * INPUT / CANVAS
            y = y * INPUT / CANVAS

        img = to_tensor(canvas)
        tgt = torch.tensor([(x - INPUT / 2) / UNIT_PX, (y - INPUT / 2) / UNIT_PX], dtype=torch.float32)
        # orig_px_per_input_px: 输入图 1 像素对应原图多少像素
        factor = (crop_w / INPUT) / s
        return img, tgt, torch.tensor([factor], dtype=torch.float32)


# ---------------- 模型 ----------------
def build_model(device):
    if BACKBONE == "resnet50":
        try:
            m = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT)
        except Exception:
            print("  [警告] 无法下载预训练权重, 使用随机初始化")
            m = tvm.resnet50(weights=None)
        m.fc = nn.Linear(2048, 2)
    else:
        try:
            m = tvm.resnet34(weights=tvm.ResNet34_Weights.DEFAULT)   # 需联网下载一次
        except Exception:
            print("  [警告] 无法下载预训练权重, 使用随机初始化")
            m = tvm.resnet34(weights=None)
        m.fc = nn.Linear(512, 2)
    nn.init.normal_(m.fc.weight, std=0.01)
    nn.init.zeros_(m.fc.bias)
    return m.to(device)


def pixel_rmse(pred, tgt, factors):
    """pred/tgt: (N,2) 回归目标单位; factors: (N,1) 输入像素->原图像素
    返回欧氏距离 RMSE 与 Kaggle 风格(分别对X,Y再平均) RMSE, 单位: 原图像素"""
    d = (pred - tgt) * UNIT_PX * factors          # (N,2) 原图像素误差
    ex, ey = d[:, 0], d[:, 1]
    rmse_eu = float(torch.sqrt((ex ** 2 + ey ** 2).mean()))
    rmse_kag = float(torch.sqrt(((ex ** 2).mean() + (ey ** 2).mean()) / 2))
    return rmse_eu, rmse_kag


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ps, ts, fs = [], [], []
    for img, tgt, f in loader:
        img = img.to(device, non_blocking=True)
        out = model(img)
        ps.append(out.cpu())
        ts.append(tgt)
        fs.append(f)
    return pixel_rmse(torch.cat(ps), torch.cat(ts), torch.cat(fs))


def train_one_fold(train_recs, val_recs, epochs, device, fold_id, seed,
                   mixup_alpha=0.0, ema_decay=0.0):
    torch.manual_seed(seed + fold_id)
    random.seed(seed + fold_id)
    model = build_model(device)
    tr_loader = DataLoader(FoveaDataset(train_recs, True), batch_size=8, shuffle=True,
                           num_workers=0, drop_last=False, pin_memory=True)
    va_loader = DataLoader(FoveaDataset(val_recs, False), batch_size=8, shuffle=False, num_workers=0)

    fc_params = list(model.fc.parameters())
    fc_ids = {id(p) for p in fc_params}
    base_params = [p for p in model.parameters() if id(p) not in fc_ids]
    opt = torch.optim.AdamW([
        {"params": base_params, "lr": 3e-4},
        {"params": fc_params, "lr": 1.5e-3},
    ], weight_decay=1e-4)

    ema = None
    if ema_decay > 0:
        ema = {k: v.detach().clone().float() for k, v in model.state_dict().items()}

    steps_per_ep = len(tr_loader)
    total_steps = epochs * steps_per_ep
    warmup = 2 * steps_per_ep
    scaler = torch.amp.GradScaler("cuda")
    crit = nn.SmoothL1Loss()
    best = {"rmse": float("inf"), "epoch": -1, "state": None}
    top3 = []   # [(rmse, epoch, state), ...] 保留前3好epoch, 最终做权重平均(soup)

    for ep in range(epochs):
        model.train()
        tot, nb = 0.0, 0
        for it, (img, tgt, _) in enumerate(tr_loader):
            step = ep * steps_per_ep + it
            lr = min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * step / total_steps))
            opt.param_groups[0]["lr"] = 3e-4 * lr
            opt.param_groups[1]["lr"] = 1.5e-3 * lr

            img, tgt = img.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            if mixup_alpha > 0 and img.size(0) > 1:
                # mixup: 图与坐标同步线性插值(回归任务天然适配)
                lam = float(np.random.beta(mixup_alpha, mixup_alpha))
                lam = max(lam, 1 - lam)   # 保持主样本权重>0.5, 坐标偏移不过大
                perm = torch.randperm(img.size(0), device=img.device)
                img = lam * img + (1 - lam) * img[perm]
                tgt = lam * tgt + (1 - lam) * tgt[perm]
            with torch.amp.autocast("cuda", dtype=torch.float16):
                out = model(img)
                loss = crit(out, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            nb += 1
            if ema is not None:
                with torch.no_grad():
                    for k, v in model.state_dict().items():
                        if v.dtype.is_floating_point:
                            ema[k].mul_(ema_decay).add_(v.float(), alpha=1 - ema_decay)

        # 评估原始权重
        rmse_eu, rmse_kag = evaluate(model, va_loader, device)
        tag = ""
        if rmse_eu < best["rmse"]:
            best = {"rmse": rmse_eu, "kag": rmse_kag, "epoch": ep,
                    "state": copy.deepcopy(model.state_dict())}
            tag = "  <- best"
        top3.append((rmse_eu, ep, copy.deepcopy(model.state_dict())))
        top3.sort(key=lambda t: t[0])
        del top3[3:]
        print(f"  fold{fold_id} ep{ep + 1:03d}/{epochs} loss {tot / max(nb, 1):.4f} "
              f"val_RMSE(欧氏) {rmse_eu:6.2f}px  (X/Y平均 {rmse_kag:6.2f}px){tag}", flush=True)

    if len(top3) >= 2:   # soup: 前2-3好epoch权重平均, 若更优则替换best
        soup = {k: sum(t[2][k].float() for t in top3) / len(top3) for k in top3[0][2]}
        ref_dtype = best["state"][next(iter(best["state"]))].dtype
        soup = {k: v.to(ref_dtype) for k, v in soup.items()}
        model.load_state_dict(soup)
        s_eu, s_kag = evaluate(model, va_loader, device)
        print(f"  fold{fold_id} soup(top{len(top3)}) RMSE(欧氏) {s_eu:6.2f}px  (X/Y平均 {s_kag:6.2f}px)")
        if s_eu < best["rmse"]:
            best = {"rmse": s_eu, "kag": s_kag, "epoch": -1, "state": soup}
            print(f"  fold{fold_id} soup 更优, 采用")
    return best


@torch.no_grad()
def predict_test(models, device):
    """5 折模型集成 + 水平翻转 TTA, 返回 {测试图id: (x, y)} 原图像素坐标"""
    test_paths = sorted((DATA / "test").glob("*.jpg"))
    preds = {}
    for p in test_paths:
        tid = int(p.stem)
        arr, s = get_canvas(str(p))
        canvas = Image.fromarray(arr)
        factor = (CANVAS / INPUT) / s    # 输入像素 -> 原图像素
        acc = []
        for img in (canvas, canvas.transpose(Transpose.FLIP_LEFT_RIGHT)):
            x_in = img.resize((INPUT, INPUT), Resample.BILINEAR)
            t = to_tensor(x_in).unsqueeze(0).to(device)
            us = []
            for m in models:
                m.eval()
                out = m(t)[0].cpu().numpy()
                us.append(out)
            u = np.mean(us, axis=0)
            acc.append(u)
        u_orig, u_flip = acc[0], acc[1]
        # 翻转图的 x 目标与原图相反
        u_x = (u_orig[0] + (-u_flip[0])) / 2
        u_y = (u_orig[1] + u_flip[1]) / 2
        x_in = u_x * UNIT_PX + INPUT / 2
        y_in = u_y * UNIT_PX + INPUT / 2
        W0, H0 = Image.open(p).size
        preds[tid] = (float(np.clip(x_in * factor, 0, W0 - 1)),
                      float(np.clip(y_in * factor, 0, H0 - 1)))
    return preds


def make_submission(preds):
    sub = pd.read_csv(DATA / "sample_submission.csv")
    vals = {}
    for tid, (x, y) in preds.items():
        vals[f"{tid}_Fovea_X"] = round(x, 2)
        vals[f"{tid}_Fovea_Y"] = round(y, 2)
    sub["value"] = sub["ImageID"].map(vals)
    assert sub["value"].notna().all(), "存在未预测的提交行"
    out_csv = ROOT / "submission.csv"
    sub.to_csv(out_csv, index=False)
    return out_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-fold", type=int, default=0,
                    help="只训练指定折(1-based), 用于快速实验; 0=全部")
    ap.add_argument("--seed", type=int, default=SEED, help="训练种子")
    ap.add_argument("--out-dir", type=str, default="", help="输出目录(默认 output), 不同种子分目录")
    ap.add_argument("--skip-infer", action="store_true", help="跳过测试集推理")
    ap.add_argument("--resume", action="store_true",
                    help="跳过已有 fold{k}_meta.json 的折, 断点续训")
    ap.add_argument("--input", type=int, default=384, help="网络输入尺寸 (v16 实验)")
    ap.add_argument("--mixup", type=float, default=0.0, help="mixup beta 参数 (0=关闭)")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA 衰减系数 (0=关闭)")
    args = ap.parse_args()

    # 行缓冲: 重定向到文件时也能实时看到日志
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

    set_seed(args.seed)
    set_input(args.input)
    global OUT
    if args.out_dir:
        OUT = ROOT / args.out_dir
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == 'cuda' else ""))
    torch.backends.cudnn.benchmark = True

    records = load_records()
    print(f"训练样本: {len(records)} 张, 测试图: {len(list((DATA / 'test').glob('*.jpg')))} 张")

    # 基线参考: 直接预测训练集均值坐标的 RMSE
    mx, my = np.mean([r["x"] for r in records]), np.mean([r["y"] for r in records])
    base = math.sqrt(np.mean([(r["x"] - mx) ** 2 + (r["y"] - my) ** 2 for r in records]))
    print(f"基线(预测均值坐标) RMSE = {base:.1f} px\n")

    # K 折交叉验证
    K = args.folds
    idx = np.arange(len(records))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    folds_rmse, fold_models = [], []
    t0 = time.time()
    for k in range(K):
        if args.only_fold and (k + 1) != args.only_fold:
            continue
        val_idx = idx[k::K]
        tr_idx = np.setdiff1d(idx, val_idx)
        ckpt = OUT / f"fold{k + 1}.pth"
        meta = OUT / f"fold{k + 1}_meta.json"
        model = build_model(device)
        if args.resume and ckpt.exists() and meta.exists():
            info = json.loads(meta.read_text(encoding="utf-8"))
            model.load_state_dict(torch.load(ckpt, map_location=device))
            fold_models.append(model)
            folds_rmse.append(info["rmse"])
            print(f"Fold {k + 1}: 已有权重, 跳过 (RMSE={info['rmse']:.2f}px)")
            continue
        print(f"\n===== Fold {k + 1}/{K} (train {len(tr_idx)} / val {len(val_idx)}) =====")
        best = train_one_fold([records[i] for i in tr_idx], [records[i] for i in val_idx],
                              args.epochs, device, k + 1, args.seed,
                              mixup_alpha=args.mixup, ema_decay=args.ema)
        model.load_state_dict(best["state"])
        fold_models.append(model)
        folds_rmse.append(best["rmse"])
        torch.save(best["state"], ckpt)
        meta.write_text(json.dumps({"rmse": best["rmse"], "kag": best["kag"],
                                    "epoch": best["epoch"]}, ensure_ascii=False), encoding="utf-8")
        print(f"Fold {k + 1} 最佳: ep{best['epoch'] + 1} RMSE(欧氏)={best['rmse']:.2f}px "
              f"(X/Y平均={best['kag']:.2f}px)")

    mean_r = float(np.mean(folds_rmse))
    std_r = float(np.std(folds_rmse))
    print("\n================ 交叉验证结果 ================")
    print("各折 RMSE(px): " + ", ".join(f"{r:.2f}" for r in folds_rmse))
    print(f"平均 RMSE = {mean_r:.2f} ± {std_r:.2f} px   (目标 < 20 px: {'达成' if mean_r < 20 else '未达成'})")
    print(f"总耗时 {time.time() - t0:.0f}s")

    (OUT / "cv_summary.json").write_text(json.dumps({
        "fold_rmse_euclid_px": [round(r, 3) for r in folds_rmse],
        "mean": round(mean_r, 3), "std": round(std_r, 3),
        "baseline_mean_rmse_px": round(base, 2),
        "epochs": args.epochs, "folds": K,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # 测试集推理(集成 + TTA)并生成提交文件
    if args.skip_infer:
        print("\n--skip-infer: 跳过测试集推理")
        return
    print("\n对测试集推理 (集成 %d 个折模型 + 翻转TTA)..." % len(fold_models))
    preds = predict_test(fold_models, device)
    out_csv = make_submission(preds)
    print(f"已生成提交文件: {out_csv}")
    for tid in sorted(preds):
        x, y = preds[tid]
        print(f"  {tid:04d}.jpg -> Fovea ({x:.1f}, {y:.1f})")


if __name__ == "__main__":
    main()
