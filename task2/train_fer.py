# -*- coding: utf-8 -*-
"""FER 情感分类：ResNet18 迁移学习 + 5折交叉验证集成。
类别映射（按文件夹名排序）: Angry=0, Fear=1, Happy=2, Neutral=3, Sad=4, Surprise=5
"""
import os, sys, io, time, random, csv, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import torchvision.transforms as T
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.model_selection import StratifiedKFold
from contextlib import redirect_stdout

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "fer_data", "fer_data", "train")
TEST_DIR = os.path.join(ROOT, "fer_data", "fer_data", "test")
CKPT_DIR = os.path.join(ROOT, "models")
os.makedirs(CKPT_DIR, exist_ok=True)

SEED = 42
N_FOLDS = 5
IMG = 48
BATCH = 256
LR = 3e-4
WD = 5e-4
WARMUP = 1
EPOCHS = 22
LABEL_SMOOTH = 0.1
NUM_CLASSES = 6
# 训练集文件夹 -> 新规则标签（Competition Rules: 0:愤怒 1:恐惧 2:快乐 3:悲伤 4:惊讶 5:中性）
FOLDER_TO_LABEL = {
    "Angry": 0,
    "Fear":  1,
    "Happy": 2,
    "Sad":   3,
    "Surprise": 4,
    "Neutral": 5,
}
LABEL_TO_NAME = {v: k for k, v in FOLDER_TO_LABEL.items()}
FOLDERS = sorted(FOLDER_TO_LABEL.keys())  # 仍按字母序读取文件，保证与原版索引一致
assert FOLDERS == ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"], FOLDERS

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
torch.backends.cudnn.benchmark = True
DEV = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- 数据 ----------------
class FERDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples  # list[(path, label or None)]
        self.tf = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        img = Image.open(p).convert("L")
        x = self.tf(img)
        return x, (y if y is not None else 0), p

def build_index():
    files, labels = [], []
    for c in FOLDERS:
        d = os.path.join(TRAIN_DIR, c)
        for f in os.listdir(d):
            files.append(os.path.join(d, f)); labels.append(FOLDER_TO_LABEL[c])
    return np.array(files), np.array(labels)

def test_files():
    return sorted(os.listdir(TEST_DIR))

MEAN, STD = [0.5077], [0.2550]  # FER 近似统计值

train_tf = T.Compose([
    T.RandomCrop(48, padding=4, padding_mode="edge"),
    T.RandomHorizontalFlip(),
    T.RandomApply([T.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.9, 1.1))], p=0.7),
    T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.2),
    T.RandomAutocontrast(p=0.2),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
    T.RandomErasing(p=0.25, scale=(0.02, 0.12)),
])
eval_tf = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])

def make_model():
    m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    w = m.conv1.weight.data
    m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    m.conv1.weight.data = w.mean(dim=1, keepdim=True)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.fc.in_features, NUM_CLASSES))
    return m.to(DEV)

criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    x = lam * x + (1 - lam) * x[idx]
    return x, y, y[idx], lam

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); tot, correct, loss_sum = 0, 0, 0.0
    for x, y, _ in loader:
        x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
        out = model(x); loss_sum += criterion(out, y).item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item(); tot += y.size(0)
    return loss_sum / tot, correct / tot

def cosine_lr(opt, base_lr, step, total, warmup):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))

def train_one_fold(files, labels, tr_idx, va_idx, fold):
    tr_ds = FERDataset([(files[i], int(labels[i])) for i in tr_idx], train_tf)
    va_ds = FERDataset([(files[i], int(labels[i])) for i in va_idx], eval_tf)
    tr_dl = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=4,
                       pin_memory=True, drop_last=True, persistent_workers=True)
    va_dl = DataLoader(va_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    model = make_model()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps_per = len(tr_dl); total_steps = steps_per * EPOCHS
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(opt, LR, s, total_steps, WARMUP * steps_per) / LR)
    best, bad, best_state = 0.0, 0, None
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); run_loss, n = 0.0, 0
        for x, y, _ in tr_dl:
            x, y = x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)
            x, ya, yb, lam = mixup(x, y)
            out = model(x)
            loss = lam * criterion(out, ya) + (1 - lam) * criterion(out, yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            run_loss += loss.item() * y.size(0); n += y.size(0)
        vl, va = evaluate(model, va_dl)
        tag = ""
        if va > best:
            best = va; bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, os.path.join(CKPT_DIR, f"fold{fold}.pt"))
            tag = " *"
        else:
            bad += 1
        print(f"[fold{fold}] ep{ep+1:02d} loss {run_loss/n:.4f} val_loss {vl:.4f} val_acc {va:.4f} ({time.time()-t0:.0f}s){tag}", flush=True)
        if bad >= 6:
            print(f"[fold{fold}] early stop", flush=True); break
    del model; torch.cuda.empty_cache()
    return best

@torch.no_grad()
def predict_all(folds):
    fs = test_files()
    ds = FERDataset([(os.path.join(TEST_DIR, f), None) for f in fs], eval_tf)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)
    probs_sum = None
    for fold in folds:
        model = make_model()
        model.load_state_dict(torch.load(os.path.join(CKPT_DIR, f"fold{fold}.pt"), map_location=DEV))
        model.eval()
        ps = []
        for x, _, _ in dl:
            x = x.to(DEV, non_blocking=True)
            ps.append(F.softmax(model(x), 1).cpu())
        ps = torch.cat(ps)
        probs_sum = ps if probs_sum is None else probs_sum + ps
        print(f"[pred] fold{fold} done", flush=True)
        del model; torch.cuda.empty_cache()
    preds = probs_sum.argmax(1).numpy()

    # 与提交模板对齐（若顺序一致）
    sub_path = os.path.join(ROOT, "submission.csv")
    order = None
    if os.path.exists(sub_path):
        with open(sub_path, newline="") as f:
            rows = list(csv.reader(f))[1:]
        m = {name: int(em) for name, em in rows}
        if all(n in m for n in fs):
            order = [m[n] for n in fs]
            acc = float(np.mean(np.array(order) == preds))
            print(f"[pred] 与现有 submission.csv 的一致率: {acc:.4f}", flush=True)
    out = os.path.join(ROOT, "submission_new.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["ID", "Emotion"])
        for n, p in zip(fs, preds):
            w.writerow([n, int(p)])
    print(f"[pred] saved -> {out}", flush=True)

if __name__ == "__main__":
    files, labels = build_index()
    print("train size:", len(files), "per-class:", {LABEL_TO_NAME[i]: int((labels == i).sum()) for i in range(NUM_CLASSES)}, flush=True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    accs = []
    for fold, (tr, va) in enumerate(skf.split(files, labels)):
        accs.append(train_one_fold(files, labels, tr, va, fold))
        print(f"[fold{fold}] best val_acc = {accs[-1]:.4f}", flush=True)
    print("CV mean:", float(np.mean(accs)), flush=True)
    predict_all(range(N_FOLDS))
