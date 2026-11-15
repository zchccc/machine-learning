#!/usr/bin/env python3
"""Train, infer and create a submission for the fundus vessel task.

Examples (run from the ``segmentation`` directory)::

    python vessel_segmentation.py train --epochs 120 --out runs/best.pt
    python vessel_segmentation.py predict --checkpoint runs/best.pt

The predicted PNG/JPEG masks use the competition convention: vessel=0,
background=255.  The ``predict`` command also writes ``submission.csv``.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def numeric_files(directory: Path) -> list[Path]:
    files = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files, key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**9, p.name))


def resize_image(array: np.ndarray, size: int, nearest: bool = False) -> np.ndarray:
    mode = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    image = Image.fromarray(array)
    return np.asarray(image.resize((size, size), mode))


class FundusDataset(Dataset):
    def __init__(self, image_paths: list[Path], label_paths: dict[str, Path] | None,
                 size: int, augment: bool = False):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.size = size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = np.asarray(Image.open(path).convert("RGB"))
        image = resize_image(image, self.size).astype(np.float32) / 255.0
        label = None
        if self.label_paths is not None:
            raw = np.asarray(Image.open(self.label_paths[path.stem]).convert("L"))
            raw = resize_image(raw, self.size, nearest=True)
            # The source labels are black vessels on a white background.
            label = (raw < 128).astype(np.float32)

        if self.augment:
            if random.random() < 0.5:
                image = image[:, ::-1].copy()
                if label is not None:
                    label = label[:, ::-1].copy()
            if random.random() < 0.5:
                image = image[::-1, :].copy()
                if label is not None:
                    label = label[::-1, :].copy()
            if random.random() < 0.5:
                k = random.randrange(4)
                image = np.rot90(image, k).copy()
                if label is not None:
                    label = np.rot90(label, k).copy()
            # Mild color perturbation retains vessel contrast while reducing overfit.
            if random.random() < 0.5:
                pil = Image.fromarray(np.uint8(np.clip(image * 255, 0, 255)))
                pil = ImageEnhance.Brightness(pil).enhance(random.uniform(0.85, 1.15))
                pil = ImageEnhance.Contrast(pil).enhance(random.uniform(0.85, 1.20))
                image = np.asarray(pil).astype(np.float32) / 255.0

        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if label is None:
            return image, path.name
        return image, torch.from_numpy(label[None]).float()


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, base: int = 24):
        super().__init__()
        self.down1 = DoubleConv(3, base)
        self.down2 = DoubleConv(base, base * 2)
        self.down3 = DoubleConv(base * 2, base * 4)
        self.down4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.bridge = DoubleConv(base * 8, base * 16)
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, 2)
        self.conv4 = DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, 2)
        self.conv3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, 2)
        self.conv2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, 2)
        self.conv1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    @staticmethod
    def cat(up, skip):
        # Inputs are square and divisible by 16; this also handles odd sizes safely.
        dh = skip.shape[-2] - up.shape[-2]
        dw = skip.shape[-1] - up.shape[-1]
        if dh or dw:
            up = nn.functional.pad(up, (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2))
        return torch.cat((skip, up), dim=1)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))
        x = self.bridge(self.pool(x4))
        x = self.conv4(self.cat(self.up4(x), x4))
        x = self.conv3(self.cat(self.up3(x), x3))
        x = self.conv2(self.cat(self.up2(x), x2))
        x = self.conv1(self.cat(self.up1(x), x1))
        return self.head(x)


def dice_score(probability: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probability = probability.flatten(1)
    target = target.flatten(1)
    return ((2 * (probability * target).sum(1) + eps) /
            (probability.sum(1) + target.sum(1) + eps)).mean()


class DiceBCELoss(nn.Module):
    def forward(self, logits, target):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        dice_loss = 1 - dice_score(torch.sigmoid(logits), target)
        return 0.5 * bce + 0.5 * dice_loss


def run_epoch(model, loader, optimizer, device, criterion, train: bool):
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    count = 0
    for image, target in loader:
        image, target = image.to(device), target.to(device)
        with torch.set_grad_enabled(train):
            logits = model(image)
            loss = criterion(logits, target)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        batch = image.shape[0]
        total_loss += loss.item() * batch
        total_dice += dice_score((torch.sigmoid(logits) > 0.5).float(), target).item() * batch
        count += batch
    return total_loss / max(count, 1), total_dice / max(count, 1)


def train(args):
    seed_everything(args.seed)
    root = Path(args.data)
    image_dir, label_dir = root / "train" / "image", root / "train" / "label"
    images = numeric_files(image_dir)
    labels = {p.stem: p for p in numeric_files(label_dir)}
    if not images or set(p.stem for p in images) != set(labels):
        raise RuntimeError("train/image and train/label must contain matching image IDs")
    ids = [p.stem for p in images]
    rng = random.Random(args.seed)
    rng.shuffle(ids)
    if args.val_fraction <= 0:
        # Useful for the final competition fit: all labelled images are used.
        train_ids, val_ids = set(ids), set(ids)
    else:
        n_val = min(len(ids) - 1, max(1, round(len(ids) * args.val_fraction)))
        val_ids, train_ids = set(ids[:n_val]), set(ids[n_val:])
    train_paths = [p for p in images if p.stem in train_ids]
    val_paths = [p for p in images if p.stem in val_ids]
    train_loader = DataLoader(FundusDataset(train_paths, labels, args.size, True),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(FundusDataset(val_paths, labels, args.size, False),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=torch.cuda.is_available())
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UNet(args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = DiceBCELoss()
    best_dice, stale = -1.0, 0
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"device={device} train={len(train_paths)} val={len(val_paths)} size={args.size}")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_epoch(model, train_loader, optimizer, device, criterion, True)
        with torch.no_grad():
            val_loss, val_dice = run_epoch(model, val_loader, optimizer, device, criterion, False)
        scheduler.step()
        print(f"epoch {epoch:03d} train_loss={train_loss:.4f} train_dice={train_dice:.4f} "
              f"val_loss={val_loss:.4f} val_dice={val_dice:.4f}")
        if val_dice > best_dice:
            best_dice, stale = val_dice, 0
            torch.save({"model": model.state_dict(), "size": args.size, "base": args.base,
                        "val_ids": sorted(val_ids), "dice": best_dice}, out)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}; best val_dice={best_dice:.4f}")
                break
    print(f"saved {out} (best val_dice={best_dice:.4f})")


@torch.no_grad()
def predict_one(model, image: np.ndarray, size: int, device, threshold: float, tta: bool) -> np.ndarray:
    original_h, original_w = image.shape[:2]
    resized = resize_image(image, size).astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float()[None].to(device)
    probabilities = []
    for flip in (0, 1, 2, 3) if tta else (0,):
        augmented = tensor.flip((-1,)) if flip == 1 else tensor.flip((-2,)) if flip == 2 else tensor.flip((-1, -2)) if flip == 3 else tensor
        p = torch.sigmoid(model(augmented))
        if flip == 1:
            p = p.flip((-1,))
        elif flip == 2:
            p = p.flip((-2,))
        elif flip == 3:
            p = p.flip((-1, -2))
        probabilities.append(p)
    probability = torch.stack(probabilities).mean(0)[0, 0].cpu().numpy()
    mask = (probability >= threshold).astype(np.uint8)
    mask = np.asarray(Image.fromarray(mask).resize((original_w, original_h), Image.Resampling.NEAREST))
    return np.where(mask > 0, 0, 255).astype(np.uint8)


def encode_rle(mask: np.ndarray, size: int = 512) -> str:
    # Match segmentation_to_csv.py exactly: resize first, then encode all
    # non-zero values (the 255 background) in row-major order.
    encoded = np.asarray(Image.fromarray(mask).resize((size, size), Image.Resampling.BILINEAR))
    flat = (encoded.reshape(-1) > 0)
    padded = np.concatenate(([False], flat, [False])).astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    values = np.empty(starts.size * 2, dtype=np.int64)
    values[0::2] = starts + 1
    values[1::2] = ends - starts
    return " ".join(str(v) for v in values)


def predict(args):
    root = Path(args.data)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    size, base = checkpoint.get("size", args.size), checkpoint.get("base", args.base)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = UNet(base).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    image_paths = numeric_files(root / "test" / "image")
    out_dir = Path(args.mask_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in image_paths:
        image = np.asarray(Image.open(path).convert("RGB"))
        mask = predict_one(model, image, size, device, args.threshold, not args.no_tta)
        # PNG is intentional: JPEG would turn binary 0/255 edges into many
        # intermediate values, and the competition encoder treats every
        # non-zero pixel as part of the submitted mask.
        Image.fromarray(mask).save(out_dir / f"{path.stem}.png")
        rows.append((path.stem, encode_rle(mask, args.rle_size)))
    with open(args.submission, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Id", "Predicted"])
        writer.writerows(rows)
    print(f"wrote {len(rows)} masks to {out_dir} and submission to {args.submission}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(required=True)
    train_parser = sub.add_parser("train", help="train U-Net on train/image and train/label")
    train_parser.add_argument("--data", type=Path, default=Path("."))
    train_parser.add_argument("--out", type=Path, default=Path("runs/best.pt"))
    train_parser.add_argument("--size", type=int, default=512, help="square training resolution; use a multiple of 16")
    train_parser.add_argument("--base", type=int, default=24, help="first U-Net channel count")
    train_parser.add_argument("--epochs", type=int, default=120)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--workers", type=int, default=0)
    train_parser.add_argument("--lr", type=float, default=2e-4)
    train_parser.add_argument("--patience", type=int, default=25)
    train_parser.add_argument("--val-fraction", type=float, default=0.2)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default=None)
    train_parser.set_defaults(func=train)
    predict_parser = sub.add_parser("predict", help="predict test masks and create submission.csv")
    predict_parser.add_argument("--data", type=Path, default=Path("."))
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--mask-dir", type=Path, default=Path("image"))
    predict_parser.add_argument("--submission", type=Path, default=Path("submission.csv"))
    predict_parser.add_argument("--size", type=int, default=512)
    predict_parser.add_argument("--base", type=int, default=24)
    predict_parser.add_argument("--threshold", type=float, default=0.5)
    predict_parser.add_argument("--rle-size", type=int, default=512,
                                help="size used by the competition RLE converter")
    predict_parser.add_argument("--no-tta", action="store_true")
    predict_parser.add_argument("--device", default=None)
    predict_parser.set_defaults(func=predict)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
