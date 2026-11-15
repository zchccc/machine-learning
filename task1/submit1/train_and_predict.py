"""Plant seedling classification for task 2.

The bundled dataset is deliberately small (five classes and 100 labelled
images per class).  This script uses a compact, dependency-light image
classifier so it can run in the supplied Python environment without a GPU.
It combines colour, green-plant masks, spatial statistics and HOG-like
gradients, then trains a two-hidden-layer MLP and a regularised multinomial
model whose probabilities are blended for the final prediction.

Usage:
    python train_and_predict.py
    python train_and_predict.py --data-root dataset-for-task2/dataset-for-task2
    python train_and_predict.py --epochs 350 --output submission-for-task2.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageOps


DEFAULT_CLASSES = [
    "Black-grass",
    "Common wheat",
    "Loose Silky-bent",
    "Scentless Mayweed",
    "Sugar beet",
]


def _resize_rgb(path: Path, size: int = 64) -> np.ndarray:
    """Read an image, preserve its aspect ratio, and return float RGB pixels."""
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        # Padding after proportional resize prevents a tall seedling from
        # being squashed into a very different shape.
        image.thumbnail((size, size), Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (size, size), tuple(int(x) for x in image.resize((1, 1)).getpixel((0, 0))))
        canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
        return np.asarray(canvas, dtype=np.float32) / 255.0


def _histogram(values: np.ndarray, bins: int, low: float, high: float, mask: np.ndarray | None = None) -> np.ndarray:
    if mask is not None:
        values = values[mask]
    if values.size == 0:
        return np.zeros(bins, dtype=np.float32)
    result, _ = np.histogram(values, bins=bins, range=(low, high))
    result = result.astype(np.float32)
    return result / (result.sum() + 1e-6)


def _hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Small vectorised RGB->HSV conversion; values are in [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maximum = np.max(rgb, axis=-1)
    minimum = np.min(rgb, axis=-1)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    rmax = (maximum == r) & nonzero
    gmax = (maximum == g) & nonzero
    bmax = (maximum == b) & nonzero
    hue[rmax] = ((g[rmax] - b[rmax]) / delta[rmax]) % 6.0
    hue[gmax] = ((b[gmax] - r[gmax]) / delta[gmax]) + 2.0
    hue[bmax] = ((r[bmax] - g[bmax]) / delta[bmax]) + 4.0
    hue /= 6.0
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-6)
    return hue, saturation, maximum


def _block_statistics(channel: np.ndarray, grid: int = 4) -> np.ndarray:
    height, width = channel.shape
    rows = np.array_split(channel, grid, axis=0)
    output: list[float] = []
    for row in rows:
        for block in np.array_split(row, grid, axis=1):
            output.extend((float(block.mean()), float(block.std()), float(np.quantile(block, 0.9))))
    return np.asarray(output, dtype=np.float32)


def _hog(channel: np.ndarray, cells: int = 8, bins: int = 9) -> np.ndarray:
    """HOG descriptor with a fixed number of cells, implemented in NumPy."""
    gy, gx = np.gradient(channel)
    magnitude = np.hypot(gx, gy)
    angle = (np.arctan2(gy, gx) + math.pi) * bins / (2.0 * math.pi)
    height, width = channel.shape
    cell_h, cell_w = height // cells, width // cells
    result = np.zeros((cells, cells, bins), dtype=np.float32)
    for y in range(cells):
        for x in range(cells):
            mag = magnitude[y * cell_h : (y + 1) * cell_h, x * cell_w : (x + 1) * cell_w].ravel()
            ang = angle[y * cell_h : (y + 1) * cell_h, x * cell_w : (x + 1) * cell_w].ravel()
            # Bilinear interpolation between neighbouring orientation bins.
            left = np.floor(ang).astype(np.int32) % bins
            frac = ang - np.floor(ang)
            right = (left + 1) % bins
            np.add.at(result[y, x], left, mag * (1.0 - frac))
            np.add.at(result[y, x], right, mag * frac)
    result /= np.sqrt(np.sum(result * result, axis=-1, keepdims=True) + 1e-5)
    return result.ravel()


def _shape_features(mask: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Rotation-tolerant summaries of the segmented plant silhouette."""
    size = mask.shape[0]
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    mass = float(weights.sum()) + 1e-6
    cx = float((weights * xx).sum() / mass)
    cy = float((weights * yy).sum() / mass)
    radius = np.hypot(xx - cx, yy - cy) / size
    radial = []
    for lo, hi in zip(np.linspace(0.0, 0.75, 9)[:-1], np.linspace(0.0, 0.75, 9)[1:]):
        radial.append(float(weights[(radius >= lo) & (radius < hi)].sum() / mass))
    # Angular moments discard absolute orientation but retain elongation and
    # multi-lobed structure (the first moment is intentionally omitted).
    angle = np.arctan2(yy - cy, xx - cx)
    moments = []
    for order in range(2, 10):
        moments.append(float(np.abs((weights * np.exp(1j * order * angle)).sum()) / mass))
    row_mass = mask.mean(axis=1)
    col_mass = mask.mean(axis=0)
    quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    return np.asarray(
        radial
        + moments
        + [
            float(mask.mean()),
            float(mask.sum() / (np.count_nonzero(mask) + 1e-6)),
            float(mask.sum() / (size * size)),
            float(np.quantile(row_mass, quantiles).mean()),
            float(np.quantile(col_mass, quantiles).mean()),
            float(np.max(row_mass)),
            float(np.max(col_mass)),
            float(np.mean(np.abs(np.diff(row_mass)))),
            float(np.mean(np.abs(np.diff(col_mass)))),
        ],
        dtype=np.float32,
    )


def image_features(path: Path, size: int = 64) -> np.ndarray:
    """Extract robust appearance and shape features from one seedling image."""
    rgb = _resize_rgb(path, size)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    hue, saturation, value = _hsv(rgb)
    # Excess green separates leaves from the gravel/tray background.
    excess = g - 0.5 * (r + b)
    green_ratio = g / (r + g + b + 1e-4)
    mask = (excess > 0.025) & (green_ratio > 0.34) & (saturation > 0.12)

    result: list[np.ndarray] = []
    result.extend(
        [
            _histogram(hue, 18, 0.0, 1.0, saturation > 0.08),
            _histogram(saturation, 12, 0.0, 1.0),
            _histogram(value, 12, 0.0, 1.0),
            _histogram(excess, 16, -0.35, 0.35),
            _histogram(green_ratio, 16, 0.15, 0.60),
        ]
    )
    # Restricting the same histograms to likely plant pixels reduces the
    # influence of gravel and tray backgrounds.
    result.extend(
        [
            _histogram(hue, 18, 0.0, 1.0, mask),
            _histogram(saturation, 12, 0.0, 1.0, mask),
            _histogram(excess, 16, -0.35, 0.35, mask),
            _histogram(green_ratio, 16, 0.15, 0.60, mask),
        ]
    )
    for channel in (r, g, b, excess, green_ratio):
        result.append(_block_statistics(channel, grid=4))
    # Spatial mask summaries capture leaf area, centroid and elongation.
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    weights = np.maximum(excess, 0.0) * (0.5 + saturation)
    weights *= (green_ratio > 0.30)
    mass = float(weights.sum()) + 1e-6
    cx, cy = float((weights * xx).sum() / mass), float((weights * yy).sum() / mass)
    cov_xx = float((weights * (xx - cx) ** 2).sum() / mass)
    cov_yy = float((weights * (yy - cy) ** 2).sum() / mass)
    cov_xy = float((weights * (xx - cx) * (yy - cy)).sum() / mass)
    result.append(
        np.asarray(
            [
                float(mask.mean()),
                float(weights.mean()),
                cx / size,
                cy / size,
                math.sqrt(max(cov_xx, 0.0)) / size,
                math.sqrt(max(cov_yy, 0.0)) / size,
                cov_xy / (size * size),
                float(np.mean(excess > 0.05)),
                float(np.mean(green_ratio > 0.38)),
            ],
            dtype=np.float32,
        )
    )
    result.append(_shape_features(mask, weights))
    # A low-resolution green map retains the broad leaf arrangement.
    small = Image.fromarray(np.clip((excess + 0.25) * 255.0 / 0.5, 0, 255).astype(np.uint8)).resize((16, 16), Image.Resampling.BILINEAR)
    result.append(np.asarray(small, dtype=np.float32).ravel() / 255.0)
    result.append(_hog(0.6 * excess + 0.4 * g, cells=8, bins=9))

    # Normalize around the plant bounding box.  This is complementary to the
    # full-frame map for photos where the seedling occupies only a corner.
    strong = (weights > np.quantile(weights, 0.80)) & (weights > 0.015)
    ys, xs = np.where(strong)
    if len(xs) >= 4:
        margin = 3
        x0, x1 = max(0, int(xs.min()) - margin), min(size, int(xs.max()) + margin + 1)
        y0, y1 = max(0, int(ys.min()) - margin), min(size, int(ys.max()) + margin + 1)
        crop = excess[y0:y1, x0:x1]
        crop_image = Image.fromarray(np.clip((crop + 0.25) * 255.0 / 0.5, 0, 255).astype(np.uint8)).resize((32, 32), Image.Resampling.BILINEAR)
        crop_channel = np.asarray(crop_image, dtype=np.float32) / 255.0
        result.append(crop_channel.ravel())
        result.append(_hog(crop_channel, cells=4, bins=9))
    else:
        result.append(np.zeros(32 * 32, dtype=np.float32))
        result.append(np.zeros(4 * 4 * 9, dtype=np.float32))
    return np.concatenate(result).astype(np.float32)


def load_dataset(data_root: Path, size: int = 64) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    train_root = data_root / "train"
    classes = [p.name for p in sorted(train_root.iterdir()) if p.is_dir()]
    if set(DEFAULT_CLASSES).issubset(classes):
        classes = DEFAULT_CLASSES.copy()
    class_to_id = {name: i for i, name in enumerate(classes)}
    features: list[np.ndarray] = []
    labels: list[int] = []
    names: list[str] = []
    for class_name in classes:
        paths = sorted((train_root / class_name).glob("*.png"))
        for path in paths:
            features.append(image_features(path, size))
            labels.append(class_to_id[class_name])
            names.append(path.name)
    test_names = sorted(p.name for p in (data_root / "test").glob("*.png"))
    test_features = [image_features(data_root / "test" / name, size) for name in test_names]
    return np.vstack(features), np.asarray(labels, dtype=np.int64), classes, test_names, np.vstack(test_features)


def select_model_features(features: np.ndarray) -> np.ndarray:
    """Keep the validated colour and silhouette feature blocks.

    Full-frame pixel/HOG blocks remain available in ``image_features`` for
    experimentation, but they mostly describe the gravel/tray background for
    this small dataset.  The selected 161-dimensional representation is
    deterministic and is used by both validation and submission generation.
    """
    # 74 global histograms + 62 masked histograms, then 240 block statistics,
    # 9 spatial summaries and 25 rotation-tolerant shape values.
    hist_end = 74 + 62
    shape_start = hist_end + 240 + 9
    shape_end = shape_start + 25
    return np.concatenate([features[:, :hist_end], features[:, shape_start:shape_end]], axis=1)


def standardize(train: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-5] = 1.0
    return (train - mean) / scale, (other - mean) / scale, mean, scale


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - scores.max(axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(scores, -40.0, 40.0))
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def train_softmax(x: np.ndarray, y: np.ndarray, classes: int, epochs: int = 300, lr: float = 0.08, l2: float = 0.002, seed: int = 7) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    x1 = np.column_stack([x, np.ones(len(x), dtype=np.float32)])
    weights = rng.normal(0.0, 0.01, (x1.shape[1], classes)).astype(np.float32)
    one_hot = np.eye(classes, dtype=np.float32)[y]
    best = weights.copy()
    best_loss = float("inf")
    stale = 0
    for epoch in range(epochs):
        probability = softmax(x1 @ weights)
        loss = -float(np.mean(np.log(probability[np.arange(len(y)), y]) + 1e-8)) + l2 * float(np.sum(weights[:-1] ** 2))
        gradient = (x1.T @ (probability - one_hot)) / len(y)
        gradient[:-1] += 2.0 * l2 * weights[:-1]
        # A gentle decay makes the optimization stable with highly correlated
        # histogram and HOG features.
        step = lr / math.sqrt(1.0 + epoch / 40.0)
        weights -= step * gradient
        if loss < best_loss:
            best_loss, best = loss, weights.copy()
            stale = 0
        else:
            stale += 1
            if stale > 35:
                break
    return best, best_loss


def softmax_predict(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return softmax(np.column_stack([x, np.ones(len(x), dtype=np.float32)]) @ weights)


def train_mlp(
    x: np.ndarray,
    y: np.ndarray,
    classes: int,
    epochs: int = 260,
    hidden: tuple[int, int] = (64, 32),
    lr: float = 0.006,
    l2: float = 0.0008,
    seed: int = 7,
) -> dict[str, np.ndarray]:
    """Train a small two-hidden-layer network using only NumPy.

    It is intentionally compact: all 500 labelled samples fit in one batch,
    making Adam deterministic and avoiding a heavyweight framework dependency.
    """
    rng = np.random.default_rng(seed)
    dimensions = [x.shape[1], hidden[0], hidden[1], classes]
    params: dict[str, np.ndarray] = {}
    for index in range(3):
        fan_in, fan_out = dimensions[index], dimensions[index + 1]
        params[f"w{index}"] = (rng.standard_normal((fan_in, fan_out)) * math.sqrt(2.0 / fan_in)).astype(np.float32)
        params[f"b{index}"] = np.zeros(fan_out, dtype=np.float32)
    moments = {key: np.zeros_like(value) for key, value in params.items()}
    velocities = {key: np.zeros_like(value) for key, value in params.items()}
    target = np.eye(classes, dtype=np.float32)[y]
    best_params = {key: value.copy() for key, value in params.items()}
    best_loss = float("inf")
    stale = 0
    for epoch in range(1, epochs + 1):
        z1 = x @ params["w0"] + params["b0"]
        h1 = np.maximum(z1, 0.0)
        z2 = h1 @ params["w1"] + params["b1"]
        h2 = np.maximum(z2, 0.0)
        logits = h2 @ params["w2"] + params["b2"]
        probability = softmax(logits)
        loss = -float(np.mean(np.log(probability[np.arange(len(y)), y]) + 1e-8))
        loss += l2 * sum(float(np.sum(params[f"w{i}"] ** 2)) for i in range(3))
        d3 = (probability - target) / len(y)
        gradients = {
            "w2": h2.T @ d3 + 2.0 * l2 * params["w2"],
            "b2": d3.sum(axis=0),
        }
        d2 = (d3 @ params["w2"].T) * (z2 > 0.0)
        gradients["w1"] = h1.T @ d2 + 2.0 * l2 * params["w1"]
        gradients["b1"] = d2.sum(axis=0)
        d1 = (d2 @ params["w1"].T) * (z1 > 0.0)
        gradients["w0"] = x.T @ d1 + 2.0 * l2 * params["w0"]
        gradients["b0"] = d1.sum(axis=0)
        step = lr / math.sqrt(1.0 + epoch / 90.0)
        for key in params:
            moments[key] = 0.9 * moments[key] + 0.1 * gradients[key]
            velocities[key] = 0.999 * velocities[key] + 0.001 * gradients[key] ** 2
            corrected_moment = moments[key] / (1.0 - 0.9**epoch)
            corrected_velocity = velocities[key] / (1.0 - 0.999**epoch)
            params[key] -= step * corrected_moment / (np.sqrt(corrected_velocity) + 1e-8)
        if loss < best_loss:
            best_loss, best_params = loss, {key: value.copy() for key, value in params.items()}
            stale = 0
        else:
            stale += 1
            if stale > 45:
                break
    return best_params


def mlp_predict(x: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    h1 = np.maximum(x @ params["w0"] + params["b0"], 0.0)
    h2 = np.maximum(h1 @ params["w1"] + params["b1"], 0.0)
    return softmax(h2 @ params["w2"] + params["b2"])


def knn_predict(train_x: np.ndarray, train_y: np.ndarray, query_x: np.ndarray, k: int = 7) -> np.ndarray:
    # Squared Euclidean distance in a standardized feature space.
    result = np.empty((len(query_x), int(train_y.max()) + 1), dtype=np.float32)
    for start in range(0, len(query_x), 64):
        batch = query_x[start : start + 64]
        distances = ((batch[:, None, :] - train_x[None, :, :]) ** 2).mean(axis=2)
        nearest = np.argpartition(distances, kth=min(k, len(train_x) - 1), axis=1)[:, :k]
        votes = np.zeros((len(batch), result.shape[1]), dtype=np.float32)
        weights = 1.0 / (distances[np.arange(len(batch))[:, None], nearest] + 0.05)
        np.add.at(votes, (np.arange(len(batch))[:, None], train_y[nearest]), weights)
        result[start : start + len(batch)] = votes / (votes.sum(axis=1, keepdims=True) + 1e-8)
    return result


def mean_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: int) -> float:
    f1_values = []
    for label in range(classes):
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1_values.append(2.0 * precision * recall / (precision + recall + 1e-8))
    return float(np.mean(f1_values))


def stratified_split(y: np.ndarray, validation_fraction: float = 0.2, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for label in np.unique(y):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        cut = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices.extend(indices[:cut].tolist())
        train_indices.extend(indices[cut:].tolist())
    return np.asarray(train_indices), np.asarray(validation_indices)


def fit_and_score(features: np.ndarray, labels: np.ndarray, seed: int = 7) -> tuple[float, dict[str, float]]:
    train_idx, valid_idx = stratified_split(labels, seed=seed)
    train_x, valid_x, _, _ = standardize(features[train_idx], features[valid_idx])
    train_y, valid_y = labels[train_idx], labels[valid_idx]
    weights, _ = train_softmax(train_x, train_y, int(labels.max()) + 1, epochs=450, l2=0.0003, seed=seed)
    linear = softmax_predict(valid_x, weights)
    network_params = train_mlp(train_x, train_y, int(labels.max()) + 1, epochs=400, lr=0.012, l2=0.0005, seed=seed)
    network = mlp_predict(valid_x, network_params)
    scores = {}
    for linear_weight in (0.0, 0.25, 0.50, 0.75, 1.0):
        probabilities = linear_weight * linear + (1.0 - linear_weight) * network
        scores[f"linear_weight_{linear_weight:.2f}"] = mean_f1(valid_y, np.argmax(probabilities, axis=1), int(labels.max()) + 1)
    # A 50/50 blend lies on a stable plateau across independent splits.
    return scores["linear_weight_0.50"], {"linear_weight": 0.5, **scores}


def write_submission(path: Path, names: Sequence[str], classes: Sequence[str], probabilities: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predictions = np.argmax(probabilities, axis=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ID", "Category"])
        writer.writerows((name, classes[int(label)]) for name, label in zip(names, predictions))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("dataset-for-task2/dataset-for-task2"))
    parser.add_argument("--output", type=Path, default=Path("submission-for-task2.csv"))
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=350)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    raw_features, labels, classes, test_names, raw_test_features = load_dataset(args.data_root, args.size)
    features = select_model_features(raw_features)
    test_features = select_model_features(raw_test_features)
    validation_f1, diagnostics = fit_and_score(features, labels, args.seed)
    train_x, test_x, mean, scale = standardize(features, test_features)
    weights, train_loss = train_softmax(train_x, labels, len(classes), epochs=args.epochs, l2=0.0003, seed=args.seed)
    linear = softmax_predict(test_x, weights)
    network_params = train_mlp(train_x, labels, len(classes), epochs=max(args.epochs, 400), lr=0.012, l2=0.0005, seed=args.seed)
    network = mlp_predict(test_x, network_params)
    blend = 0.5
    probabilities = blend * linear + (1.0 - blend) * network
    write_submission(args.output, test_names, classes, probabilities)

    metadata = {
        "classes": classes,
        "train_images": int(len(labels)),
        "test_images": int(len(test_names)),
        "feature_dimensions": int(features.shape[1]),
        "validation_mean_f1": validation_f1,
        "validation_diagnostics": diagnostics,
        "linear_probability_weight": blend,
        "train_loss": train_loss,
        "output": str(args.output),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
