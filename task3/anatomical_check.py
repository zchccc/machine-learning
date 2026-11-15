# -*- coding: utf-8 -*-
"""解剖学合理性检查:
1. 用"亮斑检测"估计每张图的视盘中心与直径
2. 在训练集上统计 GT 中心凹相对视盘的几何关系(距离/角度, 以视盘直径为单位)
3. 用该先验在测试集上给出"解剖学预期位置", 与模型预测对比, 找出异常预测
"""
import glob, os, json
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(ROOT, "detection", "train")
TEST_DIR = os.path.join(ROOT, "detection", "test")
GT_CSV = os.path.join(ROOT, "detection", "fovea_localization_train_GT.csv")
SUB = os.path.join(ROOT, "submission.csv")


def find_disc(img):
    """粗略估计视盘中心与直径: 取亮度前1%像素的形心+散布."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    H, W = g.shape
    # 大幅模糊抑制血管/亮点影响
    im = Image.fromarray(g.astype(np.uint8)).filter(ImageFilter.GaussianBlur(21))
    g2 = np.asarray(im, dtype=np.float32)
    thr = np.percentile(g2, 99.0)
    ys, xs = np.where(g2 >= thr)
    cx, cy = xs.mean(), ys.mean()
    d = max(np.percentile(xs, 90) - np.percentile(xs, 10),
            np.percentile(ys, 90) - np.percentile(ys, 10))
    return float(cx), float(cy), float(d)


def main():
    gt = pd.read_csv(GT_CSV)
    gt_map = {}
    for _, r in gt.iterrows():
        fn = str(r["data"])
        stem, ext = os.path.splitext(fn)
        key = stem.zfill(4)
        p = os.path.join(TRAIN_DIR, key + ext)
        if not os.path.exists(p):
            p = os.path.join(TRAIN_DIR, key + ".jpg")
        gt_map[key] = (float(r["Fovea_X"]), float(r["Fovea_Y"]), p)

    # --- 训练集: 视盘 -> 中心凹 几何关系 ---
    stats = []
    for key, (fx, fy, p) in gt_map.items():
        img = Image.open(p)
        dx, dy, dd = find_disc(img)
        vx, vy = fx - dx, fy - dy
        dist = (vx * vx + vy * vy) ** 0.5
        stats.append({"id": key, "disc": (dx, dy, dd),
                      "fovea": (fx, fy), "vec": (vx, vy),
                      "dist_dd": dist / dd, "ang": np.degrees(np.arctan2(vy, vx))})
    dd_arr = np.array([s["dist_dd"] for s in stats])
    ang_arr = np.array([s["ang"] for s in stats])
    print("== 训练集 视盘->中心凹 先验 (单位: 视盘直径 DD) ==")
    print(f"距离/DD: mean={dd_arr.mean():.2f} std={dd_arr.std():.2f} "
          f"min={dd_arr.min():.2f} max={dd_arr.max():.2f}")
    print(f"角度(deg): mean={ang_arr.mean():.1f} std={ang_arr.std():.1f}")
    # 水平/垂直分量
    hx = np.array([s["vec"][0] / s["disc"][2] for s in stats])
    hy = np.array([s["vec"][1] / s["disc"][2] for s in stats])
    print(f"水平分量/DD: mean={hx.mean():.2f} std={hx.std():.2f} "
          f"min={hx.min():.2f} max={hx.max():.2f}")
    print(f"垂直分量/DD: mean={hy.mean():.2f} std={hy.std():.2f} "
          f"min={hy.min():.2f} max={hy.max():.2f}")
    # 逐图列出离群者
    for s in stats:
        if abs(s["dist_dd"] - dd_arr.mean()) > 2.5 * dd_arr.std():
            print(f"  [训练离群] {s['id']}: dist={s['dist_dd']:.2f}DD ang={s['ang']:.0f}")

    # --- 测试集: 先验位置 vs 预测 ---
    sub = pd.read_csv(SUB)
    vals = dict(zip(sub["ImageID"], sub["value"]))
    rows = []
    for p in sorted(glob.glob(os.path.join(TEST_DIR, "*"))):
        key = os.path.splitext(os.path.basename(p))[0]
        kint = str(int(key))  # submission 用无前导零 id
        px = vals.get(f"{kint}_Fovea_X", 0.0)
        py = vals.get(f"{kint}_Fovea_Y", 0.0)
        img = Image.open(p)
        dx, dy, dd = find_disc(img)
        # 先验: 视盘中心 + 训练集平均向量(DD 单位)
        prior_x = dx + hx.mean() * dd
        prior_y = dy + hy.mean() * dd
        dev = ((px - prior_x) ** 2 + (py - prior_y) ** 2) ** 0.5
        rows.append({"id": key, "pred": (px, py), "prior": (prior_x, prior_y),
                     "disc": (dx, dy, dd), "dev_px": dev, "dev_dd": dev / dd,
                     "pred_dd": ((px - dx) ** 2 + (py - dy) ** 2) ** 0.5 / dd})
    rows.sort(key=lambda r: -r["dev_dd"])
    print("\n== 测试集: 预测 vs 解剖学先验 (按偏差排序) ==")
    print(f"{'id':>6} {'pred_x':>8} {'pred_y':>8} {'prior_x':>8} {'prior_y':>8} "
          f"{'dev_px':>7} {'dev_dd':>6} {'pred_dd':>7}")
    for r in rows:
        flag = "  <== 偏差大" if r["dev_dd"] > 0.8 else ""
        print(f"{r['id']:>6} {r['pred'][0]:8.1f} {r['pred'][1]:8.1f} "
              f"{r['prior'][0]:8.1f} {r['prior'][1]:8.1f} "
              f"{r['dev_px']:7.1f} {r['dev_dd']:6.2f} {r['pred_dd']:7.2f}{flag}")


if __name__ == "__main__":
    main()
