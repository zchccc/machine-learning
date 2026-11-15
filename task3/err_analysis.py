# -*- coding: utf-8 -*-
"""误差诊断:
1. XML 框中心 vs GT 偏差统计 (框先验的可用性)
2. 逐图 OOF 误差排名 (找长尾失败模式)
3. 按图像属性(亮度/视盘位置/图像尺寸)分析误差相关性
"""
import glob, os, re
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.abspath(__file__))
GT_CSV = os.path.join(ROOT, "detection", "fovea_localization_train_GT.csv")
XML_DIR = os.path.join(ROOT, "detection", "train_location")
TRAIN_DIR = os.path.join(ROOT, "detection", "train")
CV_JSON = os.path.join(ROOT, "output", "cv_summary.json")


def parse_xml(p):
    t = ET.parse(p)
    box = t.find("object/bndbox")
    x1, y1 = float(box.find("xmin").text), float(box.find("ymin").text)
    x2, y2 = float(box.find("xmax").text), float(box.find("ymax").text)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def main():
    gt = pd.read_csv(GT_CSV)
    gt_map = {}
    for _, r in gt.iterrows():
        stem = os.path.splitext(str(r["data"]))[0]
        key = stem.zfill(4)
        gt_map[key] = (float(r["Fovea_X"]), float(r["Fovea_Y"]))

    # 1. XML 框中心 vs GT
    offs, sizes = [], []
    for key, (fx, fy) in gt_map.items():
        xp = os.path.join(XML_DIR, key + ".xml")
        if not os.path.exists(xp):
            continue
        cx, cy, w, h = parse_xml(xp)
        offs.append((fx - cx, fy - cy))
        sizes.append((w, h))
    offs = np.array(offs); sizes = np.array(sizes)
    print("== XML框中心 vs GT ==")
    print(f"框尺寸: w={sizes[:,0].mean():.0f}±{sizes[:,0].std():.0f}  h={sizes[:,1].mean():.0f}±{sizes[:,1].std():.0f}")
    print(f"GT-框中心: dx mean={offs[:,0].mean():.1f} std={offs[:,0].std():.1f} "
          f"|dy| max={np.abs(offs[:,1]).max():.1f}  dx max={np.abs(offs[:,0]).max():.1f}")

    # 2. 逐图 OOF 误差
    cv_path = None
    for cand in ["cv_summary.json", "cv_summary_s2h.json", "cv_summary_heat.json"]:
        p = os.path.join(ROOT, "output", cand)
        if os.path.exists(p):
            cv_path = p
    print(f"\nOOF来源: {cv_path}")
    if cv_path:
        d = __import__("json").load(open(cv_path, encoding="utf-8"))
        print("内容键:", list(d.keys())[:8])


if __name__ == "__main__":
    main()
