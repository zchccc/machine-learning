# -*- coding: utf-8 -*-
"""对比 train_location XML 的 Fovea 框中心与 GT 坐标, 找 GT 可疑样本"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "detection"

gt = pd.read_csv(DATA / "fovea_localization_train_GT.csv").set_index("data")

rows = []
for xmlp in sorted((DATA / "train_location").glob("*.xml")):
    tid = int(xmlp.stem)
    root = ET.parse(xmlp).getroot()
    # XML 记录的原图尺寸
    xw = int(root.find("size/width").text)
    xh = int(root.find("size/height").text)
    # 实际 jpg 尺寸
    W, H = Image.open(DATA / "train" / f"{tid:04d}.jpg").size
    box = None
    for obj in root.findall("object"):
        if obj.findtext("name") == "Fovea":
            b = obj.find("bndbox")
            box = [int(b.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax")]
    g = gt.loc[tid]
    if box is None:
        rows.append((tid, W, H, xw, xh, None, None, g["Fovea_X"], g["Fovea_Y"]))
        continue
    # XML -> 实际 jpg 坐标换算
    sx, sy = W / xw, H / xh
    cx = (box[0] + box[2]) / 2 * sx
    cy = (box[1] + box[3]) / 2 * sy
    rows.append((tid, W, H, xw, xh, cx, cy, g["Fovea_X"], g["Fovea_Y"]))

print(f"{'图':>4} {'实际WxH':>11} {'XML尺寸':>11} {'框中心x':>8} {'框中心y':>8} {'GT_x':>8} {'GT_y':>8} {'dx':>7} {'dy':>7} {'dist':>7} {'GT在框内':>6}")
bad = []
for tid, W, H, xw, xh, cx, cy, gx, gy in rows:
    if cx is None:
        print(f"{tid:4d} {W}x{H} {xw}x{xh}  无Fovea框")
        continue
    dx, dy = gx - cx, gy - cy
    dist = (dx ** 2 + dy ** 2) ** 0.5
    # 判断 GT 是否落在 XML 框内(换算到实际坐标)
    inside = False
    box_half = None
    if tid in gt.index:
        xmlp = DATA / "train_location" / f"{tid:04d}.xml"
        r = ET.parse(xmlp).getroot()
        for obj in r.findall("object"):
            if obj.findtext("name") == "Fovea":
                b = obj.find("bndbox")
                bb = [int(b.findtext(t)) for t in ("xmin", "ymin", "xmax", "ymax")]
                sx, sy = W / xw, H / xh
                x0, y0, x1, y1 = bb[0] * sx, bb[1] * sy, bb[2] * sx, bb[3] * sy
                inside = x0 <= gx <= x1 and y0 <= gy <= y1
    mark = "" if dist < 100 else "  <-- 可疑"
    if dist >= 100 or not inside:
        bad.append((tid, dist, inside))
    print(f"{tid:4d} {W:5d}x{H:<5d} {xw:5d}x{xh:<5d} {cx:8.0f} {cy:8.0f} {gx:8.0f} {gy:8.0f} {dx:7.0f} {dy:7.0f} {dist:7.0f} {str(inside):>6}{mark}")

print(f"\nGT 距 XML 框中心 >=100px 或在框外: {len(bad)} 张")
for tid, dist, inside in bad:
    print(f"  图{tid:03d}: dist={dist:.0f}px inside={inside}")
