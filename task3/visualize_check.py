# -*- coding: utf-8 -*-
"""将 submission.csv 预测点画到测试图上, 生成可视化检查图"""
import pandas as pd
from PIL import Image, ImageDraw

sub = pd.read_csv("submission.csv")
vals = dict(zip(sub["ImageID"], sub["value"]))
for tid in [81, 82, 84, 85, 88, 90, 92, 96, 100]:
    x = vals[f"{tid}_Fovea_X"]; y = vals[f"{tid}_Fovea_Y"]
    img = Image.open(f"detection/test/{tid:04d}.jpg").convert("RGB")
    d = ImageDraw.Draw(img)
    R = 60
    for w in (3, 2):
        d.ellipse([x - R, y - R, x + R, y + R], outline=(255, 0, 0), width=w)
        d.line([x - R - 40, y, x + R + 40, y], fill=(255, 0, 0), width=w)
        d.line([x, y - R - 40, x, y + R + 40], fill=(255, 0, 0), width=w)
        R = 60
    img.thumbnail((760, 760))
    img.save(f"output/check_{tid:04d}.jpg")
    print(tid, x, y)
