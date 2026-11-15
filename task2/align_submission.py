# -*- coding: utf-8 -*-
"""把 submission_new.csv 按 submission.csv(模板) 的 ID 顺序对齐并覆盖 submission.csv。
用法：py align_submission.py
"""
import csv, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
new_path = os.path.join(ROOT, "submission_new.csv")
tmpl_path = os.path.join(ROOT, "submission.csv")

if not os.path.exists(new_path):
    sys.exit("未找到 submission_new.csv，请先运行 py train_fer.py")

with open(tmpl_path, newline="") as f:
    tmpl = list(csv.reader(f))
header_t, rows_t = tmpl[0], tmpl[1:]

with open(new_path, newline="") as f:
    pred = {r[0]: r[1] for r in csv.reader(f)}

missing = [r[0] for r in rows_t if r[0] not in pred]
extra = [k for k in pred if k not in {r[0] for r in rows_t}]
if missing: sys.exit(f"预测文件缺少 {len(missing)} 个 ID，例如 {missing[:3]}")
if extra:   print(f"[warn] 预测文件多出 {len(extra)} 个 ID，将被忽略", flush=True)

with open(tmpl_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(header_t)
    w.writerows((r[0], pred[r[0]]) for r in rows_t)
print(f"已按模板顺序覆盖 {tmpl_path}（{len(rows_t)} 行）", flush=True)
