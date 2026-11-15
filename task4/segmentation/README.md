# 眼底血管分割

`vessel_segmentation.py` 是一个可复现的 PyTorch U-Net 训练和推理脚本，适配当前目录结构：

```text
segmentation/
  train/image/*.jpg
  train/label/*.jpg
  test/image/*.jpg
```

标签中黑色像素是血管（0），白色像素是背景（255）。模型内部把血管作为前景训练，预测结果仍保存为竞赛要求的血管=0、背景=255。

## 安装

Windows + NVIDIA GPU 建议使用项目根目录的虚拟环境。当前已验证 RTX 4060、驱动 566.07：

```powershell
cd "C:\Users\管理员\Desktop\machine learing\task4"
.\.venv\Scripts\Activate.ps1
cd segmentation
python -m pip install -r requirements.txt
```

如果尚未创建虚拟环境：

```powershell
cd "C:\Users\管理员\Desktop\machine learing\task4"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
cd segmentation
python -m pip install -r requirements.txt
```

## 训练

在 `segmentation` 目录执行：

```bash
python vessel_segmentation.py train --epochs 120 --out runs/best.pt
```

默认使用固定随机种子、80/20 图像级验证划分、Dice+BCE 损失和早停。GPU 可用时自动使用 CUDA；也可以显式指定 `--device cpu` 或 `--device cuda`。

提交最终模型时可以让 20 张标注图全部参与训练（此时验证指标是训练集指标）：

```bash
python vessel_segmentation.py train --val-fraction 0 --out runs/final.pt
```

## 预测和提交

```bash
python vessel_segmentation.py predict --checkpoint runs/best.pt
```

该命令会生成：

- `image/1.png` ... `image/20.png`：原图尺寸的无损二值预测图；
- `submission.csv`：按数字图像 ID 排序，并复刻题目脚本先缩放到 `512x512`、再用 1-based 行优先索引进行 RLE 编码。

RLE 与题目提供的 `segmentation_to_csv.py` 保持一致，编码的是预测图中的非零像素（即背景 255）。如需关闭四方向翻转测试增强，追加 `--no-tta`。

## 复现实验

训练过程会把最佳验证 Dice、输入尺寸、网络宽度和验证 ID 保存到 checkpoint。预测命令从 checkpoint 读取网络配置，因此不需要手动保持训练参数一致。若显存不足，可使用 `--size 384 --base 16 --batch-size 1`。
