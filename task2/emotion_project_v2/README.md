# FER Emotion Classification v2

本项目修复了提交标签编码，并保留原有的 ResNet18 迁移学习、五折交叉验证、MixUp 和测试时集成。

官方编码：

- `0=Angry`（愤怒）
- `1=Fear`（恐惧）
- `2=Happy`（快乐）
- `3=Sad`（悲伤）
- `4=Surprise`（惊讶）
- `5=Neutral`（中性）

原训练代码按文件夹字母顺序输出，导致 `Neutral/Sad/Surprise` 三类发生错位。`train_emotion.py` 在内部训练标签和最终 CSV 中显式使用官方编码。

运行：

```powershell
python train_emotion.py
```

输出文件为 `submission_corrected.csv`。如果已有 `models/fold*.pt`，脚本会直接预测；删除模型目录后可重新训练。
