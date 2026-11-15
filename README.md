# Machine Learning Course Projects

This repository contains the source code and lightweight submission artifacts for four computer-vision course tasks.

| Task | Problem | Main method | Evaluation recorded in the project |
| --- | --- | --- | --- |
| task1 | Plant seedling classification | Color, silhouette and HOG features with a linear/MLP probability blend | Mean F1 validation |
| task2 | Facial expression classification | ImageNet-pretrained ResNet-18, augmentation, mixup and five-fold ensemble | Five-fold validation accuracy |
| task3 | Fovea localization | ImageNet-pretrained ResNet-34 coordinate regression with geometric augmentation and TTA | Pixel RMSE |
| task4 | Retinal vessel segmentation | U-Net with Dice+BCE loss and four-direction TTA | Validation Dice |

## Layout

- `task1/submit1/`: plant classification training and prediction script, diagnostics and submission file.
- `task2/`: FER training script, checkpoints excluded from Git, and submission files.
- `task3/`: fovea localization scripts and lightweight metadata/submission files.
- `task4/segmentation/`: U-Net training/inference script, requirements and submission encoder.

Datasets, image folders, Python caches, model checkpoints, virtual environments and large experiment dumps are intentionally excluded by `.gitignore`. Place the original competition data in the paths expected by each task before running training.

## Reproduction commands

```text
# task1
cd task1/submit1
python train_and_predict.py

# task2
python task2/train_fer.py

# task3
python task3/train_fovea.py

# task4
cd task4/segmentation
python vessel_segmentation.py train --epochs 120 --out runs/best.pt
python vessel_segmentation.py predict --checkpoint runs/best.pt
```

Repository: https://github.com/zchccc/machine-learning
