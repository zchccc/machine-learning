# Plant Classification Submission

`submission-for-task2.csv` contains one prediction for every PNG in the test
directory.  The required columns are `ID` (the original test filename) and
`Category` (one of the five training-folder names).

The result is reproducible with the bundled Python runtime and does not need a
GPU or third-party machine-learning package:

```text
python train_and_predict.py
```

The program reads `dataset-for-task2/dataset-for-task2`, performs a stratified
Mean F1 validation, trains a compact NumPy MLP plus a regularized classifier,
and writes both the CSV and a JSON diagnostics file.  Mean F1 is calculated as
the unweighted mean of the per-class F1 values, matching the task definition.
