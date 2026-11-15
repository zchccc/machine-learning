# 多种子训练守护: 每个种子 5 折, 进程死掉自动 --resume 续训
param([int[]]$Seeds = @(777, 2024))
foreach ($sd in $Seeds) {
    $dir = "output_s$sd"
    $log = "train_s$sd.log"
    for ($i = 1; $i -le 30; $i++) {
        $done = (Get-ChildItem $dir -Filter "fold*_meta.json" -ErrorAction SilentlyContinue).Count
        if ($done -ge 5) { Write-Output "SEED $sd ALL 5 FOLDS DONE"; break }
        Write-Output "=== seed $sd round $i (have $done/5 meta), starting ==="
        py -u train_fovea.py --epochs 60 --folds 5 --seed $sd --out-dir $dir --skip-infer 2>&1 | Tee-Object -FilePath $log -Append
        Start-Sleep -Seconds 5
    }
}
Write-Output "multi-seed guard exit"
