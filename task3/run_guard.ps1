# 训练守护循环: 进程死掉自动用 --resume 续训, 直到 5 折 meta 齐全
$log = "train_v14.log"
for ($i = 1; $i -le 30; $i++) {
    $done = (Get-ChildItem output -Filter "fold*_meta.json" -ErrorAction SilentlyContinue).Count
    if ($done -ge 5) { Write-Output "ALL 5 FOLDS DONE"; break }
    Write-Output "=== round $i (have $done/5 meta), starting ==="
    py -u train_fovea.py --epochs 60 --folds 5 --resume 2>&1 | Tee-Object -FilePath $log -Append
    Start-Sleep -Seconds 5
}
Write-Output "guard loop exit"
