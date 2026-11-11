# Phase 9, extra ablations: alpha 0.25 and 0.75 at T = 2, then their evaluations. Same logging as run_remaining.ps1.
# Usage (from project2):  powershell -ExecutionPolicy Bypass -File scripts\run_alpha.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$log = "results\metrics\run_alpha.log"

function Step($name, [string[]]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -FilePath $log -Append
    & $py @cmd *>> $log
    if ($LASTEXITCODE -ne 0) { "FAILED $name (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append; exit 1 }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $name" | Tee-Object -FilePath $log -Append
}

foreach ($A in "0.25", "0.75") {
    Step "train KD T=2 a=$A" @("-m", "kd.train_student", "--config", "configs\student.yaml", "--mode", "kd",
                               "--temperature", "2", "--alpha", $A)
}
foreach ($run in "student_kd_T2_a0.25", "student_kd_T2_a0.75") {
    Step "eval $run" @("-m", "kd.evaluate", "--ckpt", "checkpoints\$run\model_best.pt")
}
"ALL DONE" | Tee-Object -FilePath $log -Append
