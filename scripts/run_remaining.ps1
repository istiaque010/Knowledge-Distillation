# Runs the distillation runs (Phases 8-9) one after another, then all evaluations (Phases 10-11).
# One job at a time, so every training time and speed measurement has the whole CPU.
# Usage (from project2):  powershell -ExecutionPolicy Bypass -File scripts\run_remaining.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:PYTHONIOENCODING = "utf-8"
$py = ".\.venv\Scripts\python.exe"
$log = "results\metrics\run_remaining.log"

function Step($name, [string[]]$cmd) {
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START $name" | Tee-Object -FilePath $log -Append
    & $py @cmd *>> $log
    if ($LASTEXITCODE -ne 0) { "FAILED $name (exit $LASTEXITCODE)" | Tee-Object -FilePath $log -Append; exit 1 }
    "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] DONE  $name" | Tee-Object -FilePath $log -Append
}

foreach ($T in "2", "1", "4") {
    Step "train KD T=$T a=0.5" @("-m", "kd.train_student", "--config", "configs\student.yaml", "--mode", "kd",
                                  "--temperature", $T, "--alpha", "0.5")
}
Step "eval teacher" @("-m", "kd.evaluate", "--model", "teacher")
foreach ($run in "student_scratch", "student_kd_T2_a0.5", "student_kd_T1_a0.5", "student_kd_T4_a0.5") {
    Step "eval $run" @("-m", "kd.evaluate", "--ckpt", "checkpoints\$run\model_best.pt")
}
Step "inference benchmark" @("scripts\benchmark_inference.py", "--model", "teacher",
    "--ckpt", "checkpoints\student_scratch\model_best.pt", "--ckpt", "checkpoints\student_kd_T2_a0.5\model_best.pt")
"ALL DONE" | Tee-Object -FilePath $log -Append
