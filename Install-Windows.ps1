param([switch]$Offline, [string]$Wheelhouse = 'wheelhouse')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
if (-not (Test-Path '.venv\Scripts\python.exe')) {
    py -3.11 -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.11 x64, or create .venv manually with your Python version.' }
}
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if ($Offline) {
    & $ProjectPython -m pip install --no-index --find-links $Wheelhouse -r requirements.txt
} else {
    & $ProjectPython -m pip install -r requirements.txt
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
if (-not (Test-Path config.yaml)) { Copy-Item config.example.yaml config.yaml }
& $ProjectPython -c "import cv2, torch, ultralytics; print('OpenCV',cv2.__version__); print('Torch',torch.__version__); print('CUDA',torch.cuda.is_available()); print('Ultralytics',ultralytics.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'Dependency verification failed.' }
Write-Host 'Ready. Read README_FA.md; configure credentials and camera geometry.'
