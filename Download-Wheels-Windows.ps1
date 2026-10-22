# ONLINE Windows x64 machine, same Python version as OFFLINE destination.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
py -3.11 -m pip download --dest wheelhouse -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Wheel download failed; do not treat partial wheelhouse as complete.' }
Write-Host 'General dependencies downloaded. For GPU, separately obtain matching official CUDA torch wheels and dependencies.'
Write-Host 'Test Install-Windows.ps1 -Offline in a clean venv before transferring.'
