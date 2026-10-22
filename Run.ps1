param(
    [ValidateSet('doctor','preview','capture','calibrate','manual','detect','run','track','gpu','ptz-capture','ptz-calibrate')][string]$Mode='doctor',
    [switch]$EnablePtz,
    [switch]$Ranges,
    [string]$Device = ''
)
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$ProjectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $ProjectPython)) { throw 'Run Install-Windows.ps1 first, or use your existing Python environment manually.' }
if ($EnablePtz -and $Mode -ne 'track') { throw '-EnablePtz is only supported with track.' }
if ($Ranges -and $Mode -ne 'track') { throw '-Ranges is only supported with track.' }
if ($Device -and $Mode -notin @('gpu','detect','run','track')) { throw '-Device is supported with gpu, detect, run or track.' }
$CommandArguments = @('-m', 'axis_stereo')
if ($Mode -eq 'detect') {
    $CommandArguments += @('run','--detect-only')
} else {
    $CommandArguments += $Mode
}
if ($EnablePtz) { $CommandArguments += '--enable-ptz' }
if ($Ranges) { $CommandArguments += '--ranges' }
if ($Device) { $CommandArguments += @('--device', $Device) }
& $ProjectPython @CommandArguments
if ($LASTEXITCODE -ne 0) { throw 'Command failed; read ERROR above and README_FA.md.' }
