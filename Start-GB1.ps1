#requires -Version 7.0
<#
Prepare the local environment and measured data, then run the public GB1 workflow.
-PrepareOnly performs no experiment. -Offline uses the small random/model example;
its preparation can install public dependencies and download the official data.
Independent API keys are requested by the application with hidden terminal input.
#>
[CmdletBinding()]
param(
    [switch]$PrepareOnly,
    [switch]$Offline,
    [string]$Python,
    [string]$BaseUrl,
    [string]$Model,
    [ValidateSet('chat', 'responses')][string]$ApiMode,
    [string]$ReasoningEffort,
    [switch]$SaveConnection,
    [string]$IndexUrl = 'https://pypi.org/simple'
)

$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$projectRoot = $PSScriptRoot
$bootstrap = Join-Path $projectRoot 'tools/quick_start/bootstrap.py'
$launcher = Join-Path $projectRoot 'tools/quick_start/launch.py'
if (-not (Test-Path -LiteralPath $bootstrap -PathType Leaf)) {
    throw 'The public quick-start files are missing from this project.'
}
$candidates = [System.Collections.Generic.List[object]]::new()
$existing = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (Test-Path -LiteralPath $existing -PathType Leaf) { $candidates.Add(@($existing)) }
if ($Python) { $candidates.Add(@($Python)) }
$pyCommand = Get-Command 'py' -ErrorAction SilentlyContinue
if ($pyCommand) { $candidates.Add(@($pyCommand.Source, '-3.12')) }
foreach ($name in @('python3.12', 'python')) {
    $command = Get-Command $name -ErrorAction SilentlyContinue
    if ($command -and $command.Source -notlike '*WindowsApps*') { $candidates.Add(@($command.Source)) }
}
$userFolder = [Environment]::GetFolderPath([Environment+SpecialFolder]::UserProfile)
if ($userFolder) {
    $bundled = Join-Path $userFolder '.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/python.exe'
    if (Test-Path -LiteralPath $bundled -PathType Leaf) { $candidates.Add(@($bundled)) }
}
$selected = $null
foreach ($candidate in $candidates) {
    try {
        $executable = $candidate[0]
        $prefix = @($candidate | Select-Object -Skip 1)
        $arguments = @($prefix) + @('-B', '-c', 'import sys; print("GB1_PYTHON_312" if sys.version_info[:2] == (3,12) else "GB1_PYTHON_OTHER")')
        $version = & $executable @arguments 2>$null
        $code = $LASTEXITCODE
        if ($code -eq 0 -and $version -contains 'GB1_PYTHON_312') { $selected = $candidate; break }
    } catch {
        continue
    }
}
if ($null -eq $selected) { throw 'Python 3.12 is required. Supply -Python with an existing Python 3.12 executable or use the DSH GUI entry.' }
$baseExecutable = $selected[0]
$basePrefix = @($selected | Select-Object -Skip 1)
$report = Join-Path $projectRoot ('.quick-start/reports/terminal-' + [guid]::NewGuid().ToString('N') + '.json')
$prepareArguments = @($basePrefix) + @('-B', $bootstrap, '--project-root', $projectRoot, '--prepare-only', '--report', $report, '--index-url', $IndexUrl)
& $baseExecutable @prepareArguments
$prepareCode = $LASTEXITCODE
if ($prepareCode -ne 0) { exit $prepareCode }
if ($PrepareOnly) { exit 0 }
$pythonExecutable = Join-Path $projectRoot '.venv/Scripts/python.exe'
$runArguments = @('-B', $launcher, '--project-root', $projectRoot)
if ($Offline) { $runArguments += '--offline' }
if ($PSBoundParameters.ContainsKey('BaseUrl')) { $runArguments += @('--base-url', $BaseUrl) }
if ($Model) { $runArguments += @('--model', $Model) }
if ($ApiMode) { $runArguments += @('--api-mode', $ApiMode) }
if ($PSBoundParameters.ContainsKey('ReasoningEffort')) { $runArguments += @('--reasoning-effort', $ReasoningEffort) }
if ($SaveConnection) { $runArguments += '--save-connection' }
& $pythonExecutable @runArguments
exit $LASTEXITCODE
