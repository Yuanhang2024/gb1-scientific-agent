<#
.SYNOPSIS
    一键构建并把本仓库的 DSH 插件（dsh-gb1-scientific-agent）安装到独立的 DSH profile。

.DESCRIPTION
    脚本依次完成：定位 dsh 命令行与 Node.js 工具链 → 在 dsh-plugin 目录安装依赖并构建插件
    → 通过 `dsh plugin --profile <Profile> add ./packages/gb1` 把插件安装到独立 profile。
    不修改已有 profile，不需要管理员权限，不接触任何密钥。

.PARAMETER Profile
    DSH profile 名称，默认 gb1。

.PARAMETER Dsh
    dsh 命令行的显式路径；省略时依次探测 PATH 与常见安装位置。

.PARAMETER Node
    Node.js 解释器的显式路径；省略时依次探测 PATH 与常见安装位置（要求 Node 24 或更高）。

.PARAMETER SkipBuild
    跳过依赖安装与构建（仅当 packages/gb1/lib 已存在时使用）。

.EXAMPLE
    pwsh -NoProfile -File .\Install-DSH-Plugin.ps1
#>
[CmdletBinding()]
param(
    [string]$Profile = 'gb1',
    [string]$Dsh = '',
    [string]$Node = '',
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$RepoRoot = $PSScriptRoot
$PluginDir = Join-Path $RepoRoot 'dsh-plugin'

function Find-Tool {
    param([string]$Explicit, [string]$Name, [string[]]$Probes)
    if ($Explicit) {
        if (Test-Path -LiteralPath $Explicit) { return (Resolve-Path -LiteralPath $Explicit).Path }
        throw "$Name 不存在：$Explicit"
    }
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    foreach ($p in $Probes) {
        if ($p -and (Test-Path -LiteralPath $p)) { return (Resolve-Path -LiteralPath $p).Path }
    }
    return $null
}

function Step([string]$Message) { Write-Host ("==> " + $Message) -ForegroundColor Cyan }
function Fail([string]$Message) { Write-Host ("[失败] " + $Message) -ForegroundColor Red; exit 1 }

# 1. 定位 dsh 命令行。
$DshCmd = Find-Tool -Explicit $Dsh -Name 'dsh' -Probes @(
    'D:\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd',
    "$env:LOCALAPPDATA\Programs\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd",
    "$env:ProgramFiles\DeepSeek Harness\resources\runtime\cli\bin\dsh.cmd"
)
if (-not $DshCmd) {
    Fail '未找到 dsh 命令行。请安装 DeepSeek Harness 0.2.2-rc.1，或用 -Dsh 指定 dsh.cmd 的完整路径。'
}
Step ("dsh 命令行：" + $DshCmd)

# 2. 定位 Node.js（>= 24）。
$NodeCmd = Find-Tool -Explicit $Node -Name 'node' -Probes @(
    'D:\Programs\Strata\node\node.exe',
    "$env:LOCALAPPDATA\Programs\nodejs\node.exe",
    "$env:ProgramFiles\nodejs\node.exe"
)
if (-not $NodeCmd) {
    Fail '未找到 Node.js 24+。请安装后重试，或用 -Node 指定 node.exe 的完整路径。'
}
$NodeDir = Split-Path -Parent $NodeCmd
$nodeVersion = (& $NodeCmd --version) 2>$null
Step ("Node.js：" + $nodeVersion + "（" + $NodeCmd + "）")

# 3. 构建插件（生成 packages/gb1/lib）；-SkipBuild 时跳过，此时无需 pnpm。
if (-not $SkipBuild) {
    $env:COREPACK_ENABLE_DOWNLOAD_PROMPT = '0'
    $PnpmCmd = Get-Command pnpm -ErrorAction SilentlyContinue
    $PnpmArgs = @()
    if (-not $PnpmCmd) {
        $direct = Join-Path $NodeDir 'pnpm.cmd'
        if (Test-Path -LiteralPath $direct) {
            $PnpmCmd = @{ Source = $direct }
        } else {
            $corepack = Join-Path $NodeDir 'corepack.cmd'
            if (Test-Path -LiteralPath $corepack) {
                $PnpmCmd = @{ Source = $corepack }
                $PnpmArgs = @('pnpm')
            } else {
                Fail '未找到 pnpm，且 Node 目录中没有 corepack。请先安装 pnpm（npm install -g pnpm）。'
            }
        }
    }
    Step ("pnpm：" + $PnpmCmd.Source)

    Step '安装插件构建依赖（pnpm install）'
    Push-Location $PluginDir
    try {
        & $PnpmCmd.Source (@($PnpmArgs) + @('install')) | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail 'pnpm install 失败。' }

        Step '构建插件（node scripts/build.mjs）'
        & $NodeCmd 'scripts/build.mjs' | Out-Host
        if ($LASTEXITCODE -ne 0) { Fail '插件构建失败。' }
    } finally {
        Pop-Location
    }
}
$LibEntry = Join-Path $PluginDir 'packages\gb1\lib\index.js'
if (-not (Test-Path -LiteralPath $LibEntry)) {
    Fail "构建产物不存在：$LibEntry（请去掉 -SkipBuild 重新运行）。"
}

# 5. 安装到独立 profile。
Step ("安装插件到 profile '$Profile'（dsh plugin add）")
Push-Location $PluginDir
try {
    & $DshCmd plugin --profile $Profile add ./packages/gb1 | Out-Host
    if ($LASTEXITCODE -ne 0) { Fail 'dsh plugin add 失败。' }
} finally {
    Pop-Location
}

# 6. 收尾说明。
Write-Host ''
Write-Host '安装完成。后续步骤：' -ForegroundColor Green
Write-Host ("  1. 检查配置：  dsh --profile $Profile --dump-config")
Write-Host ("  2. 启动 DSH：  dsh --profile $Profile")
Write-Host ("  3. 在 DSH 插件设置中把 projectRoot 指向仓库根目录：$RepoRoot")
Write-Host '  4. 首次使用：打开左侧"蛋白质进化"，点击"一键准备并运行"（自动安装 Python 依赖并下载校验官方 S1）。'
