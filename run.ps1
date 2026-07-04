$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
$EnvFile = Join-Path $ProjectRoot ".env"

function Write-Header {
    Clear-Host
    Write-Host "======================================================" -ForegroundColor Cyan
    Write-Host "              PyTorchi: Ore analyzer" -ForegroundColor Cyan
    Write-Host "              Docker launch wizard" -ForegroundColor Cyan
    Write-Host "======================================================" -ForegroundColor Cyan
    Write-Host
}

function Read-EnvValue([string]$Name, [string]$Default) {
    if (Test-Path $EnvFile) {
        $match = Get-Content $EnvFile |
            Where-Object { $_ -match "^$([regex]::Escape($Name))=" } |
            Select-Object -Last 1
        if ($match) {
            return ($match -split "=", 2)[1]
        }
    }
    return $Default
}

function Set-EnvValue([string]$Name, [string]$Value) {
    $lines = if (Test-Path $EnvFile) { @(Get-Content $EnvFile) } else { @() }
    $pattern = "^$([regex]::Escape($Name))="
    $updated = $false
    $result = foreach ($line in $lines) {
        if ($line -match $pattern) {
            if (-not $updated) {
                "$Name=$Value"
                $updated = $true
            }
        } else {
            $line
        }
    }
    if (-not $updated) {
        $result += "$Name=$Value"
    }
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllLines(
        $EnvFile,
        [string[]]$result,
        $utf8NoBom
    )
}

function Read-WithDefault([string]$Prompt, [string]$Default) {
    $value = Read-Host "$Prompt [$Default]"
    if ([string]::IsNullOrWhiteSpace($value)) { return $Default }
    return $value.Trim()
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    throw "Docker was not found. Install and start Docker Desktop."
}
docker compose version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Docker Compose v2 is not available."
}

Write-Header
Write-Host "Select inference mode:"
Write-Host "  1. CPU"
Write-Host "  2. CUDA / NVIDIA GPU"
$modeChoice = Read-Host "Mode [1]"
if ([string]::IsNullOrWhiteSpace($modeChoice)) { $modeChoice = "1" }

$composeFiles = @("-f", "docker-compose.yml")
$modeDevice = "cpu"
$modeLabel = "CPU"
if ($modeChoice -eq "2") {
    $composeFiles += @("-f", "docker-compose.gpu.yml")
    $modeDevice = "cuda"
    $modeLabel = "CUDA"
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        Write-Warning "nvidia-smi was not found. Check NVIDIA Container Toolkit."
    }
}

$defaultModelDir = Join-Path $HOME "models"
$appHost = Read-WithDefault "APP_HOST (use 127.0.0.1 for local access only)" (Read-EnvValue "APP_HOST" "127.0.0.1")
$appPort = Read-WithDefault "APP_PORT" (Read-EnvValue "APP_PORT" "8080")
if ($appPort -notmatch "^\d+$" -or [int]$appPort -lt 1 -or [int]$appPort -gt 65535) {
    throw "APP_PORT must be a number from 1 to 65535."
}
$modelDir = Read-WithDefault "MODEL_DIR (directory containing model weights)" (Read-EnvValue "MODEL_DIR" $defaultModelDir)
$talcCheckpoint = Read-WithDefault "TALC_CHECKPOINT_FILE" (Read-EnvValue "TALC_CHECKPOINT_FILE" "talc.pt")
$sulfideCheckpoint = Read-WithDefault "SULFIDE_CHECKPOINT_FILE" (Read-EnvValue "SULFIDE_CHECKPOINT_FILE" "sulfide.pt")

Write-Header
Write-Host "Mode:                $modeLabel"
Write-Host "Address:             http://localhost:$appPort"
Write-Host "MODEL_DIR:          $modelDir"
Write-Host "Talc checkpoint:    $talcCheckpoint"
Write-Host "Sulfide checkpoint: $sulfideCheckpoint"
Write-Host

if (-not (Test-Path $modelDir -PathType Container)) {
    Write-Warning "MODEL_DIR does not exist; analysis will be unavailable until model weights are added."
} else {
    if (-not (Test-Path (Join-Path $modelDir $talcCheckpoint))) {
        Write-Warning "$talcCheckpoint was not found."
    }
    if (-not (Test-Path (Join-Path $modelDir $sulfideCheckpoint))) {
        Write-Warning "$sulfideCheckpoint was not found."
    }
}

if (Test-Path $EnvFile) {
    $backup = "$EnvFile.backup.$(Get-Date -Format 'yyyyMMdd_HHmmss')"
    Copy-Item $EnvFile $backup
    Write-Host "Backup .env: $backup" -ForegroundColor DarkGray
}

Set-EnvValue "APP_HOST" $appHost
Set-EnvValue "APP_PORT" $appPort
Set-EnvValue "MODEL_DIR" $modelDir
Set-EnvValue "TALC_CHECKPOINT_FILE" $talcCheckpoint
Set-EnvValue "SULFIDE_CHECKPOINT_FILE" $sulfideCheckpoint
Set-EnvValue "MODEL_DEVICE" $modeDevice

Write-Host "Validating configuration..." -ForegroundColor Cyan
& docker compose @composeFiles config --quiet
if ($LASTEXITCODE -ne 0) { throw "Invalid Docker Compose configuration." }

Write-Host "Building and starting containers..." -ForegroundColor Cyan
& docker compose @composeFiles up --build -d
if ($LASTEXITCODE -ne 0) { throw "Failed to start the application." }

$url = "http://localhost:$appPort"
Write-Host
Write-Host "Ready: $url" -ForegroundColor Green
$openBrowser = Read-Host "Open the application in a browser? [Y/n]"
if ([string]::IsNullOrWhiteSpace($openBrowser) -or $openBrowser -match "^[Yy]") {
    Start-Process $url
}
