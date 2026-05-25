param(
    [ValidateSet(
        "self_check",
        "train_patterns",
        "compare_gradients",
        "compare_decoders",
        "compare_materials",
        "compare_renderers",
        "compare_frequency_constraints",
        "all"
    )]
    [string]$Experiment = "all",
    [string]$CondaEnv = "3dv"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$LogDir = Join-Path $Root "analysis\logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$ExperimentMap = @{
    self_check = "examples\self_check\run.py"
    train_patterns = "examples\train_patterns\run.py"
    compare_gradients = "examples\compare_gradients\run.py"
    compare_decoders = "examples\compare_decoders\run.py"
    compare_materials = "examples\compare_materials\run.py"
    compare_renderers = "examples\compare_renderers\run.py"
    compare_frequency_constraints = "examples\compare_frequency_constraints\run.py"
}

if ($Experiment -eq "all") {
    $Experiments = @(
        "self_check",
        "train_patterns",
        "compare_gradients",
        "compare_decoders",
        "compare_materials",
        "compare_frequency_constraints",
        "compare_renderers"
    )
} else {
    $Experiments = @($Experiment)
}

Push-Location $Root
try {
    foreach ($Name in $Experiments) {
        $Script = $ExperimentMap[$Name]
        $Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
        $LogPath = Join-Path $LogDir "$Name`_$Stamp.log"
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] START $Name -> $LogPath"
        $Start = Get-Date
        conda run -n $CondaEnv python $Script 2>&1 | Tee-Object -FilePath $LogPath
        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $Name failed with exit code $LASTEXITCODE. See $LogPath"
        }
        $Elapsed = (Get-Date) - $Start
        Write-Host "[$(Get-Date -Format 'HH:mm:ss')] DONE $Name ($([math]::Round($Elapsed.TotalSeconds, 2))s)"
    }
} finally {
    Pop-Location
}
