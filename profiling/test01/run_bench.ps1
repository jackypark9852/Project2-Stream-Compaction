# run_bench.ps1
# PowerShell runner for your CSV-capable benchmark binary

param(
  [string]$ExePath = ".\cis5650_stream_compaction_test.exe",       # path to your built exe
  [int]$MinExp = 2,
  [int]$MaxExp = 28,
  [int]$Runs = 5                        # should match what the program prints in summary
)

# ---- Setup output folder & CSVs ----
$stamp = (Get-Date).ToString("yyyy-MM-dd_HH-mm-ss")
$outDir = Join-Path -Path "." -ChildPath "results\$stamp"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $outDir "logs") -Force | Out-Null

$runsCsv = Join-Path $outDir "runs.csv"
$summaryCsv = Join-Path $outDir "summary.csv"
$metaFile = Join-Path $outDir "meta.txt"

# CSV headers
"timestamp,test_id,exp,n,pot_kind,run_idx,time_ms,status" | Out-File -FilePath $runsCsv -Encoding ascii -Force
"timestamp,test_id,exp,n,pot_kind,runs,ok_runs,avg_ms,std_ms,min_ms,max_ms,status" | Out-File -FilePath $summaryCsv -Encoding ascii -Force

# Metadata (optional)
$gpu = ""
try {
  $gpu = (nvidia-smi --query-gpu="name,driver_version" --format=csv,noheader 2>$null) -join "; "
} catch { }
"build=Release" | Out-File -FilePath $metaFile -Encoding ascii -Append
"exe=$ExePath"   | Out-File -FilePath $metaFile -Encoding ascii -Append
"gpu=$gpu"       | Out-File -FilePath $metaFile -Encoding ascii -Append
"timestamp=$stamp" | Out-File -FilePath $metaFile -Encoding ascii -Append

# ---- Define tests (one flag each; your program prints test_id/pot_kind itself) ----
$tests = @(
  "--scan-cpu-pot",
  # "--scan-cpu-npot",
  "--scan-naive-pot",
  # "--scan-naive-npot",
  "--scan-efficient-pot",
  # "--scan-efficient-npot",
  "--scan-thrust-pot"
  # "--scan-thrust-npot",
  # "--compact-cpu-noscan-pot",
  # "--compact-cpu-noscan-npot",
  # "--compact-cpu-scan",
  # "--compact-efficient-pot",
  # "--compact-efficient-npot"
)

Write-Host "Output -> $outDir"
Write-Host "Exe     -> $ExePath"
Write-Host "Exps    -> $MinExp..$MaxExp"
Write-Host "Runs    -> $Runs"
Write-Host ""

# ---- Helper: run one test/exp, warm-up then measured, split CSV ----
function Invoke-One {
  param(
    [string]$flag,
    [int]$exp
  )

  $logBase = "$($flag.TrimStart('-').Replace('=','_').Replace(' ','_'))_exp$('{0:d2}' -f $exp)"
  $logPath = Join-Path $outDir "logs\$logBase.txt"

  # Warm-up (discard output)
  try {
    & $ExePath --csv --runs=1 --exp=$exp $flag *> $null
  } catch {
    # even if warm-up fails, continue
    "WARMUP ERROR: $($_.Exception.Message)" | Out-File -FilePath $logPath -Encoding ascii -Append
  }

  # Measured run
  $out = $null
  try {
    $out = & $ExePath --csv --runs=$Runs --exp=$exp $flag 2>&1
  } catch {
    # capture any thrown error text
    $out = @("EXCEPTION: $($_.Exception.Message)")
  }

  # Always log raw output
  $out | Out-File -FilePath $logPath -Encoding ascii -Append

  if ($null -eq $out) { return }

  # Append RUN lines to runs.csv (strip 'RUN,')
  $out | Where-Object { $_ -is [string] -and $_.StartsWith("RUN,") } |
    ForEach-Object { $_.Substring(4) } |
    Add-Content -Path $runsCsv -Encoding ascii

  # Append SUMMARY lines to summary.csv (strip 'SUMMARY,')
  $out | Where-Object { $_ -is [string] -and $_.StartsWith("SUMMARY,") } |
    ForEach-Object { $_.Substring(8) } |
    Add-Content -Path $summaryCsv -Encoding ascii
}

# ---- Sweep tests × exponents ----
foreach ($flag in $tests) {
  Write-Host "=== $flag ==="
  foreach ($exp in $MinExp..$MaxExp) {
    Write-Host ("  exp={0,2}" -f $exp) -NoNewline
    Invoke-One -flag $flag -exp $exp
    Write-Host "  ...done"
  }
}

Write-Host ""
Write-Host "All done."
Write-Host "  runs.csv   -> $runsCsv"
Write-Host "  summary.csv-> $summaryCsv"
Write-Host "  logs/      -> $($outDir)\logs"
