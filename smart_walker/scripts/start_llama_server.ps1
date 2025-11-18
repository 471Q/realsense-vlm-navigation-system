param(
  [string]$ModelPath = "models\\llm\\qwen2.5-7b-instruct\\qwen2.5-7b-instruct-q5_k_m-00001-of-00002.gguf",
  [string]$MmprojPath = "",
  [int]$NGL = 30,
  [int]$Ctx = 2048,
  [int]$Threads = 8,
  [int]$Port = 8080,
  [int]$MainGpu = -1,
  # Optional sampling/formatting controls (helpful to reduce echoing and match finetune templates)
  [string]$ChatTemplate = "",   # e.g. "llama-2", "chatml", path to a jinja file, or empty to use model default
  [double]$RepeatPenalty = 0,     # e.g. 1.10 or 1.15; 0 means don't pass flag
  [int]$RepeatLastN = -1,         # e.g. 512 or 1024; <0 means don't pass flag
  [double]$Temperature = -1,      # e.g. 0.7; <0 means don't pass flag
  [double]$TopP = -1,             # e.g. 0.9; <0 means don't pass flag
  [int]$Seed = -1,                # -1 = random; >=0 to fix seed
  [switch]$NoWarmup,              # pass --no-warmup
  [int]$CacheRam = -1,            # e.g. 8192; <0 means don't pass flag
  [switch]$Jinja,                 # enable Jinja chat templates (needed for names like "llama-2")
  [switch]$Verbose,
  # Default to modern name; will auto-fallback to others below
  [string]$ServerExe = "tools\\llama_cpp\\llama-server.exe"
)

# Resolve repo root and paths
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$serverPath = Join-Path $repoRoot $ServerExe
# Fallbacks: older/newer package names
if (-not (Test-Path $serverPath)) {
  $alt1 = Join-Path $repoRoot "tools\\llama_cpp\\server.exe"
  $alt2 = Join-Path $repoRoot "tools\\llama_cpp\\rpc-server.exe"
  if (Test-Path $alt1) { $serverPath = $alt1 }
  elseif (Test-Path $alt2) { $serverPath = $alt2 }
}
$modelPathAbs = Join-Path $repoRoot $ModelPath

if (-not (Test-Path $serverPath)) {
  Write-Error "llama-server executable not found. Place 'llama-server.exe' (or 'server.exe'/'rpc-server.exe') under tools\\llama_cpp\\."
  exit 1
}
if (-not (Test-Path $modelPathAbs)) {
  Write-Error "Model not found at '$ModelPath'. Ensure the GGUF file exists."
  exit 1
}

Write-Host "Starting llama.cpp server..." -ForegroundColor Cyan
Write-Host "  Server: $serverPath" -ForegroundColor DarkGray
Write-Host "  Model : $modelPathAbs" -ForegroundColor DarkGray

$args = @('-m', $modelPathAbs, '-ngl', $NGL, '-c', $Ctx, '-t', $Threads, '--port', $Port)
if ($MmprojPath -and $MmprojPath.Trim() -ne '') {
  $mmAbs = if ([System.IO.Path]::IsPathRooted($MmprojPath)) { $MmprojPath } else { Join-Path $repoRoot $MmprojPath }
  if (-not (Test-Path $mmAbs)) {
    Write-Error "mmproj not found at '$MmprojPath'. If using a vision model (e.g., LLaVA), provide the mmproj GGUF path."
    exit 1
  }
  $args += @('--mmproj', $mmAbs)
}
if ($MainGpu -ge 0) { $args += @('--main-gpu', $MainGpu) }
if ($ChatTemplate -and $ChatTemplate.Trim() -ne '') {
  $args += @('--chat-template', $ChatTemplate)
  # Many named templates (e.g., "llama-2") require --jinja; add it automatically when a template is provided
  $args += @('--jinja')
}
if ($RepeatPenalty -gt 0) { $args += @('--repeat-penalty', ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $RepeatPenalty))) }
if ($RepeatLastN -ge 0) { $args += @('--repeat-last-n', $RepeatLastN) }
if ($Temperature -ge 0) { $args += @('--temp', ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $Temperature))) }
if ($TopP -ge 0) { $args += @('--top-p', ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, "{0}", $TopP))) }
if ($Seed -ge 0) { $args += @('--seed', $Seed) }
if ($NoWarmup) { $args += @('--no-warmup') }
if ($CacheRam -ge 0) { $args += @('--cache-ram', $CacheRam) }
if ($Verbose) { $args += @('--verbose') }

# Pretty-print args
$displayArgs = @("-ngl $NGL", "-c $Ctx", "-t $Threads", "--port $Port")
if ($MmprojPath -and $MmprojPath.Trim() -ne '') { $displayArgs += "--mmproj $MmprojPath" }
if ($MainGpu -ge 0) { $displayArgs += "--main-gpu $MainGpu" }
if ($ChatTemplate -and $ChatTemplate.Trim() -ne '') { $displayArgs += "--chat-template $ChatTemplate"; $displayArgs += "--jinja" }
if ($RepeatPenalty -gt 0) { $displayArgs += "--repeat-penalty $RepeatPenalty" }
if ($RepeatLastN -ge 0) { $displayArgs += "--repeat-last-n $RepeatLastN" }
if ($Temperature -ge 0) { $displayArgs += "--temp $Temperature" }
if ($TopP -ge 0) { $displayArgs += "--top-p $TopP" }
if ($Seed -ge 0) { $displayArgs += "--seed $Seed" }
if ($NoWarmup) { $displayArgs += "--no-warmup" }
if ($CacheRam -ge 0) { $displayArgs += "--cache-ram $CacheRam" }
if ($Verbose) { $displayArgs += "--verbose" }
Write-Host ("  Args  : " + ($displayArgs -join ' ')) -ForegroundColor DarkGray

& $serverPath @args
