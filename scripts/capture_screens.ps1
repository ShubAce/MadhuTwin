# Capture dashboard screenshots for the README, presentation and video thumbnails.
# Requires the API (port 8000) and the dashboard dev server (port 5173) to be running.
#   powershell -ExecutionPolicy Bypass -File scripts/capture_screens.ps1
param([string]$Base = "http://localhost:5173")
$ErrorActionPreference = "Stop"
$edge = "C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
$root = Split-Path -Parent $PSScriptRoot
$out = Join-Path $root "docs\screenshots"
New-Item -ItemType Directory -Force $out | Out-Null
$prof = Join-Path $env:TEMP "mt-edge-capture"

$index = Get-Content (Join-Path $root "artifacts\demo\index.json") -Raw | ConvertFrom-Json
function Story($word) { ($index | Where-Object { $_.story -like "*$word*" } | Select-Object -First 1).id }
$hypo = Story "Premixed"
$fest = Story "Festival"
$ill = Story "illness"
# moments chosen by scripts/make_report.py: the first hypo alert, and the illness patient's worst day
$tok = Get-Content (Join-Path $root "artifacts\results\demo_tokens.json") -Raw | ConvertFrom-Json
$hc = $tok.hypo_clock
$ic = $tok.ill_clock

$shots = @(
  @{ name = "panel_light.png";     w = 1440; h = 1000; url = "/?theme=light#/" },
  @{ name = "panel_dark.png";      w = 1440; h = 1000; url = "/?theme=dark#/" },
  @{ name = "patient_light.png";   w = 1440; h = 1240; url = "/?theme=light#/patient/$hypo" + "?clock=$hc" },
  @{ name = "patient_full.png";    w = 1440; h = 2300; url = "/?theme=light#/patient/$ill" + "?clock=$ic" },
  @{ name = "whatif_light.png";    w = 1440; h = 2300; url = "/?theme=light#/patient/$hypo" + "?tab=whatif&food=rice_dal&walk=15&run=1&clock=780" },
  @{ name = "agp_light.png";       w = 1440; h = 2000; url = "/?theme=light#/patient/$fest" + "?tab=agp" },
  @{ name = "fhir_light.png";      w = 1440; h = 2000; url = "/?theme=light#/patient/$hypo" + "?tab=fhir" },
  @{ name = "companion_hi.png";    w = 460;  h = 980;  url = "/?theme=light#/companion/$ill" + "?lang=hi&clock=$ic" },
  @{ name = "companion_kn.png";    w = 460;  h = 980;  url = "/?theme=light#/companion/$ill" + "?lang=kn&clock=$ic" },
  @{ name = "evidence_light.png";  w = 1440; h = 3000; url = "/?theme=light#/evidence" },
  @{ name = "privacy_light.png";   w = 1440; h = 1300; url = "/?theme=light#/privacy" }
)
foreach ($s in $shots) {
  $file = Join-Path $out $s.name
  $ErrorActionPreference = "Continue"  # Edge reports progress on stderr, which PowerShell 5.1 would treat as fatal
  & $edge --headless --disable-gpu --no-first-run --hide-scrollbars "--user-data-dir=$prof" "--window-size=$($s.w),$($s.h)" --virtual-time-budget=25000 "--screenshot=$file" ($Base + $s.url) 2>&1 | Out-Null
  Start-Sleep -Milliseconds 800
  Write-Output ("{0,-22} {1,8:N0} bytes" -f $s.name, (Get-Item $file).Length)
}
