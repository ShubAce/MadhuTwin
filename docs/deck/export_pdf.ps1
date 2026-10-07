# Export a .pptx to PDF (and optionally PNG per slide) using installed Microsoft PowerPoint.
#   powershell -File docs/deck/export_pdf.ps1 -Pptx docs/architecture.pptx [-PngDir artifacts/shots/arch]
param(
  [Parameter(Mandatory = $true)][string]$Pptx,
  [string]$PngDir = ""
)
$ErrorActionPreference = "Stop"
$full = (Resolve-Path $Pptx).Path
$pdf = [System.IO.Path]::ChangeExtension($full, ".pdf")
$app = New-Object -ComObject PowerPoint.Application
try {
  # Open(FileName, ReadOnly, Untitled, WithWindow)
  $pres = $app.Presentations.Open($full, -1, 0, 0)  # msoTrue = -1, msoFalse = 0
  $pres.SaveAs($pdf, 32)  # ppSaveAsPDF
  if ($PngDir -ne "") {
    New-Item -ItemType Directory -Force $PngDir | Out-Null
    $dir = (Resolve-Path $PngDir).Path
    $i = 1
    foreach ($s in $pres.Slides) {
      $s.Export((Join-Path $dir ("slide-{0:D2}.png" -f $i)), "PNG", 1920, 1080)
      $i++
    }
  }
  $pres.Close()
  Write-Output "PDF: $pdf"
} finally {
  $app.Quit()
  [System.Runtime.InteropServices.Marshal]::ReleaseComObject($app) | Out-Null
}
