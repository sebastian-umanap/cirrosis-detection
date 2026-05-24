# Extrae los ZIPs de CirrMRI600+ in-place.
# Uso:
#   PowerShell -ExecutionPolicy Bypass -File scripts/extract_cirrmri.ps1
#   (correr desde la raíz del proyecto cirrosis-detection)

$ErrorActionPreference = 'Stop'

$root = Join-Path $PSScriptRoot '..\data\CirrMRI600plus_raw'
$root = (Resolve-Path $root).Path
Write-Host "Extrayendo en: $root"

$zips = @(
    'Cirrhosis_T1_3D.zip',
    'Cirrhosis_T2_3D.zip',
    'Healthy_subjects.zip',
    'Metadata.zip'
    # NOTA: Cirrhosis_T2_2D.zip se ignora — usamos solo volúmenes 3D.
)

foreach ($z in $zips) {
    $src = Join-Path $root $z
    if (-not (Test-Path $src)) {
        Write-Warning "No existe $z; sáltalo si ya extraíste."
        continue
    }
    Write-Host "  -> $z"
    Expand-Archive -Path $src -DestinationPath $root -Force
}

Write-Host "Listo. Contenido final:"
Get-ChildItem -Path $root -Recurse -Depth 1 | Select-Object FullName
