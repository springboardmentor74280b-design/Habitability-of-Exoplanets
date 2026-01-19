# PowerShell script to POST to /predict using Invoke-RestMethod
# Usage: .\scripts\test_predict.ps1 -Url 'http://127.0.0.1:5000'
param(
    [string]$Url = 'http://127.0.0.1:5000'
)

$payload = @{
    HSI = 0.5
    planet_density = 5.5
    pl_eqt = 280
    pl_rade = 1.1
    pl_bmasse = 1.2
    st_teff = 3500
    star_luminosity = 0.02
    star_type_M = 1
    star_type_K = 0
    star_type_G = 0
} | ConvertTo-Json -Compress

try {
    $resp = Invoke-RestMethod -Uri "$Url/predict" -Method POST -ContentType 'application/json' -Body $payload -TimeoutSec 10
    $resp | ConvertTo-Json -Depth 3
} catch {
    Write-Error "Request failed: $_"
}
