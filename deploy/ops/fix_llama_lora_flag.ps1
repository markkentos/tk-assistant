$ErrorActionPreference = "Stop"
$p = "D:\pet-project\tk-assistant\deploy\docker-compose.yml"
$t = Get-Content $p -Raw -Encoding UTF8
if ($t -like "*--lora /models/e6v2_lora-f16.gguf*") {
    Write-Output "ALREADY PATCHED"
} else {
    $anchor = "-l /models/e6v2_lora-f16.gguf"
    if ($t.Contains($anchor) -eq $false) { throw "anchor not found" }
    $t = $t.Replace($anchor, "--lora /models/e6v2_lora-f16.gguf")
    Set-Content -Path $p -Value $t -Encoding UTF8 -NoNewline
    Write-Output "PATCHED"
}
