$ErrorActionPreference = "Stop"
$p = "D:\pet-project\tk-assistant\deploy\docker-compose.yml"
$t = Get-Content $p -Raw -Encoding UTF8
if ($t -like "*e6v2_lora-f16.gguf*") {
    Write-Output "ALREADY PATCHED"
} else {
    $anchor = "-m /models/qwen2.5-7b-tk-q4_k_m.gguf"
    if ($t.Contains($anchor) -eq $false) { throw "anchor not found" }
    $add = "-m /models/qwen2.5-7b-instruct-q4_k_m.gguf`r`n      -l /models/e6v2_lora-f16.gguf"
    $t = $t.Replace($anchor, $add)
    Set-Content -Path $p -Value $t -Encoding UTF8 -NoNewline
    Write-Output "PATCHED"
}
