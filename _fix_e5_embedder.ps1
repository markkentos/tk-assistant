$ErrorActionPreference = "Stop"
$p = "D:\pet-project\tk-assistant\deploy\docker-compose.yml"
$t = Get-Content $p -Raw -Encoding UTF8
if ($t -like "*EMBEDDER_MODEL*") {
    Write-Output "ALREADY PATCHED"
} else {
    $anchor = "      - LLAMA_SERVER_URL=http://llama-server:8080"
    if ($t.Contains($anchor) -eq $false) { throw "anchor not found" }
    $add = "      - EMBEDDER_MODEL=intfloat/multilingual-e5-base"
    $t = $t.Replace($anchor, "$anchor`r`n$add")
    Set-Content -Path $p -Value $t -Encoding UTF8 -NoNewline
    Write-Output "PATCHED"
}
