$ErrorActionPreference = "Stop"
$p = "D:\pet-project\tk-assistant\deploy\docker-compose.yml"
$t = Get-Content $p -Raw -Encoding UTF8
if ($t -like "*root/.cache/huggingface*") {
    Write-Output "ALREADY PATCHED"
} else {
    $anchor = "      - ../data/processed/tk_articles.json:/app/data/tk_articles.json:ro"
    if ($t.Contains($anchor) -eq $false) { throw "anchor not found" }
    $add = "      - D:\models\hf:/root/.cache/huggingface:ro"
    $t = $t.Replace($anchor, "$anchor`r`n$add")
    Set-Content -Path $p -Value $t -Encoding UTF8 -NoNewline
    Write-Output "PATCHED"
}
