$ErrorActionPreference = "Stop"
$p = "D:\pet-project\tk-assistant\deploy\docker-compose.yml"
$t = Get-Content $p -Raw -Encoding UTF8
$old = "-m /models/qwen2.5-7b-instruct-q4_k_m.gguf"
$new = "-m /models/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
if ($t -like "*$new*") {
    Write-Output "ALREADY PATCHED"
} elseif ($t.Contains($old)) {
    $t = $t.Replace($old, $new)
    Set-Content -Path $p -Value $t -Encoding UTF8 -NoNewline
    Write-Output "PATCHED"
} else {
    throw "anchor not found"
}
