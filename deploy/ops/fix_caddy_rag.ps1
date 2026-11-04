$ErrorActionPreference = "Stop"
$p = "D:\pet-project\tk-assistant\deploy\Caddyfile"
$t = [System.IO.File]::ReadAllText($p)
if ($t -like "*reverse_proxy rag:8000*") {
    Write-Output "ALREADY RAG"
} else {
    $anchor = "reverse_proxy llama-server:8080"
    if ($t.Contains($anchor) -eq $false) { throw "anchor not found" }
    $t = $t.Replace($anchor, "reverse_proxy rag:8000")
    [System.IO.File]::WriteAllText($p, $t, [System.Text.UTF8Encoding]::new($false))
    Write-Output "PATCHED"
}
