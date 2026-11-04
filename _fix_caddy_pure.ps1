$ErrorActionPreference = "Stop"
$p = "D:\pet-project\tk-assistant\deploy\Caddyfile"
$t = [System.IO.File]::ReadAllText($p)
if ($t -like "*llama-server:8080*") {
    Write-Output "ALREADY PATCHED"
} else {
    $anchor = "reverse_proxy rag:8000"
    if ($t.Contains($anchor) -eq $false) { throw "anchor not found" }
    $t = $t.Replace($anchor, "reverse_proxy llama-server:8080")
    [System.IO.File]::WriteAllText($p, $t, [System.Text.UTF8Encoding]::new($false))
    Write-Output "PATCHED"
}
