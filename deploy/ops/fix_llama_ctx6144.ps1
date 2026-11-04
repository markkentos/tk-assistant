$ErrorActionPreference = "Stop"
$p = "D:\pet-project\tk-assistant\deploy\docker-compose.yml"
$t = [System.IO.File]::ReadAllText($p)
$old = "-c 4096 -ngl 99"
$new = "-c 6144 -ngl 99"
if ($t -like "*$new*") {
    Write-Output "ALREADY PATCHED"
} elseif ($t.Contains($old)) {
    $t = $t.Replace($old, $new)
    [System.IO.File]::WriteAllText($p, $t, [System.Text.UTF8Encoding]::new($false))
    Write-Output "PATCHED"
} else {
    throw "anchor not found"
}
