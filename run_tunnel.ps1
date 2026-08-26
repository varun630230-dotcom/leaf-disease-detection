for ($i = 0; $i -lt 1000; $i++) {
    Write-Host "Starting localtunnel tunnel (attempt $i)..."
    npx localtunnel --port 5173
    Start-Sleep -Seconds 2
}
