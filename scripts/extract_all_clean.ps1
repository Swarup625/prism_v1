Add-Type -AssemblyName System.Drawing

$teamsSrcPath = "C:\Users\acer\.gemini\antigravity\brain\12f24986-e59b-4b66-82a0-c9f1f95480cd\.user_uploaded\media_1794476298270_49bcc92d.jpg"
$prismSrcPath = "C:\Users\acer\.gemini\antigravity\brain\12f24986-e59b-4b66-82a0-c9f1f95480cd\.user_uploaded\media_1794477408396_35f6bed7.jpg"

$frontendLogosDir = "D:\Trupti_programming\AI\prism_v1\frontend\assets\logos"
$docsLogosDir     = "D:\Trupti_programming\AI\prism_v1\docs\assets\logos"
$frontendAssetsDir= "D:\Trupti_programming\AI\prism_v1\frontend\assets"
$docsAssetsDir    = "D:\Trupti_programming\AI\prism_v1\docs\assets"

# Helper for outer background removal (BFS flood fill from image borders)
function Remove-OuterWhite($bmp) {
    [int]$w = $bmp.Width
    [int]$h = $bmp.Height
    $visited = New-Object 'bool[,]' $w, $h
    $queue = New-Object System.Collections.Generic.Queue[System.Drawing.Point]

    function Is-WhiteBg($col) {
        return ($col.R -ge 238 -and $col.G -ge 238 -and $col.B -ge 238)
    }

    $lastY = $h - 1
    $lastX = $w - 1

    for ($x = 0; $x -lt $w; $x++) {
        if (Is-WhiteBg ($bmp.GetPixel($x, 0))) {
            $queue.Enqueue([System.Drawing.Point]::new($x, 0))
            $visited[$x, 0] = $true
        }
        if (Is-WhiteBg ($bmp.GetPixel($x, $lastY))) {
            $queue.Enqueue([System.Drawing.Point]::new($x, $lastY))
            $visited[$x, $lastY] = $true
        }
    }
    for ($y = 0; $y -lt $h; $y++) {
        if (Is-WhiteBg ($bmp.GetPixel(0, $y)) -and -not $visited[0, $y]) {
            $queue.Enqueue([System.Drawing.Point]::new(0, $y))
            $visited[0, $y] = $true
        }
        if (Is-WhiteBg ($bmp.GetPixel($lastX, $y)) -and -not $visited[$lastX, $y]) {
            $queue.Enqueue([System.Drawing.Point]::new($lastX, $y))
            $visited[$lastX, $y] = $true
        }
    }

    $dx = @(0, 0, 1, -1)
    $dy = @(1, -1, 0, 0)

    while ($queue.Count -gt 0) {
        $pt = $queue.Dequeue()
        $bmp.SetPixel($pt.X, $pt.Y, [System.Drawing.Color]::FromArgb(0, 255, 255, 255))

        for ($i = 0; $i -lt 4; $i++) {
            $nx = $pt.X + $dx[$i]
            $ny = $pt.Y + $dy[$i]
            if ($nx -ge 0 -and $nx -lt $w -and $ny -ge 0 -and $ny -lt $h) {
                if (-not $visited[$nx, $ny]) {
                    $visited[$nx, $ny] = $true
                    if (Is-WhiteBg ($bmp.GetPixel($nx, $ny))) {
                        $queue.Enqueue([System.Drawing.Point]::new($nx, $ny))
                    }
                }
            }
        }
    }
}

# 1. Process all 10 IPL team logos
$teamsBmp = [System.Drawing.Bitmap]::FromFile($teamsSrcPath)

$regions = @(
    @{ key = "CSK";  searchMinX = 0;   searchMaxX = 230;  searchMinY = 40;  searchMaxY = 265 },
    @{ key = "MI";   searchMinX = 231; searchMaxX = 450;  searchMinY = 60;  searchMaxY = 255 },
    @{ key = "RCB";  searchMinX = 452; searchMaxX = 614;  searchMinY = 20;  searchMaxY = 268 },
    @{ key = "KKR";  searchMinX = 615; searchMaxX = 798;  searchMinY = 20;  searchMaxY = 275 },
    @{ key = "GT";   searchMinX = 800; searchMaxX = 1023; searchMinY = 40;  searchMaxY = 265 },

    @{ key = "RR";   searchMinX = 0;   searchMaxX = 206;  searchMinY = 280; searchMaxY = 500 },
    @{ key = "DC";   searchMinX = 207; searchMaxX = 398;  searchMinY = 270; searchMaxY = 500 },
    @{ key = "PBKS"; searchMinX = 400; searchMaxX = 566;  searchMinY = 280; searchMaxY = 500 },
    @{ key = "LSG";  searchMinX = 568; searchMaxX = 796;  searchMinY = 300; searchMaxY = 490 },
    @{ key = "SRH";  searchMinX = 798; searchMaxX = 1023; searchMinY = 280; searchMaxY = 500 }
)

foreach ($r in $regions) {
    $minX = $r.searchMaxX; $maxX = $r.searchMinX
    $minY = $r.searchMaxY; $maxY = $r.searchMinY

    for ($y = $r.searchMinY; $y -le $r.searchMaxY; $y++) {
        for ($x = $r.searchMinX; $x -le $r.searchMaxX; $x++) {
            $p = $teamsBmp.GetPixel($x, $y)
            if ($p.R -lt 242 -or $p.G -lt 242 -or $p.B -lt 242) {
                if ($x -lt $minX) { $minX = $x }
                if ($x -gt $maxX) { $maxX = $x }
                if ($y -lt $minY) { $minY = $y }
                if ($y -gt $maxY) { $maxY = $y }
            }
        }
    }

    $pad = 3
    $cropX = [Math]::Max(0, $minX - $pad)
    $cropY = [Math]::Max(0, $minY - $pad)
    $cropW = [Math]::Min($teamsBmp.Width - $cropX, ($maxX - $minX + 1) + ($pad * 2))
    $cropH = [Math]::Min($teamsBmp.Height - $cropY, ($maxY - $minY + 1) + ($pad * 2))

    $rect = [System.Drawing.Rectangle]::new($cropX, $cropY, $cropW, $cropH)
    Write-Host "$($r.key): Cropping rect X=$cropX, Y=$cropY, W=$cropW, H=$cropH"

    $cropped = New-Object System.Drawing.Bitmap($cropW, $cropH, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($cropped)
    $g.DrawImage($teamsBmp, 0, 0, $rect, [System.Drawing.GraphicsUnit]::Pixel)
    $g.Dispose()

    Remove-OuterWhite $cropped

    $cropped.Save((Join-Path $frontendLogosDir "$($r.key).png"), [System.Drawing.Imaging.ImageFormat]::Png)
    $cropped.Save((Join-Path $docsLogosDir "$($r.key).png"), [System.Drawing.Imaging.ImageFormat]::Png)
    $cropped.Dispose()
}

$teamsBmp.Dispose()
Write-Host "All 10 IPL team logos cleanly extracted and saved!"

# 2. Process PRISM Logo Variants
$prismBmp = [System.Drawing.Bitmap]::FromFile($prismSrcPath)

# A. Prism Symbol Only (X: 45 to 352, Y: 33 to 310)
$symRect = [System.Drawing.Rectangle]::new(45, 33, 308, 278)
$symBmp = New-Object System.Drawing.Bitmap($symRect.Width, $symRect.Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$sg = [System.Drawing.Graphics]::FromImage($symBmp)
$sg.DrawImage($prismBmp, 0, 0, $symRect, [System.Drawing.GraphicsUnit]::Pixel)
$sg.Dispose()

for ($y = 0; $y -lt $symBmp.Height; $y++) {
    for ($x = 0; $x -lt $symBmp.Width; $x++) {
        $p = $symBmp.GetPixel($x, $y)
        if ($p.R -lt 22 -and $p.G -lt 22 -and $p.B -lt 22) {
            $symBmp.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(0, 0, 0, 0))
        }
    }
}
$symBmp.Save((Join-Path $frontendAssetsDir "prism-symbol.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$symBmp.Save((Join-Path $docsAssetsDir "prism-symbol.png"), [System.Drawing.Imaging.ImageFormat]::Png)

# 32x32 Favicon
$fav32 = New-Object System.Drawing.Bitmap(32, 32, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$fg = [System.Drawing.Graphics]::FromImage($fav32)
$fg.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
$fg.DrawImage($symBmp, 0, 0, 32, 32)
$fg.Dispose()
$fav32.Save((Join-Path $frontendAssetsDir "prism-favicon.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$fav32.Save((Join-Path $docsAssetsDir "prism-favicon.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$fav32.Dispose()

$symBmp.Dispose()

# B. Prism Full Logo + Tagline (X: 45 to 995, Y: 33 to 310)
$fullRect = [System.Drawing.Rectangle]::new(45, 33, 951, 278)
$fullBmp = New-Object System.Drawing.Bitmap($fullRect.Width, $fullRect.Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$fullG = [System.Drawing.Graphics]::FromImage($fullBmp)
$fullG.DrawImage($prismBmp, 0, 0, $fullRect, [System.Drawing.GraphicsUnit]::Pixel)
$fullG.Dispose()

for ($y = 0; $y -lt $fullBmp.Height; $y++) {
    for ($x = 0; $x -lt $fullBmp.Width; $x++) {
        $p = $fullBmp.GetPixel($x, $y)
        if ($p.R -lt 22 -and $p.G -lt 22 -and $p.B -lt 22) {
            $fullBmp.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(0, 0, 0, 0))
        }
    }
}
$fullBmp.Save((Join-Path $frontendAssetsDir "prism-full-logo.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$fullBmp.Save((Join-Path $docsAssetsDir "prism-full-logo.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$fullBmp.Dispose()

$prismBmp.Dispose()
Write-Host "PRISM Symbol, 32x32 Favicon, and Full Logo extracted successfully!"
