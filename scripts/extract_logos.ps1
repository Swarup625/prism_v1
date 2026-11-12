Add-Type -AssemblyName System.Drawing

$srcPath = "C:\Users\acer\.gemini\antigravity\brain\12f24986-e59b-4b66-82a0-c9f1f95480cd\.user_uploaded\media_1794476298270_49bcc92d.jpg"
$prismPath = "C:\Users\acer\.gemini\antigravity\brain\12f24986-e59b-4b66-82a0-c9f1f95480cd\.user_uploaded\media_1794477408396_35f6bed7.jpg"
$outDirFrontend = "D:\Trupti_programming\AI\prism_v1\frontend\assets\logos"
$outDirDocs = "D:\Trupti_programming\AI\prism_v1\docs\assets\logos"
$outDirAssetsFrontend = "D:\Trupti_programming\AI\prism_v1\frontend\assets"
$outDirAssetsDocs = "D:\Trupti_programming\AI\prism_v1\docs\assets"

$src = [System.Drawing.Bitmap]::FromFile($srcPath)

function Get-TightBounds($bmp, $startX, $startY, $width, $height) {
    $endX = [Math]::Min($bmp.Width, $startX + $width)
    $endY = [Math]::Min($bmp.Height, $startY + $height)

    $minX = $endX
    $maxX = $startX
    $minY = $endY
    $maxY = $startY

    for ($y = $startY; $y -lt $endY; $y++) {
        for ($x = $startX; $x -lt $endX; $x++) {
            $p = $bmp.GetPixel($x, $y)
            if ($p.R -lt 242 -or $p.G -lt 242 -or $p.B -lt 242) {
                if ($x -lt $minX) { $minX = $x }
                if ($x -gt $maxX) { $maxX = $x }
                if ($y -lt $minY) { $minY = $y }
                if ($y -gt $maxY) { $maxY = $y }
            }
        }
    }

    $pad = 6
    $minX = [Math]::Max(0, $minX - $pad)
    $minY = [Math]::Max(0, $minY - $pad)
    $maxX = [Math]::Min($bmp.Width - 1, $maxX + $pad)
    $maxY = [Math]::Min($bmp.Height - 1, $maxY + $pad)

    $w = $maxX - $minX + 1
    $h = $maxY - $minY + 1

    return [System.Drawing.Rectangle]::new($minX, $minY, $w, $h)
}

$regions = @(
    @{ key = "CSK";  col = 0; row = 0 },
    @{ key = "MI";   col = 1; row = 0 },
    @{ key = "RCB";  col = 2; row = 0 },
    @{ key = "KKR";  col = 3; row = 0 },
    @{ key = "GT";   col = 4; row = 0 },

    @{ key = "RR";   col = 0; row = 1 },
    @{ key = "DC";   col = 1; row = 1 },
    @{ key = "PBKS"; col = 2; row = 1 },
    @{ key = "LSG";  col = 3; row = 1 },
    @{ key = "SRH";  col = 4; row = 1 }
)

$colWidth = [int]($src.Width / 5)
$rowHeight = [int]($src.Height / 2)

foreach ($r in $regions) {
    $startX = $r.col * $colWidth
    $startY = $r.row * $rowHeight
    
    $rect = Get-TightBounds $src $startX $startY $colWidth $rowHeight
    Write-Host "$($r.key): X=$($rect.X), Y=$($rect.Y), W=$($rect.Width), H=$($rect.Height)"

    $cropped = New-Object System.Drawing.Bitmap($rect.Width, $rect.Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($cropped)
    $g.DrawImage($src, 0, 0, $rect, [System.Drawing.GraphicsUnit]::Pixel)
    $g.Dispose()

    for ($y = 0; $y -lt $cropped.Height; $y++) {
        for ($x = 0; $x -lt $cropped.Width; $x++) {
            $p = $cropped.GetPixel($x, $y)
            if ($p.R -gt 242 -and $p.G -gt 242 -and $p.B -gt 242) {
                $cropped.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(0, 255, 255, 255))
            }
        }
    }

    $outFrontend = Join-Path $outDirFrontend "$($r.key).png"
    $outDocs = Join-Path $outDirDocs "$($r.key).png"
    $cropped.Save($outFrontend, [System.Drawing.Imaging.ImageFormat]::Png)
    $cropped.Save($outDocs, [System.Drawing.Imaging.ImageFormat]::Png)
    $cropped.Dispose()
}

$src.Dispose()

# Now copy PRISM logo
Copy-Item -Path $prismPath -Destination (Join-Path $outDirAssetsFrontend "prism-logo.jpg") -Force
Copy-Item -Path $prismPath -Destination (Join-Path $outDirAssetsDocs "prism-logo.jpg") -Force

# Also create a transparent version of PRISM logo (since background is pure black, we can keep it as jpg or png)
$pImg = [System.Drawing.Bitmap]::FromFile($prismPath)
# For web, save as prism-logo.png
$pPng = New-Object System.Drawing.Bitmap($pImg.Width, $pImg.Height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
$pg = [System.Drawing.Graphics]::FromImage($pPng)
$pg.DrawImage($pImg, 0, 0)
$pg.Dispose()
# In prism-logo, black background can be blended using CSS mix-blend-mode: screen on dark mode!
# Or we can also make the black background transparent:
for ($y = 0; $y -lt $pPng.Height; $y++) {
    for ($x = 0; $x -lt $pPng.Width; $x++) {
        $p = $pPng.GetPixel($x, $y)
        # If very close to black
        if ($p.R -lt 12 -and $p.G -lt 12 -and $p.B -lt 12) {
            $pPng.SetPixel($x, $y, [System.Drawing.Color]::FromArgb(0, 0, 0, 0))
        }
    }
}
$pPng.Save((Join-Path $outDirAssetsFrontend "prism-logo.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$pPng.Save((Join-Path $outDirAssetsDocs "prism-logo.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$pPng.Dispose()
$pImg.Dispose()

Write-Host "Success: All logos and PRISM logo processed!"
