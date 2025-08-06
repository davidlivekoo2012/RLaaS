# PowerShell script to fix line endings in the RLaaS project

param(
    [switch]$DryRun = $false
)

Write-Host "RLaaS Line Ending Fix Script" -ForegroundColor Green
Write-Host "=============================" -ForegroundColor Green

if ($DryRun) {
    Write-Host "Running in DRY RUN mode - no files will be modified" -ForegroundColor Yellow
}

# File extensions that should have LF line endings
$LFExtensions = @(
    "*.py", "*.md", "*.txt", "*.yml", "*.yaml", "*.json", 
    "*.toml", "*.cfg", "*.ini", "*.sh", "*.bash", "*.sql",
    "Dockerfile", "Makefile", ".gitignore", ".gitattributes",
    ".editorconfig", ".env*"
)

# File extensions that should have CRLF line endings
$CRLFExtensions = @(
    "*.bat", "*.cmd", "*.ps1"
)

function Convert-LineEndings {
    param(
        [string]$FilePath,
        [string]$TargetEnding
    )
    
    if (-not (Test-Path $FilePath)) {
        Write-Warning "File not found: $FilePath"
        return
    }
    
    try {
        $content = Get-Content -Path $FilePath -Raw
        
        if ($null -eq $content) {
            Write-Host "  Skipping empty file: $FilePath" -ForegroundColor Gray
            return
        }
        
        # Detect current line endings
        $hasCRLF = $content.Contains("`r`n")
        $hasLF = $content.Contains("`n") -and -not $content.Contains("`r`n")
        
        $currentEnding = if ($hasCRLF) { "CRLF" } elseif ($hasLF) { "LF" } else { "None" }
        
        if ($currentEnding -eq $TargetEnding -or $currentEnding -eq "None") {
            Write-Host "  OK: $FilePath ($currentEnding)" -ForegroundColor Green
            return
        }
        
        Write-Host "  Converting: $FilePath ($currentEnding -> $TargetEnding)" -ForegroundColor Yellow
        
        if (-not $DryRun) {
            # Normalize to LF first
            $normalizedContent = $content -replace "`r`n", "`n" -replace "`r", "`n"
            
            # Convert to target ending
            if ($TargetEnding -eq "CRLF") {
                $normalizedContent = $normalizedContent -replace "`n", "`r`n"
            }
            
            # Write back to file with UTF-8 encoding (no BOM)
            $utf8NoBom = New-Object System.Text.UTF8Encoding $false
            [System.IO.File]::WriteAllText($FilePath, $normalizedContent, $utf8NoBom)
        }
    }
    catch {
        Write-Error "Failed to process file: $FilePath - $($_.Exception.Message)"
    }
}

function Process-Files {
    param(
        [string[]]$Patterns,
        [string]$TargetEnding
    )
    
    Write-Host "`nProcessing files for $TargetEnding line endings..." -ForegroundColor Cyan
    
    foreach ($pattern in $Patterns) {
        Write-Host "  Pattern: $pattern" -ForegroundColor Gray
        
        $files = Get-ChildItem -Path . -Recurse -Include $pattern -File | 
                 Where-Object { 
                     $_.FullName -notmatch "\\\.git\\" -and 
                     $_.FullName -notmatch "\\__pycache__\\" -and
                     $_.FullName -notmatch "\\node_modules\\" -and
                     $_.FullName -notmatch "\\\.venv\\" -and
                     $_.FullName -notmatch "\\venv\\" -and
                     $_.FullName -notmatch "\\build\\" -and
                     $_.FullName -notmatch "\\dist\\"
                 }
        
        foreach ($file in $files) {
            Convert-LineEndings -FilePath $file.FullName -TargetEnding $TargetEnding
        }
    }
}

# Main execution
Write-Host "`nStarting line ending conversion..." -ForegroundColor Blue

# Process LF files
Process-Files -Patterns $LFExtensions -TargetEnding "LF"

# Process CRLF files
Process-Files -Patterns $CRLFExtensions -TargetEnding "CRLF"

Write-Host "`nLine ending conversion completed!" -ForegroundColor Green

if ($DryRun) {
    Write-Host "`nTo actually fix the files, run without -DryRun parameter:" -ForegroundColor Yellow
    Write-Host "  .\scripts\fix-line-endings.ps1" -ForegroundColor White
}
else {
    Write-Host "`nNext steps:" -ForegroundColor Cyan
    Write-Host "1. Review the changes with: git diff" -ForegroundColor White
    Write-Host "2. Stage the changes with: git add ." -ForegroundColor White
    Write-Host "3. Commit the changes with: git commit -m 'Fix line endings'" -ForegroundColor White
}
