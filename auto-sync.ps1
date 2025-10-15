# Auto-sync script for Windows PowerShell
# This script monitors file changes and automatically commits and pushes to GitHub

param(
    [string]$Branch = "main",
    [int]$IntervalSeconds = 30,
    [switch]$Verbose
)

Write-Host "🔄 Starting auto-sync for branch: $Branch" -ForegroundColor Green
Write-Host "📁 Monitoring directory: $(Get-Location)" -ForegroundColor Blue
Write-Host "⏱️  Check interval: $IntervalSeconds seconds" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop" -ForegroundColor Red

# Function to check if there are changes
function Test-GitChanges {
    $status = git status --porcelain
    return $status.Length -gt 0
}

# Function to commit and push changes
function Sync-Changes {
    try {
        # Add all changes
        git add .
        
        # Get list of changed files
        $changedFiles = git diff --cached --name-only
        
        if ($changedFiles) {
            # Create commit message with changed files
            $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
            $commitMessage = "🔄 Auto-sync: $timestamp`n`nChanged files:`n$($changedFiles -join '`n')"
            
            # Commit changes
            git commit -m $commitMessage
            
            if ($LASTEXITCODE -eq 0) {
                # Push to GitHub
                git push origin $Branch
                
                if ($LASTEXITCODE -eq 0) {
                    Write-Host "✅ Successfully synced to GitHub!" -ForegroundColor Green
                    if ($Verbose) {
                        Write-Host "📝 Commit: $commitMessage" -ForegroundColor Cyan
                    }
                } else {
                    Write-Host "❌ Failed to push to GitHub" -ForegroundColor Red
                }
            } else {
                Write-Host "⚠️  No changes to commit" -ForegroundColor Yellow
            }
        } else {
            Write-Host "📝 No changes detected" -ForegroundColor Gray
        }
    } catch {
        Write-Host "❌ Error during sync: $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Main monitoring loop
while ($true) {
    try {
        if (Test-GitChanges) {
            Write-Host "🔄 Changes detected, syncing..." -ForegroundColor Yellow
            Sync-Changes
        } else {
            if ($Verbose) {
                Write-Host "📝 No changes detected at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
            }
        }
        
        Start-Sleep -Seconds $IntervalSeconds
    } catch {
        Write-Host "❌ Error in monitoring loop: $($_.Exception.Message)" -ForegroundColor Red
        Start-Sleep -Seconds $IntervalSeconds
    }
}