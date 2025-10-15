# Simple Dean Auto-Sync Script
# Connects to https://github.com/brava-skin/Dean

Write-Host "🔄 Dean Auto-Sync Starting..." -ForegroundColor Green
Write-Host "📁 Directory: $(Get-Location)" -ForegroundColor Blue
Write-Host "🌐 GitHub: https://github.com/brava-skin/Dean" -ForegroundColor Cyan
Write-Host ""

# Check if Git is available
try {
    $gitCheck = git --version 2>$null
    if ($gitCheck) {
        Write-Host "✅ Git found: $gitCheck" -ForegroundColor Green
        
        # Check if this is a git repository
        if (Test-Path ".git") {
            Write-Host "✅ Git repository found" -ForegroundColor Green
        } else {
            Write-Host "⚠️  Initializing Git repository..." -ForegroundColor Yellow
            git init
            git remote add origin https://github.com/brava-skin/Dean.git
            git add .
            git commit -m "Initial commit"
            git branch -M main
            Write-Host "✅ Git repository initialized!" -ForegroundColor Green
        }
        
        # Start monitoring loop
        Write-Host ""
        Write-Host "🚀 Starting auto-sync monitoring..." -ForegroundColor Green
        Write-Host "Press Ctrl+C to stop" -ForegroundColor Red
        Write-Host ""
        
        while ($true) {
            try {
                # Check for changes
                $status = git status --porcelain
                if ($status) {
                    Write-Host "🔄 Changes detected, syncing..." -ForegroundColor Yellow
                    
                    # Add all changes
                    git add .
                    
                    # Get changed files
                    $changedFiles = git diff --cached --name-only
                    $timestamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
                    $commitMessage = "🔄 Auto-sync: $timestamp"
                    
                    # Commit and push
                    git commit -m $commitMessage
                    if ($LASTEXITCODE -eq 0) {
                        git push origin main
                        if ($LASTEXITCODE -eq 0) {
                            Write-Host "✅ Successfully synced to GitHub!" -ForegroundColor Green
                        } else {
                            Write-Host "❌ Failed to push to GitHub" -ForegroundColor Red
                        }
                    } else {
                        Write-Host "⚠️  No changes to commit" -ForegroundColor Yellow
                    }
                } else {
                    Write-Host "📝 No changes detected at $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Gray
                }
                
                Start-Sleep -Seconds 30
            } catch {
                Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
                Start-Sleep -Seconds 30
            }
        }
    } else {
        Write-Host "❌ Git not found. Please install Git first:" -ForegroundColor Red
        Write-Host "   1. Download from: https://git-scm.com/download/win" -ForegroundColor Yellow
        Write-Host "   2. Install with default settings" -ForegroundColor Yellow
        Write-Host "   3. Restart terminal and try again" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
}
