# Simple sync script that works without Git installed
# This script will help you set up the connection to your GitHub repository

param(
    [string]$GitHubRepo = "https://github.com/brava-skin/Dean.git",
    [int]$IntervalSeconds = 30
)

Write-Host "🔄 Dean Auto-Sync Setup" -ForegroundColor Green
Write-Host "📁 Directory: $(Get-Location)" -ForegroundColor Blue
Write-Host "🌐 GitHub: $GitHubRepo" -ForegroundColor Cyan
Write-Host "⏱️  Check interval: $IntervalSeconds seconds" -ForegroundColor Yellow
Write-Host ""

# Check if Git is available
try {
    $gitVersion = git --version 2>$null
    if ($gitVersion) {
        Write-Host "✅ Git found: $gitVersion" -ForegroundColor Green
        
        # Check if this is a git repository
        if (Test-Path ".git") {
            Write-Host "✅ Git repository found" -ForegroundColor Green
            
            # Check if remote origin exists
            $remoteUrl = git remote get-url origin 2>$null
            if ($remoteUrl) {
                Write-Host "✅ Remote origin: $remoteUrl" -ForegroundColor Green
            } else {
                Write-Host "⚠️  No remote origin found. Adding..." -ForegroundColor Yellow
                git remote add origin $GitHubRepo
                Write-Host "✅ Remote origin added" -ForegroundColor Green
            }
            
            # Start monitoring
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
                        $commitMessage = "🔄 Auto-sync: $timestamp`n`nChanged files:`n$($changedFiles -join '`n')"
                        
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
                    
                    Start-Sleep -Seconds $IntervalSeconds
                } catch {
                    Write-Host "❌ Error: $($_.Exception.Message)" -ForegroundColor Red
                    Start-Sleep -Seconds $IntervalSeconds
                }
            }
        } else {
            Write-Host "⚠️  Not a Git repository. Initializing..." -ForegroundColor Yellow
            git init
            git remote add origin $GitHubRepo
            git add .
            git commit -m "Initial commit"
            git branch -M main
            git push -u origin main
            Write-Host "✅ Git repository initialized and connected to GitHub!" -ForegroundColor Green
        }
    } else {
        Write-Host "❌ Git not found. Please install Git first:" -ForegroundColor Red
        Write-Host "   1. Download Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
        Write-Host "   2. Install Git with default settings" -ForegroundColor Yellow
        Write-Host "   3. Restart your terminal" -ForegroundColor Yellow
        Write-Host "   4. Run this script again" -ForegroundColor Yellow
    }
} catch {
    Write-Host "❌ Error checking Git: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
}
