# 🔄 Auto-Sync with GitHub

This guide explains how to set up automatic synchronization between your local Cursor workspace and your GitHub repository.

## 🚀 Quick Start

### 1. **Initial Setup** (One-time)
```bash
# If you haven't initialized git yet:
git init
git remote add origin YOUR_GITHUB_REPO_URL
git add .
git commit -m "Initial commit"
git push -u origin main
```

### 2. **Start Auto-Sync**
```bash
# Option 1: Double-click auto-sync.bat
# Option 2: Run in terminal
auto-sync.bat
```

That's it! Your changes will now automatically sync to GitHub every 30 seconds.

## 📋 How It Works

### **Auto-Sync Process**
1. **File Monitoring**: Watches for file changes every 30 seconds
2. **Auto-Commit**: Automatically commits changes with timestamp
3. **Auto-Push**: Pushes commits to your GitHub repository
4. **Status Updates**: Shows sync status in the terminal

### **What Gets Synced**
- ✅ All file changes in your project
- ✅ New files and directories
- ✅ File deletions
- ✅ Configuration changes
- ❌ Files in `.gitignore` (excluded)

## 🛠️ Configuration Options

### **Change Sync Frequency**
Edit `auto-sync.ps1` and modify the `IntervalSeconds` parameter:
```powershell
# Default: 30 seconds
powershell -File auto-sync.ps1 -IntervalSeconds 60  # 1 minute
powershell -File auto-sync.ps1 -IntervalSeconds 10  # 10 seconds
```

### **Change Branch**
```powershell
# Default: main branch
powershell -File auto-sync.ps1 -Branch develop
```

### **Verbose Mode**
```powershell
# Show detailed sync information
powershell -File auto-sync.ps1 -Verbose
```

## 📁 File Structure

```
dean/
├── auto-sync.ps1          # PowerShell sync script
├── auto-sync.bat           # Windows batch launcher
├── setup-git-sync.bat      # One-time setup script
├── .gitignore              # Git ignore rules
└── README_SYNC.md          # This file
```

## 🔧 Manual Commands

### **Start Auto-Sync**
```bash
# Simple start
auto-sync.bat

# With custom settings
powershell -File auto-sync.ps1 -Branch main -IntervalSeconds 30 -Verbose
```

### **Stop Auto-Sync**
- Press `Ctrl+C` in the terminal
- Close the terminal window

### **Manual Sync** (One-time)
```bash
git add .
git commit -m "Manual sync: $(Get-Date)"
git push origin main
```

## 🚨 Troubleshooting

### **"Git repository not found"**
```bash
# Initialize git repository
git init
git remote add origin YOUR_GITHUB_REPO_URL
```

### **"No remote origin found"**
```bash
# Add your GitHub repository
git remote add origin https://github.com/yourusername/yourrepo.git
```

### **"Failed to push to GitHub"**
- Check your internet connection
- Verify GitHub credentials
- Ensure you have push permissions to the repository

### **PowerShell Execution Policy Error**
```bash
# Run this command once:
powershell -Command "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force"
```

### **Auto-sync not detecting changes**
- Check if files are in `.gitignore`
- Verify file permissions
- Try manual sync to test

## 📊 Sync Status Messages

### **Success Messages**
- `✅ Successfully synced to GitHub!` - Changes pushed successfully
- `📝 No changes detected` - No changes to sync
- `🔄 Changes detected, syncing...` - Changes found, starting sync

### **Error Messages**
- `❌ Failed to push to GitHub` - Push failed, check connection
- `⚠️ No changes to commit` - No changes staged
- `❌ Error during sync` - General sync error

## 🔒 Security Notes

### **What's Excluded**
The following files are automatically excluded from sync:
- Environment files (`.env`, `.env.local`)
- Database files (`*.sqlite`, `*.db`)
- Log files (`*.log`)
- Temporary files (`*.tmp`, `*.temp`)
- IDE files (`.vscode/`, `.idea/`)
- OS files (`.DS_Store`, `Thumbs.db`)

### **Sensitive Data**
- Never commit API keys or passwords
- Use `.env` files for sensitive configuration
- Review `.gitignore` for your specific needs

## 🎯 Best Practices

### **Commit Messages**
Auto-sync creates descriptive commit messages:
```
🔄 Auto-sync: 2024-01-15 14:30:25

Changed files:
src/main.py
config/settings.yaml
README.md
```

### **File Organization**
- Keep related files together
- Use descriptive filenames
- Organize with folders when needed

### **Sync Frequency**
- **30 seconds**: Good for active development
- **60 seconds**: Balanced approach
- **10 seconds**: For very active editing (may be resource-intensive)

## 🔄 Alternative Methods

### **VS Code Git Integration**
If you prefer VS Code's built-in Git:
1. Open VS Code
2. Enable auto-save
3. Use VS Code's Git panel for commits
4. Push manually or set up auto-push

### **GitHub Desktop**
1. Install GitHub Desktop
2. Open your repository
3. Enable auto-commit in settings
4. Changes sync automatically

### **Command Line Git**
```bash
# Watch for changes and auto-commit
git add .
git commit -m "Auto-sync: $(date)"
git push origin main
```

## 📞 Support

### **Common Issues**
1. **Sync not working**: Check internet connection and GitHub credentials
2. **Files not syncing**: Verify they're not in `.gitignore`
3. **Permission errors**: Ensure you have push access to the repository
4. **PowerShell errors**: Run the execution policy command

### **Getting Help**
- Check the terminal output for error messages
- Verify your GitHub repository URL
- Ensure you have the latest version of Git
- Test manual git commands first

---

**🎉 Your Dean project will now automatically sync with GitHub every 30 seconds!**

Just make changes in Cursor, and they'll appear on GitHub automatically. No more manual commits or pushes needed!
