#!/bin/bash

# Dean macOS Setup Script
# This script sets up Dean on macOS with optimal configurations

set -e

echo "🍎 Setting up Dean on macOS..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS only"
    exit 1
fi

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo "📦 Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
else
    echo "✅ Homebrew already installed"
fi

# Install Python 3.9+ if needed
if ! command -v python3 &> /dev/null || ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 9) else 1)" 2>/dev/null; then
    echo "🐍 Installing Python 3.9+ via Homebrew..."
    brew install python@3.11
else
    echo "✅ Python 3.9+ already installed"
fi

# Install SQLite (usually comes with macOS, but ensure it's up to date)
if ! command -v sqlite3 &> /dev/null; then
    echo "🗄️ Installing SQLite..."
    brew install sqlite
else
    echo "✅ SQLite already installed"
fi

# Create virtual environment
echo "🔧 Setting up virtual environment..."
cd "$(dirname "$0")/.."
python3 -m venv venv
source venv/bin/activate

# Install dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/digests
mkdir -p data/snapshots
mkdir -p logs

# Set up log rotation (macOS specific)
echo "📋 Setting up log rotation..."
cat > rotate_logs.sh << 'EOF'
#!/bin/bash
# Rotate logs weekly
find logs/ -name "*.log" -mtime +7 -delete
find data/digests/ -name "*.jsonl" -mtime +30 -delete
EOF

chmod +x rotate_logs.sh

# Create launchd service template
echo "🚀 Creating launchd service template..."
mkdir -p ~/Library/LaunchAgents
cat > ~/Library/LaunchAgents/com.dean.automation.plist.template << EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.dean.automation</string>
    <key>ProgramArguments</key>
    <array>
        <string>$(which python3)</string>
        <string>$(pwd)/src/main.py</string>
        <string>--profile</string>
        <string>production</string>
    </array>
    <key>StartInterval</key>
    <integer>7200</integer> <!-- 2 hours in seconds -->
    <key>WorkingDirectory</key>
    <string>$(pwd)</string>
    <key>StandardOutPath</key>
    <string>$(pwd)/logs/automation.log</string>
    <key>StandardErrorPath</key>
    <string>$(pwd)/logs/automation_error.log</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
</dict>
</plist>
EOF

echo ""
echo "✅ macOS setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Create your .env file with API credentials"
echo "2. Configure your settings in config/settings.yaml"
echo "3. Test the installation: python src/main.py --dry-run"
echo "4. To enable automation:"
echo "   - Edit ~/Library/LaunchAgents/com.dean.automation.plist.template"
echo "   - Rename to com.dean.automation.plist"
echo "   - Run: launchctl load ~/Library/LaunchAgents/com.dean.automation.plist"
echo ""
echo "🔧 Useful commands:"
echo "   - Activate venv: source venv/bin/activate"
echo "   - Test API: python -c \"from src.meta_client import MetaClient; print('API ready')\""
echo "   - Check logs: tail -f logs/automation.log"
echo ""
echo "📚 For detailed setup instructions, see docs/INSTALLATION.md"
