#!/bin/bash

# Dean Final DigitalOcean Deployment Script
# Complete setup with all optimizations

set -e

echo "🚀 Dean ML-Enhanced Automation - Final DigitalOcean Deployment"
echo "=============================================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Please run as root: sudo bash deploy_final.sh"
    exit 1
fi

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install dependencies
echo "🐍 Installing Python 3.11 and dependencies..."
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip
apt install -y git curl wget build-essential libssl-dev libffi-dev
apt install -y htop nano vim jq

# Create dean user
echo "👤 Creating dean user..."
useradd -m -s /bin/bash dean || true
usermod -aG sudo dean || true

# Create project directory
echo "📁 Setting up project directory..."
mkdir -p /opt/dean
chown dean:dean /opt/dean

# Clone repository
echo "📥 Cloning repository..."
cd /opt/dean
if [ ! -d ".git" ]; then
    sudo -u dean git clone https://github.com/brava-skin/Dean.git .
else
    sudo -u dean git pull
fi

# Create virtual environment
echo "🔧 Setting up Python virtual environment..."
sudo -u dean python3.11 -m venv /opt/dean/venv
sudo -u dean /opt/dean/venv/bin/pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
sudo -u dean /opt/dean/venv/bin/pip install -r /opt/dean/requirements.txt

# Create environment file
echo "🔐 Creating environment file..."
sudo -u dean tee /opt/dean/.env > /dev/null <<'EOF'
# Meta Ads API Configuration
FB_APP_ID=your_app_id_here
FB_APP_SECRET=your_app_secret_here
FB_ACCESS_TOKEN=your_access_token_here
FB_AD_ACCOUNT_ID=your_ad_account_id_here
FB_PIXEL_ID=your_pixel_id_here
FB_PAGE_ID=your_page_id_here
IG_ACTOR_ID=your_ig_actor_id_here

# Supabase Configuration
SUPABASE_URL=your_supabase_url_here
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key_here
SUPABASE_TABLE=meta_creatives

# Business Configuration
STORE_URL=your_store_url_here
SLACK_WEBHOOK_URL=your_slack_webhook_here
BREAKEVEN_CPA=your_breakeven_cpa_here
COGS_PER_PURCHASE=your_cogs_here
USD_EUR_RATE=your_exchange_rate_here

# ML Configuration
ML_MODE=true
ML_LEARNING_RATE=0.01
ML_CONFIDENCE_THRESHOLD=0.7

# Rate Limiting Configuration (24/7 UI PROTECTION)
META_REQUEST_DELAY=2.0
META_PEAK_HOURS_DELAY=3.0
META_NIGHT_HOURS_DELAY=1.5
META_MAX_CONCURRENT_INSIGHTS=1
META_RETRY_MAX=12
META_BACKOFF_BASE=2.0
META_USAGE_THRESHOLD=0.6
META_EMERGENCY_THRESHOLD=0.8
META_UI_PROTECTION_MODE=true
META_BUC_ENABLED=true

# Timezone
TZ=Europe/Amsterdam
EOF

# Create systemd service
echo "⚙️ Creating systemd service..."
tee /etc/systemd/system/dean-automation.service > /dev/null <<EOF
[Unit]
Description=Dean ML-Enhanced Meta Ads Automation
After=network.target

[Service]
Type=simple
User=dean
Group=dean
WorkingDirectory=/opt/dean
Environment=PATH=/opt/dean/venv/bin
EnvironmentFile=/opt/dean/.env
ExecStart=/opt/dean/venv/bin/python /opt/dean/src/main.py --profile production --continuous-mode
Restart=always
RestartSec=30
StandardOutput=journal
StandardError=journal

# Resource limits
MemoryMax=512M
CPUQuota=50%

[Install]
WantedBy=multi-user.target
EOF

# Create continuous operation script
echo "🔄 Creating continuous operation script..."
sudo -u dean tee /opt/dean/run_continuous.py > /dev/null <<'EOF'
#!/usr/bin/env python3
"""
Dean Continuous Operation Script
Optimized for DigitalOcean with advanced rate limiting
"""

import time
import logging
import signal
import sys
from datetime import datetime, timedelta
import subprocess
import os

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/dean/logs/dean_continuous.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DeanContinuousRunner:
    def __init__(self):
        self.running = True
        self.last_run = None
        self.run_interval = 300  # 5 minutes between runs (24/7)
        self.peak_hours_interval = 180  # 3 minutes during peak hours (9 AM - 6 PM)
        self.off_peak_interval = 300    # 5 minutes during off-peak hours
        self.night_interval = 600       # 10 minutes during night hours (12 AM - 6 AM)
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        
    def run_dean_cycle(self):
        """Run a single Dean automation cycle"""
        try:
            logger.info("🔄 Starting Dean automation cycle...")
            
            # Change to project directory
            os.chdir('/opt/dean')
            
            # Run Dean with optimized settings
            result = subprocess.run([
                '/opt/dean/venv/bin/python', 
                '/opt/dean/src/main.py',
                '--profile', 'production',
                '--continuous-mode'
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                logger.info("✅ Dean cycle completed successfully")
                return True
            else:
                logger.error(f"❌ Dean cycle failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ Dean cycle timed out")
            return False
        except Exception as e:
            logger.error(f"💥 Dean cycle error: {e}")
            return False
    
    def should_run_now(self):
        """Check if it's time to run based on 24/7 intelligent scheduling"""
        now = datetime.now()
        
        # Don't run if we just ran recently
        if self.last_run:
            time_since_last = (now - self.last_run).seconds
            
            # Check time of day for appropriate interval
            amsterdam_hour = now.hour
            if 9 <= amsterdam_hour <= 18:
                # Peak hours (9 AM - 6 PM): run every 3 minutes
                if time_since_last < self.peak_hours_interval:
                    return False
            elif 0 <= amsterdam_hour <= 6:
                # Night hours (12 AM - 6 AM): run every 10 minutes
                if time_since_last < self.night_interval:
                    return False
            else:
                # Off-peak hours: run every 5 minutes
                if time_since_last < self.off_peak_interval:
                    return False
        else:
            # First run
            return True
            
        return True
    
    def run(self):
        """Main continuous operation loop"""
        logger.info("🚀 Starting Dean 24/7 Continuous Operation...")
        logger.info("📊 Optimized for Meta API rate limits and ML learning")
        logger.info("🔄 Peak hours (9AM-6PM): every 3 minutes")
        logger.info("🔄 Off-peak hours: every 5 minutes")
        logger.info("🔄 Night hours (12AM-6AM): every 10 minutes")
        logger.info("🛡️ Maximum UI protection enabled")
        
        # Create logs directory
        os.makedirs('/opt/dean/logs', exist_ok=True)
        
        while self.running:
            try:
                if self.should_run_now():
                    success = self.run_dean_cycle()
                    self.last_run = datetime.now()
                    
                    if success:
                        # Success: wait for next cycle based on time of day
                        now = datetime.now()
                        if 9 <= now.hour <= 18:
                            wait_time = self.peak_hours_interval  # 3 minutes
                        elif 0 <= now.hour <= 6:
                            wait_time = self.night_interval  # 10 minutes
                        else:
                            wait_time = self.off_peak_interval  # 5 minutes
                        
                        logger.info(f"⏳ Waiting {wait_time//60} minutes until next cycle...")
                        time.sleep(wait_time)
                    else:
                        # Failure: wait longer before retry
                        logger.info("⏳ Cycle failed, waiting 10 minutes before retry...")
                        time.sleep(600)
                else:
                    # Not time to run yet, wait a bit
                    time.sleep(60)
                    
            except KeyboardInterrupt:
                logger.info("🛑 Received interrupt, shutting down...")
                break
            except Exception as e:
                logger.error(f"💥 Unexpected error: {e}")
                time.sleep(60)
        
        logger.info("👋 Dean Continuous Operation stopped")

if __name__ == "__main__":
    runner = DeanContinuousRunner()
    runner.run()
EOF

# Make script executable
chmod +x /opt/dean/run_continuous.py

# Create logs directory
mkdir -p /opt/dean/logs
chown dean:dean /opt/dean/logs

# Create monitoring script
echo "📊 Creating monitoring script..."
sudo -u dean tee /opt/dean/monitor.sh > /dev/null <<'EOF'
#!/bin/bash

echo "🔍 Dean ML-Enhanced Automation Status"
echo "======================================"
echo ""

# Service status
echo "📋 Service Status:"
systemctl status dean-automation.service --no-pager -l
echo ""

# Recent logs
echo "📊 Recent Logs:"
journalctl -u dean-automation.service --no-pager -l -n 20
echo ""

# Disk usage
echo "💾 Disk Usage:"
df -h /opt/dean
echo ""

# ML System Status
echo "🧠 ML System Status:"
if [ -f "/opt/dean/logs/dean_continuous.log" ]; then
    echo "Recent continuous logs:"
    tail -n 10 /opt/dean/logs/dean_continuous.log
else
    echo "No continuous logs found"
fi
echo ""

# Rate limiting status
echo "🚦 Rate Limiting Status:"
echo "Request delay: ${META_REQUEST_DELAY:-1.2}s"
echo "Max concurrent insights: ${META_MAX_CONCURRENT_INSIGHTS:-2}"
echo "Retry max: ${META_RETRY_MAX:-8}"
echo ""

# System resources
echo "💻 System Resources:"
echo "Memory usage:"
free -h
echo ""
echo "CPU usage:"
top -bn1 | grep "Cpu(s)"
EOF

chmod +x /opt/dean/monitor.sh

# Create log rotation
echo "📝 Setting up log rotation..."
tee /etc/logrotate.d/dean > /dev/null <<EOF
/opt/dean/logs/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 dean dean
    postrotate
        systemctl reload dean-automation.service
    endscript
}
EOF

# Enable and start service
echo "🚀 Enabling and starting Dean service..."
systemctl daemon-reload
systemctl enable dean-automation.service

echo ""
echo "🎉 Dean ML-Enhanced Automation deployment complete!"
echo ""
echo "📋 Next Steps:"
echo "1. Edit environment file: nano /opt/dean/.env"
echo "2. Add your Meta Ads API credentials"
echo "3. Add your Supabase credentials"
echo "4. Start the service: systemctl start dean-automation.service"
echo "5. Check status: systemctl status dean-automation.service"
echo "6. Monitor: /opt/dean/monitor.sh"
echo ""
echo "🔧 Service Management:"
echo "  Start:   systemctl start dean-automation.service"
echo "  Stop:    systemctl stop dean-automation.service"
echo "  Restart: systemctl restart dean-automation.service"
echo "  Status:  systemctl status dean-automation.service"
echo "  Logs:    journalctl -u dean-automation.service -f"
echo ""
echo "📊 Monitoring:"
echo "  Monitor: /opt/dean/monitor.sh"
echo "  Logs:   tail -f /opt/dean/logs/dean_continuous.log"
echo ""
echo "🚀 Your ML system will now learn continuously with optimized rate limiting!"
echo "📈 Business hours: every 5 minutes, Off hours: every 30 minutes"
echo ""
echo "⚠️  IMPORTANT: Edit /opt/dean/.env with your actual credentials before starting!"
