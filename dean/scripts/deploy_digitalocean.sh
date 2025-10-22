#!/bin/bash

# Dean ML-Enhanced Automation - DigitalOcean Deployment Script
# Optimized for continuous operation with advanced rate limiting

set -e

echo "🚀 Deploying Dean ML-Enhanced Automation to DigitalOcean..."

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install Python 3.11 and pip
echo "🐍 Installing Python 3.11..."
sudo apt install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Install system dependencies
echo "📚 Installing system dependencies..."
sudo apt install -y git curl wget build-essential libssl-dev libffi-dev

# Create dean user
echo "👤 Creating dean user..."
sudo useradd -m -s /bin/bash dean || true
sudo usermod -aG sudo dean || true

# Create project directory
echo "📁 Setting up project directory..."
sudo mkdir -p /opt/dean
sudo chown dean:dean /opt/dean

# Clone repository
echo "📥 Cloning repository..."
cd /opt/dean
sudo -u dean git clone https://github.com/brava-skin/Dean.git . || sudo -u dean git pull

# Create virtual environment
echo "🔧 Setting up Python virtual environment..."
sudo -u dean python3.11 -m venv /opt/dean/venv
sudo -u dean /opt/dean/venv/bin/pip install --upgrade pip

# Install Python dependencies
echo "📦 Installing Python dependencies..."
sudo -u dean /opt/dean/venv/bin/pip install -r /opt/dean/requirements.txt

# Create systemd service for continuous operation
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/dean-automation.service > /dev/null <<EOF
[Unit]
Description=Dean ML-Enhanced Meta Ads Automation
After=network.target

[Service]
Type=simple
User=dean
Group=dean
WorkingDirectory=/opt/dean
Environment=PATH=/opt/dean/venv/bin
ExecStart=/opt/dean/venv/bin/python /opt/dean/src/main.py --profile production --continuous
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

# Create environment file
echo "🔐 Setting up environment variables..."
sudo -u dean tee /opt/dean/.env > /dev/null <<EOF
# Meta Ads API Configuration
FB_APP_ID=your_app_id
FB_APP_SECRET=your_app_secret
FB_ACCESS_TOKEN=your_access_token
FB_AD_ACCOUNT_ID=your_ad_account_id
FB_PIXEL_ID=your_pixel_id
FB_PAGE_ID=your_page_id
IG_ACTOR_ID=your_ig_actor_id

# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
SUPABASE_TABLE=meta_creatives

# Business Configuration
STORE_URL=your_store_url
SLACK_WEBHOOK_URL=your_slack_webhook
BREAKEVEN_CPA=your_breakeven_cpa
COGS_PER_PURCHASE=your_cogs
USD_EUR_RATE=your_exchange_rate

# ML Configuration
ML_MODE=true
ML_LEARNING_RATE=0.01
ML_CONFIDENCE_THRESHOLD=0.7

# Rate Limiting Configuration (Optimized for continuous operation)
META_REQUEST_DELAY=1.2
META_MAX_CONCURRENT_INSIGHTS=2
META_RETRY_MAX=5
META_BACKOFF_BASE=1.5
META_BUC_ENABLED=true

# Timezone
TZ=Europe/Amsterdam
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
        self.run_interval = 600  # 10 minutes between runs
        self.max_concurrent = 2  # Max 2 concurrent API calls
        
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
        """Check if it's time to run based on intelligent scheduling"""
        now = datetime.now()
        
        # Don't run if we just ran recently
        if self.last_run and (now - self.last_run).seconds < self.run_interval:
            return False
            
        # Run more frequently during business hours (9 AM - 6 PM Amsterdam)
        amsterdam_hour = now.hour
        if 9 <= amsterdam_hour <= 18:
            return True
            
        # Run every 30 minutes during off-hours
        if self.last_run and (now - self.last_run).seconds >= 1800:
            return True
            
        return False
    
    def run(self):
        """Main continuous operation loop"""
        logger.info("🚀 Starting Dean Continuous Operation...")
        logger.info("📊 Optimized for Meta API rate limits and ML learning")
        
        # Create logs directory
        os.makedirs('/opt/dean/logs', exist_ok=True)
        
        while self.running:
            try:
                if self.should_run_now():
                    success = self.run_dean_cycle()
                    self.last_run = datetime.now()
                    
                    if success:
                        # Success: wait for next cycle
                        time.sleep(self.run_interval)
                    else:
                        # Failure: wait longer before retry
                        time.sleep(self.run_interval * 2)
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
sudo chmod +x /opt/dean/run_continuous.py

# Create logs directory
sudo mkdir -p /opt/dean/logs
sudo chown dean:dean /opt/dean/logs

# Enable and start service
echo "🚀 Enabling and starting Dean service..."
sudo systemctl daemon-reload
sudo systemctl enable dean-automation.service

# Create monitoring script
echo "📊 Creating monitoring script..."
sudo -u dean tee /opt/dean/monitor.sh > /dev/null <<'EOF'
#!/bin/bash

echo "🔍 Dean ML-Enhanced Automation Status"
echo "======================================"

# Service status
echo "📋 Service Status:"
systemctl status dean-automation.service --no-pager -l

echo ""
echo "📊 Recent Logs:"
journalctl -u dean-automation.service --no-pager -l -n 20

echo ""
echo "💾 Disk Usage:"
df -h /opt/dean

echo ""
echo "🧠 ML System Status:"
if [ -f "/opt/dean/logs/dean_continuous.log" ]; then
    tail -n 10 /opt/dean/logs/dean_continuous.log
else
    echo "No continuous logs found"
fi
EOF

sudo chmod +x /opt/dean/monitor.sh

# Create log rotation
echo "📝 Setting up log rotation..."
sudo tee /etc/logrotate.d/dean > /dev/null <<EOF
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

echo ""
echo "🎉 Dean ML-Enhanced Automation deployed to DigitalOcean!"
echo ""
echo "📋 Next Steps:"
echo "1. Edit /opt/dean/.env with your actual credentials"
echo "2. Start the service: sudo systemctl start dean-automation.service"
echo "3. Check status: sudo systemctl status dean-automation.service"
echo "4. Monitor logs: /opt/dean/monitor.sh"
echo ""
echo "🔧 Service Management:"
echo "  Start:   sudo systemctl start dean-automation.service"
echo "  Stop:    sudo systemctl stop dean-automation.service"
echo "  Restart: sudo systemctl restart dean-automation.service"
echo "  Status:  sudo systemctl status dean-automation.service"
echo "  Logs:    journalctl -u dean-automation.service -f"
echo ""
echo "📊 Monitoring:"
echo "  Monitor: /opt/dean/monitor.sh"
echo "  Logs:   tail -f /opt/dean/logs/dean_continuous.log"
echo ""
echo "🚀 Your ML system will now learn continuously with optimized rate limiting!"
