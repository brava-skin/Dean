# Advanced Smart Scheduler Features

This document describes the advanced features of the Smart Scheduler system that makes GitHub Actions ticks incredibly intelligent and future-proof.

## 🧠 **Smart Features Overview**

### **1. Duplicate Prevention**
- **Unique Tick IDs**: Each tick gets a unique identifier based on timestamp and type
- **Database Tracking**: SQLite database prevents duplicate execution
- **Smart Detection**: Automatically skips already-processed ticks

### **2. Intelligent Timing**
- **American Peak Hour Avoidance**: Delays execution during US business hours (9-11 AM and 2-4 PM EST)
- **American Timezone Optimization**: Avoids running during US sleep hours (2-6 AM EST)
- **Adaptive Delays**: Adjusts timing based on system performance

### **3. Advanced Reliability**
- **Smart Retry Logic**: Exponential backoff with performance-based adjustments
- **Circuit Breaker**: Prevents cascading failures
- **Health Monitoring**: Continuous system health assessment
- **Adaptive Thresholds**: Automatically adjusts based on performance

### **4. Performance Learning**
- **Auto-Optimization**: Configuration automatically tunes based on performance
- **Performance Tracking**: Detailed metrics collection and analysis
- **Smart Thresholds**: Dynamic adjustment of health and latency thresholds
- **Learning Algorithm**: Improves over time based on historical data

## 🔧 **Configuration System**

### **Advanced Settings File**
Create `config/advanced_settings.yaml`:

```yaml
scheduling:
  smart_timing: true
  adaptive_delays: true
  peak_hour_avoidance: true
  timezone_optimization: true

reliability:
  max_retries: 3
  retry_delay_base: 300  # 5 minutes
  retry_delay_multiplier: 2.0
  circuit_breaker_threshold: 5
  health_check_interval: 1800  # 30 minutes

performance:
  health_score_threshold: 70
  api_latency_threshold: 10.0
  db_latency_threshold: 2.0
  memory_usage_threshold: 80.0

monitoring:
  detailed_logging: true
  performance_tracking: true
  alert_cooldowns:
    queue_empty: 21600  # 6 hours
    system_health: 7200  # 2 hours
    api_errors: 3600  # 1 hour
    db_errors: 1800  # 30 minutes

optimization:
  auto_tune_retries: true
  auto_adjust_delays: true
  performance_learning: true
  adaptive_thresholds: true
```

## 📊 **Monitoring Dashboard**

### **Health Reports**
The system automatically generates health reports:

```bash
# Get 7-day health report
python -c "
from src.monitoring_dashboard import MonitoringDashboard
dashboard = MonitoringDashboard()
dashboard.send_health_report(days=7)
"
```

### **Performance Trends**
Track performance over time:
- Daily success rates
- Hourly distribution patterns
- Health score trends
- Error frequency analysis

### **System Recommendations**
Get intelligent recommendations:
- Performance optimization suggestions
- Resource scaling recommendations
- Configuration tuning advice
- Error pattern analysis

## 🚀 **Advanced Features**

### **1. Smart Timing**
```python
# American peak hour avoidance (EST)
est_hour = (amsterdam_hour + 6) % 24  # Convert Amsterdam to EST
if est_hour in [9, 10, 11, 14, 15, 16]:  # US business hours
    delay_execution(300)  # 5 minute delay

# American timezone optimization (EST)
if 2 <= est_hour <= 6:  # US sleep hours (2-6 AM EST = 8-12 PM Amsterdam)
    delay_execution(600)  # 10 minute delay
```

### **2. Adaptive Retry Logic**
```python
# Smart retry delays based on performance
base_delay = 300  # 5 minutes
multiplier = 2.0
retry_delay = base_delay * (multiplier ** retry_count)

# Performance-based adjustments
if recent_performance < 0.8:
    retry_delay *= 1.5  # Increase delay for poor performance
elif recent_performance > 0.95:
    retry_delay *= 0.8  # Decrease delay for excellent performance
```

### **3. Circuit Breaker**
```python
# Automatic failure detection
if consecutive_failures >= circuit_breaker_threshold:
    skip_execution("circuit_breaker_triggered")
    alert_system_health("Circuit breaker activated")
```

### **4. Performance Learning**
```python
# Automatic configuration optimization
if recent_performance < 0.8:
    increase_retry_count()
    lower_health_threshold()
elif recent_performance > 0.95:
    decrease_retry_count()
    raise_health_threshold()
```

## 📈 **Performance Metrics**

### **Health Score Calculation**
- **API Latency**: Response time to Meta API
- **Database Latency**: SQLite query performance
- **Memory Usage**: System resource utilization
- **Error Rate**: Frequency of failures
- **Success Rate**: Percentage of successful ticks

### **Adaptive Thresholds**
The system automatically adjusts thresholds based on performance:

```python
# Poor performance (score < 0.7)
health_threshold *= 0.8  # Lower threshold
api_latency_threshold *= 1.2  # Higher tolerance

# Excellent performance (score > 0.9)
health_threshold *= 1.1  # Higher threshold
api_latency_threshold *= 0.8  # Lower tolerance
```

## 🔍 **Monitoring & Alerts**

### **Real-time Health Monitoring**
- Continuous system health assessment
- Automatic performance tracking
- Intelligent alert generation
- Smart cooldown management

### **Advanced Alert System**
- **Queue Empty**: Intelligent detection with cooldowns
- **System Health**: Performance-based alerts
- **API Issues**: Latency and error monitoring
- **Database Problems**: Connection and performance alerts

### **Smart Cooldowns**
Different alert types have different cooldown periods:
- **Queue Empty**: 6 hours
- **System Health**: 2 hours
- **API Errors**: 1 hour
- **Database Errors**: 30 minutes

## 🛠️ **Troubleshooting**

### **Common Issues**

1. **High Retry Count**
   - Check system health score
   - Review error patterns
   - Adjust retry configuration

2. **Frequent Skipping**
   - Check health thresholds
   - Review timing configuration
   - Analyze performance trends

3. **Poor Performance**
   - Review monitoring dashboard
   - Check resource utilization
   - Analyze error patterns

### **Debug Commands**

```bash
# Check system health
python -c "
from src.smart_scheduler import SmartScheduler
scheduler = SmartScheduler(settings, rules, store)
print(f'Health Score: {scheduler._get_health_score():.1f}')
"

# Get performance statistics
python -c "
from src.monitoring_dashboard import MonitoringDashboard
dashboard = MonitoringDashboard()
stats = dashboard.get_tick_statistics(days=7)
print(f'Success Rate: {stats[\"success_rate\"]:.1f}%')
"

# Export metrics
python -c "
from src.monitoring_dashboard import MonitoringDashboard
dashboard = MonitoringDashboard()
metrics = dashboard.export_metrics(days=7, format='json')
print(metrics)
"
```

## 📋 **Best Practices**

### **1. Configuration Management**
- Start with default settings
- Monitor performance for 1 week
- Adjust based on recommendations
- Use auto-optimization features

### **2. Monitoring**
- Check health reports weekly
- Monitor performance trends
- Review error patterns
- Follow system recommendations

### **3. Optimization**
- Enable performance learning
- Use adaptive thresholds
- Monitor auto-optimization
- Review configuration changes

### **4. Maintenance**
- Clean up old records monthly
- Review performance metrics
- Update configuration as needed
- Monitor system health

## 🎯 **Future-Proofing Features**

### **1. Scalability**
- Automatic resource scaling
- Performance-based optimization
- Adaptive configuration
- Intelligent load balancing

### **2. Reliability**
- Circuit breaker protection
- Smart retry logic
- Health monitoring
- Failure recovery

### **3. Intelligence**
- Performance learning
- Adaptive thresholds
- Smart timing
- Auto-optimization

### **4. Monitoring**
- Comprehensive dashboards
- Real-time health tracking
- Performance analytics
- Intelligent recommendations

## 🚀 **Getting Started**

1. **Enable Advanced Features**:
   ```bash
   # The system automatically uses smart features
   python src/main.py --profile production
   ```

2. **Monitor Performance**:
   ```bash
   # Check health report
   python -c "
   from src.monitoring_dashboard import MonitoringDashboard
   dashboard = MonitoringDashboard()
   dashboard.send_health_report()
   "
   ```

3. **Optimize Configuration**:
   ```bash
   # Create advanced settings
   mkdir -p config
   # Edit config/advanced_settings.yaml with your preferences
   ```

4. **Monitor Results**:
   - Watch Slack for intelligent alerts
   - Check health reports
   - Review performance trends
   - Follow system recommendations

The Smart Scheduler transforms your simple GitHub Actions ticks into an intelligent, self-optimizing, future-proof monitoring system! 🧠✨
