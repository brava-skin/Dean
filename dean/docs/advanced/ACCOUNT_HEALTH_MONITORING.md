# Ad Account Health Monitoring

This document describes the new ad account health monitoring system that helps prevent account suspensions and payment issues.

## Overview

The account health monitoring system continuously checks your Meta ad account for potential issues that could lead to account suspension or payment failures. It provides proactive alerts to help you address problems before they become critical.

## Features

### 🔍 Health Checks

The system monitors:

- **Account Status**: Verifies the account is active and not restricted
- **Payment Method**: Checks for payment failures, expired cards, or declined transactions
- **Account Balance**: Monitors balance levels and warns when funds are low
- **Spend Caps**: Alerts when approaching daily/monthly spending limits
- **Business Verification**: Ensures business information is complete
- **API Access**: Validates that the system can access account data

### 🚨 Alert Types

#### Critical Alerts (Immediate Action Required)
- Account status is inactive or restricted
- Payment method has failed or been declined
- Account balance is zero or negative
- API access has been revoked

#### Warning Alerts (Monitor Closely)
- Account balance is low (below configured threshold)
- Approaching spend cap limit (80% by default)
- Business verification information missing
- Unusual spending patterns

#### Info Alerts (Good to Know)
- Business verification status updates
- Account configuration changes
- New payment methods added

## Configuration

### Basic Setup

The account health monitoring is configured in `config/rules.yaml`:

```yaml
account_health:
  enabled: true
  check_interval_hours: 1  # How often to check account health
  alerts:
    payment_issues: true
    low_balance: true
    spend_cap_warnings: true
    business_verification: true
  thresholds:
    balance_warning_eur: 10.0    # Alert when balance below this amount
    spend_cap_warning_pct: 80   # Alert when spent this % of spend cap
    balance_critical_eur: 0.0   # Critical alert when balance at or below this
```

### Alert Thresholds

You can customize when alerts are triggered:

- **balance_warning_eur**: Alert when account balance drops below this amount (default: €10)
- **balance_critical_eur**: Critical alert when balance reaches this level (default: €0)
- **spend_cap_warning_pct**: Alert when spending reaches this percentage of your spend cap (default: 80%)

## How It Works

### 1. Health Check Process

Every time the system runs, it performs a comprehensive health check:

1. **Account Status Verification**: Checks if the account is active and accessible
2. **Payment Method Validation**: Verifies payment methods are working
3. **Balance Monitoring**: Checks current account balance
4. **Spend Cap Analysis**: Compares current spending against limits
5. **Business Verification**: Ensures business information is complete

### 2. Alert Generation

Based on the health check results:

- **Critical Issues**: Immediate Slack alerts with error severity
- **Warnings**: Slack alerts with warning severity
- **Info**: Logged for monitoring but not always alerted

### 3. Integration with Main System

The health check is integrated into the main automation loop:

```python
# Ad account health check with alerting
account_health = check_ad_account_health(client, settings)
if not account_health["ok"]:
    notify("🚨 Ad account health issues detected - check alerts for details")
```

## Alert Examples

### Critical Alert: Payment Failure
```
🚨 CRITICAL: Ad Account Health Issues Detected
Account: act_123456789

Issues:
• Payment method issue: declined
• Account status is 3 (not active)

⚠️ Your ad account may be disabled or restricted. Check immediately!
```

### Warning Alert: Low Balance
```
💰 Low Account Balance
Account: act_123456789
Balance: 5.50 EUR

⚠️ Add funds to prevent account suspension.
```

### Warning Alert: Spend Cap
```
📊 Spend Cap Warning
Account: act_123456789
Spent: 800.00 EUR (80.0% of cap)
Cap: 1000.00 EUR

⚠️ Approaching spend limit. Consider increasing cap or monitoring spend.
```

## Troubleshooting

### Common Issues

**1. Health Check Fails**
```
Error: Account health check failed: API access denied
Solution: Verify META_ACCESS_TOKEN has ads_management permission
```

**2. False Positive Alerts**
```
Issue: Getting alerts for normal account status
Solution: Adjust thresholds in config/rules.yaml
```

**3. Missing Alerts**
```
Issue: Not receiving alerts for known issues
Solution: Check Slack webhook configuration and alert settings
```

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
# Run with debug logging
export DEBUG=true
python src/main.py --simulate
```

## Best Practices

### 1. Monitor Alert Patterns
- Review alert frequency to tune thresholds
- Investigate recurring warnings
- Update payment methods before expiration

### 2. Proactive Management
- Set up multiple payment methods
- Monitor account balance regularly
- Keep business verification information current

### 3. Response Planning
- Have backup payment methods ready
- Know your account manager contact
- Keep business documents updated

## Testing

Test the account health monitoring:

```bash
# Run the test script
python3 scripts/test_account_health.py
```

This will:
- Test basic health check functionality
- Verify configuration settings
- Demonstrate alert types
- Show monitoring capabilities

## Integration with Existing System

The account health monitoring integrates seamlessly with the existing automation:

- **No Performance Impact**: Health checks run in parallel with other operations
- **Configurable**: Can be enabled/disabled via configuration
- **Non-Blocking**: Health issues don't stop the automation (unless critical)
- **Comprehensive Logging**: All health checks are logged for analysis

## Support

If you encounter issues with account health monitoring:

1. Check the logs in `data/actions.log.jsonl`
2. Verify your Meta API permissions
3. Ensure Slack webhooks are configured
4. Review the configuration in `config/rules.yaml`

For additional help, refer to the main documentation or contact support.
