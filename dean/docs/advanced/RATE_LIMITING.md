# Meta Marketing API Rate Limiting Guide

## Overview

This system implements comprehensive rate limiting for the Meta Marketing API, addressing all rate limiting types specified in Meta's documentation. The system intelligently manages API usage to prevent rate limit violations and optimize performance.

## Rate Limiting Types Implemented

### 1. API-Level Scoring System
- **Read requests**: 1 point each
- **Write requests**: 3 points each
- **Development tier**: 60 points max, 300s block duration
- **Standard tier**: 9000 points max, 60s block duration
- **Score decay**: 300 seconds

### 2. Business Use Case (BUC) Rate Limits
- **ads_management**: 300 (dev) / 100,000 (standard) per hour
- **custom_audience**: 5,000 (dev) / 190,000 (standard) per hour
- **ads_insights**: 600 (dev) / 190,000 (standard) per hour
- **catalog_management**: 20,000 (both tiers) per hour
- **catalog_batch**: 200 (both tiers) per hour

### 3. Ads Insights Platform Rate Limits
- App-level enforcement
- Blocks all Insights API calls when limit reached
- Error codes: 4, subcode 1504022 or 1504039

### 4. Ad Account Level Limits
- **Spend limit changes**: 10 times per day
- **Ad set budget changes**: 4 times per hour per ad set
- **Ad creation limits**: Based on daily spend limit

### 5. App-Level Limits
- Application-wide rate limiting
- Blocks all API calls for the app when reached

## Configuration

### Environment Variables

```bash
# API Tier Configuration
export META_API_TIER=development  # or "standard"

# Rate Limiting Configuration
export META_BUC_ENABLED=true
export META_REQUEST_DELAY=0.5
export META_RETRY_MAX=6
export META_BACKOFF_BASE=1.0
export META_WRITE_COOLDOWN_SEC=10
export META_TIMEOUT=30

# Circuit Breaker Configuration
export META_CB_FAILS=5
export META_CB_RESET_SEC=120
```

### API Tier Selection

**Development Tier** (Default):
- Lower rate limits but sufficient for testing
- 60 API score points maximum
- 300-second block duration
- Suitable for development and testing

**Standard Tier** (Production):
- Higher rate limits for production use
- 9000 API score points maximum
- 60-second block duration
- Requires app review approval from Meta

## Usage

### Basic Usage

```python
from meta_client import MetaClient, AccountAuth, ClientConfig

# Create client with rate limiting
account = AccountAuth(
    account_id="your_account_id",
    access_token="your_access_token",
    app_id="your_app_id",
    app_secret="your_app_secret"
)

client = MetaClient(accounts=[account], cfg=ClientConfig())

# Make requests - rate limiting is automatic
insights = client.get_ad_insights(level="ad")
```

### Rate Limit Status Monitoring

```python
# Get current rate limit status
status = client.get_rate_limit_status()
print(f"API Score: {status['current_score']}/{status['max_score']}")
print(f"Usage: {status['score_usage_pct']:.1f}%")
print(f"BUC Usage: {status['buc_usage']}")
```

### Budget Change Rate Limiting

```python
# Check if budget can be changed
if client.rate_limit_manager.can_change_budget("adset_123"):
    client.update_adset_budget("adset_123", 150.0)
else:
    print("Budget change limit reached (4/hour)")
```

## Error Handling

### Rate Limit Error Codes

| Code | Subcode | Description | Action |
|------|---------|-------------|---------|
| 4 | 1504022 | Ads Insights Platform rate limit | Wait 60s, app-level block |
| 4 | 1504039 | General app-level rate limit | Wait 30s, app-level block |
| 4 | - | General application rate limit | Wait up to 60s |
| 17 | 2446079 | Ad account level API limit | Wait based on tier |
| 17 | 1885172 | Spend limit changes (10/day) | Wait 1 hour |
| 613 | 1487742 | Too many calls from ad account | Wait 60s |
| 613 | 1487632 | Ad set budget limit (4/hour) | Wait 1 hour |
| 613 | 1487225 | Ad creation limit | Wait 60s |
| 80000-80014 | - | Business Use Case rate limits | Wait 60s |

### Automatic Retry Logic

The system automatically:
1. Detects rate limit errors
2. Determines appropriate wait time
3. Implements exponential backoff
4. Retries requests when appropriate
5. Logs all rate limit events

## Monitoring and Alerting

### Built-in Monitoring

The system provides comprehensive monitoring through:

```python
status = client.get_rate_limit_status()
```

Returns:
- Current API score and usage percentage
- Block status and remaining time
- BUC usage counters
- Error counts by type
- Recent request history

### Slack Notifications

Rate limit events are automatically sent to Slack with:
- Rate limit type and severity
- Wait time required
- Current usage statistics
- Recommended actions

### Best Practices for Monitoring

1. **Regular Status Checks**: Monitor rate limit status every 15-30 minutes
2. **Usage Alerts**: Set up alerts when usage exceeds 80%
3. **Error Tracking**: Monitor error rates and types
4. **Performance Metrics**: Track API response times and success rates

## Optimization Strategies

### Request Optimization

1. **Batch Operations**: Group related requests together
2. **Request Timing**: Space out requests to avoid bursts
3. **Read vs Write**: Prefer read operations when possible
4. **Caching**: Cache frequently accessed data

### Tier Optimization

1. **Development**: Use for testing and development
2. **Standard**: Upgrade for production workloads
3. **Multiple Accounts**: Distribute load across accounts

### Error Prevention

1. **Pre-request Checks**: Verify rate limits before making requests
2. **Graceful Degradation**: Handle rate limits gracefully
3. **Circuit Breakers**: Prevent cascading failures
4. **Monitoring**: Early detection of issues

## Troubleshooting

### Common Issues

**High Rate Limit Usage**:
- Check for inefficient request patterns
- Consider upgrading to Standard tier
- Implement request queuing
- Optimize request timing

**Frequent Rate Limit Hits**:
- Increase META_REQUEST_DELAY
- Reduce concurrent operations
- Implement exponential backoff
- Monitor BUC usage patterns

**Performance Issues**:
- Check rate limit status regularly
- Optimize request patterns
- Consider multiple ad accounts
- Implement intelligent caching

### Debug Information

Enable detailed logging by setting:
```bash
export META_DEBUG=true
```

This provides:
- Detailed rate limit status
- Request/response logging
- Error categorization
- Performance metrics

## Testing

Run the comprehensive test suite:

```bash
cd dean/scripts
python test_rate_limits.py
```

This tests:
- Rate limit status functionality
- Request handling
- Error handling
- Budget change limits
- BUC rate limiting

## Migration Guide

### From Basic Rate Limiting

If upgrading from basic rate limiting:

1. **Update Environment Variables**:
   ```bash
   export META_API_TIER=development  # or standard
   export META_BUC_ENABLED=true
   ```

2. **Update Code**:
   ```python
   # Old way
   client._handle_rate_limit_error(error_data, endpoint)
   
   # New way (automatic)
   client.get_rate_limit_status()  # for monitoring
   ```

3. **Monitor Usage**:
   ```python
   # Add regular status checks
   status = client.get_rate_limit_status()
   if status['score_usage_pct'] > 80:
       # Implement backoff logic
   ```

### Production Deployment

1. **Set Standard Tier**:
   ```bash
   export META_API_TIER=standard
   ```

2. **Enable Monitoring**:
   ```python
   # Add monitoring to your main loop
   status = client.get_rate_limit_status()
   notify(f"Rate limit usage: {status['score_usage_pct']:.1f}%")
   ```

3. **Configure Alerts**:
   - Set up Slack webhooks for rate limit warnings
   - Monitor error rates
   - Track performance metrics

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the test script output
3. Monitor rate limit status
4. Check Meta Developer Console for API usage

## Changelog

### Version 2.0 (Current)
- Comprehensive rate limiting implementation
- Business Use Case (BUC) support
- Advanced error handling
- Real-time monitoring
- Budget change tracking
- Enhanced test suite

### Version 1.0 (Previous)
- Basic rate limiting
- Simple retry logic
- Limited error handling
