# Creative Intelligence System Guide

## Overview

The Creative Intelligence System is an advanced feature of Dean that provides comprehensive creative management with performance tracking, ML analysis, and AI-powered creative generation. It transforms your copy bank into an intelligent, data-driven creative optimization system.

## Features

### 🎯 **Performance Tracking**
- Track individual creative performance (CTR, ROAS, CVR by creative)
- Monitor creative combinations and their effectiveness
- Historical performance analysis and trending

### 🧠 **Machine Learning Analysis**
- Identify top-performing creative patterns
- Detect creative fatigue and performance degradation
- Predict creative performance based on historical data
- Semantic similarity analysis between creatives

### 🤖 **AI-Powered Creative Generation**
- Generate new creatives based on top performers using ChatGPT
- Create variations that maintain style while improving performance
- Automated creative optimization suggestions

### 📊 **Advanced Analytics**
- Creative performance insights and recommendations
- A/B testing framework for creative optimization
- Creative rotation strategies based on performance data

## Setup

### 1. Database Schema
First, apply the creative intelligence schema to your Supabase database:

```sql
-- Run the schema file
\i dean/creative_intelligence_schema.sql
```

### 2. Environment Variables
Add these environment variables to your `.env` file:

```bash
# OpenAI API Key for AI creative generation
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Creative Intelligence specific settings
CREATIVE_INTELLIGENCE_ENABLED=true
CREATIVE_AI_MODEL=gpt-4
```

### 3. Install Dependencies
Install the additional dependencies for creative intelligence:

```bash
pip install -r dean/creative_requirements.txt
```

### 4. Configuration
The system uses `dean/config/creative_intelligence.yaml` for configuration. Key settings:

- `enabled`: Enable/disable the creative intelligence system
- `ai_generation.enabled`: Enable AI-powered creative generation
- `performance_tracking.enabled`: Enable performance tracking
- `similarity_analysis.enabled`: Enable semantic similarity analysis

## Usage

### Automatic Integration
The Creative Intelligence System is automatically initialized when you run Dean with Supabase enabled. It will:

1. Load your existing copy bank data into Supabase
2. Start tracking creative performance automatically
3. Provide insights and recommendations

### Manual Creative Management

#### Generate AI Creatives
```python
from creative.creative_intelligence import create_creative_intelligence_system

# Initialize the system
creative_system = create_creative_intelligence_system(supabase_client, openai_api_key)

# Generate AI creatives based on a top performer
new_creatives = creative_system.generate_ai_creatives(
    source_creative_id="headline_1",
    creative_type="headline",
    count=3
)
```

#### Analyze Creative Performance
```python
# Get performance insights
insights = creative_system.analyze_creative_performance(days_back=30)

# Get top performing creatives
top_creatives = creative_system.get_top_creatives("headline", limit=10)
```

#### Find Similar Creatives
```python
# Find semantically similar creatives
similar = creative_system.find_similar_creatives("headline_1", threshold=0.7)
```

## Database Tables

### `creative_library`
Stores all creatives (primary texts, headlines, descriptions) with metadata and performance scores.

### `creative_performance`
Tracks performance metrics for each creative used in ads.

### `creative_combinations`
Tracks performance of specific creative combinations (primary text + headline + description).

### `ai_generated_creatives`
Stores AI-generated creatives with generation metadata.

### `creative_similarity`
Stores semantic similarity scores between creatives.

### `creative_patterns`
Stores ML-identified patterns in creative performance.

### `creative_ab_tests`
Manages A/B tests for creative optimization.

## Performance Metrics

The system tracks comprehensive performance metrics:

- **Engagement Metrics**: CTR, impressions, clicks
- **Conversion Metrics**: ROAS, CPA, purchase rate
- **Creative-Specific Metrics**: Engagement rate, conversion rate
- **Combination Metrics**: Performance of creative combinations

## AI Integration

### ChatGPT Integration
- Generates new creatives based on top performers
- Maintains brand voice and style consistency
- Optimizes for social media advertising best practices

### Semantic Analysis
- Uses sentence-transformers for semantic similarity
- Identifies creative patterns and themes
- Enables intelligent creative recommendations

## Best Practices

### 1. Regular Performance Review
- Review creative performance insights weekly
- Update creative rotation strategies based on data
- Monitor for creative fatigue patterns

### 2. A/B Testing
- Test new creatives against top performers
- Use statistical significance for decision making
- Document winning creative patterns

### 3. AI Creative Generation
- Generate variations of top-performing creatives
- Test AI-generated creatives against originals
- Iterate based on performance feedback

### 4. Creative Rotation
- Rotate creatives based on performance data
- Maintain freshness while preserving top performers
- Use fatigue detection for rotation timing

## Monitoring and Alerts

The system provides automated monitoring and alerts for:

- High-performing creatives that should be scaled
- Creative fatigue detection
- AI creative generation completion
- A/B test completion and results
- Performance anomalies

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Verify API key is correct and has sufficient credits
   - Check rate limits and usage quotas

2. **Similarity Analysis Issues**
   - Ensure sentence-transformers is properly installed
   - Check model download and initialization

3. **Performance Tracking Issues**
   - Verify Supabase schema is correctly applied
   - Check data permissions and access

### Debug Mode
Enable debug logging to troubleshoot issues:

```python
import logging
logging.getLogger('creative').setLevel(logging.DEBUG)
```

## Future Enhancements

Planned enhancements include:

- Image creative analysis and optimization
- Video creative performance tracking
- Advanced creative fatigue prediction
- Multi-language creative support
- Creative performance forecasting
- Integration with external creative tools

## Support

For issues or questions about the Creative Intelligence System:

1. Check the troubleshooting section
2. Review the configuration settings
3. Check the system logs for error messages
4. Verify all dependencies are properly installed

The Creative Intelligence System transforms your creative management from manual to intelligent, data-driven optimization that continuously improves your ad performance.
