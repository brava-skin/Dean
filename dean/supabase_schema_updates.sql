-- =====================================================
-- DEAN ML SYSTEM - DATABASE UPDATES ONLY
-- Run this on existing database to add ML features
-- =====================================================

-- =====================================================
-- 1. CREATE MISSING TABLES (IF NOT EXISTS)
-- =====================================================

-- ML Models storage
CREATE TABLE IF NOT EXISTS ml_models (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_type TEXT NOT NULL CHECK (model_type IN (
        'performance_predictor', 'fatigue_detector', 'cpa_predictor', 
        'roas_predictor', 'purchase_probability', 'scaling_predictor',
        'creative_similarity', 'temporal_trend', 'cross_stage_transfer'
    )),
    version INTEGER NOT NULL DEFAULT 1,
    stage TEXT CHECK (stage IN ('testing', 'validation', 'scaling', 'global')),
    model_name TEXT NOT NULL,
    model_data BYTEA, -- Serialized model
    model_metadata JSONB DEFAULT '{}',
    training_data_hash TEXT,
    features_used TEXT[],
    hyperparameters JSONB DEFAULT '{}',
    performance_metrics JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    trained_at TIMESTAMP WITH TIME ZONE,
    UNIQUE(model_type, version, stage)
);

-- ML Predictions
CREATE TABLE IF NOT EXISTS ml_predictions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ad_id TEXT NOT NULL,
    lifecycle_id TEXT NOT NULL,
    model_id UUID NOT NULL REFERENCES ml_models(id),
    stage TEXT NOT NULL,
    prediction_type TEXT NOT NULL,
    
    -- Predictions
    predicted_value DECIMAL(10,4),
    confidence_score DECIMAL(5,4) DEFAULT 0,
    prediction_interval_lower DECIMAL(10,4),
    prediction_interval_upper DECIMAL(10,4),
    
    -- Features used
    features JSONB DEFAULT '{}',
    feature_importance JSONB DEFAULT '{}',
    
    -- Context
    prediction_horizon_hours INTEGER DEFAULT 24,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() + INTERVAL '7 days'
);

-- Feature Engineering
CREATE TABLE IF NOT EXISTS feature_engineering (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ad_id TEXT NOT NULL,
    lifecycle_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    feature_value DECIMAL(15,6),
    feature_type TEXT NOT NULL CHECK (feature_type IN (
        'raw', 'rolling', 'delta', 'ratio', 'trend', 'volatility', 
        'momentum', 'seasonal', 'interaction', 'derived'
    )),
    window_size INTEGER,
    calculation_method TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(ad_id, feature_name, created_at)
);

-- Learning events
CREATE TABLE IF NOT EXISTS learning_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type TEXT NOT NULL CHECK (event_type IN (
        'promotion', 'demotion', 'kill', 'scale_up', 'scale_down', 
        'fatigue_detected', 'pattern_learned', 'threshold_adjusted',
        'model_retrained', 'feature_importance_updated'
    )),
    ad_id TEXT,
    lifecycle_id TEXT,
    from_stage TEXT,
    to_stage TEXT,
    learning_data JSONB DEFAULT '{}',
    confidence_score DECIMAL(5,4),
    impact_score DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Adaptive rules
CREATE TABLE IF NOT EXISTS adaptive_rules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    rule_name TEXT NOT NULL,
    stage TEXT NOT NULL,
    rule_type TEXT NOT NULL CHECK (rule_type IN (
        'threshold', 'budget', 'pacing', 'promotion', 'kill', 'scale'
    )),
    current_value DECIMAL(10,4) NOT NULL,
    previous_value DECIMAL(10,4),
    adjustment_reason TEXT,
    confidence_weight DECIMAL(5,4) DEFAULT 1.0,
    learning_rate DECIMAL(5,4) DEFAULT 0.1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance patterns
CREATE TABLE IF NOT EXISTS performance_patterns (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    pattern_type TEXT NOT NULL CHECK (pattern_type IN (
        'ctr_trend', 'cpa_stability', 'roas_momentum', 'fatigue_onset',
        'seasonal_effect', 'daypart_optimization', 'creative_similarity'
    )),
    stage TEXT NOT NULL,
    pattern_data JSONB NOT NULL,
    confidence_score DECIMAL(5,4) DEFAULT 0,
    impact_score DECIMAL(5,4) DEFAULT 0,
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE
);

-- Time series data
CREATE TABLE IF NOT EXISTS time_series_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ad_id TEXT NOT NULL,
    lifecycle_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value DECIMAL(15,6) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    window_type TEXT,
    stage TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Temporal analysis
CREATE TABLE IF NOT EXISTS temporal_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    ad_id TEXT,
    analysis_type TEXT NOT NULL CHECK (analysis_type IN (
        'trend', 'seasonality', 'cyclical', 'irregular', 'fatigue_curve'
    )),
    time_series_data JSONB NOT NULL,
    trend_direction TEXT CHECK (trend_direction IN ('up', 'down', 'stable', 'volatile')),
    trend_strength DECIMAL(5,4),
    seasonality_detected BOOLEAN DEFAULT FALSE,
    seasonal_period INTEGER,
    forecast_data JSONB,
    confidence_interval JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Creative intelligence
CREATE TABLE IF NOT EXISTS creative_intelligence (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    creative_id TEXT NOT NULL,
    ad_id TEXT,
    creative_type TEXT CHECK (creative_type IN ('video', 'image', 'carousel', 'collection')),
    
    -- Creative attributes
    duration_seconds INTEGER,
    aspect_ratio TEXT,
    file_size_mb DECIMAL(8,2),
    resolution TEXT,
    color_palette JSONB,
    text_overlay BOOLEAN DEFAULT FALSE,
    music_present BOOLEAN DEFAULT FALSE,
    voice_over BOOLEAN DEFAULT FALSE,
    
    -- Performance by creative
    avg_ctr DECIMAL(5,4) DEFAULT 0,
    avg_cpa DECIMAL(8,2) DEFAULT 0,
    avg_roas DECIMAL(5,2) DEFAULT 0,
    performance_rank INTEGER,
    similarity_vector TEXT, -- Will be updated to vector type later
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Creative similarity
CREATE TABLE IF NOT EXISTS creative_similarity (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    creative_a TEXT NOT NULL,
    creative_b TEXT NOT NULL,
    similarity_score DECIMAL(5,4) NOT NULL,
    similarity_type TEXT CHECK (similarity_type IN ('visual', 'performance', 'combined')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(creative_a, creative_b, similarity_type)
);

-- System health
CREATE TABLE IF NOT EXISTS system_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    stage TEXT NOT NULL,
    health_score INTEGER NOT NULL CHECK (health_score >= 0 AND health_score <= 100),
    stability_score DECIMAL(5,2) DEFAULT 0,
    confidence_score DECIMAL(5,2) DEFAULT 0,
    efficiency_score DECIMAL(5,2) DEFAULT 0,
    metrics JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Account health
CREATE TABLE IF NOT EXISTS account_health (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    account_id TEXT NOT NULL,
    health_status TEXT NOT NULL CHECK (health_status IN ('healthy', 'warning', 'critical')),
    balance DECIMAL(10,2),
    spend_cap DECIMAL(10,2),
    auto_charge_threshold DECIMAL(10,2),
    payment_status TEXT,
    business_verification_status TEXT,
    restrictions JSONB DEFAULT '{}',
    alerts JSONB DEFAULT '[]',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Daily summaries
CREATE TABLE IF NOT EXISTS daily_summaries (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    date DATE NOT NULL,
    stage TEXT,
    
    -- Aggregated metrics
    total_spend DECIMAL(10,2) DEFAULT 0,
    total_impressions INTEGER DEFAULT 0,
    total_clicks INTEGER DEFAULT 0,
    total_purchases INTEGER DEFAULT 0,
    avg_cpa DECIMAL(8,2) DEFAULT 0,
    avg_roas DECIMAL(5,2) DEFAULT 0,
    
    -- Stage-specific metrics
    ads_active INTEGER DEFAULT 0,
    ads_promoted INTEGER DEFAULT 0,
    ads_killed INTEGER DEFAULT 0,
    ads_scaled INTEGER DEFAULT 0,
    
    -- ML insights
    predictions_accuracy DECIMAL(5,4) DEFAULT 0,
    confidence_trend DECIMAL(5,4) DEFAULT 0,
    learning_events_count INTEGER DEFAULT 0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(date, stage)
);

-- =====================================================
-- 2. ADD MISSING COLUMNS TO EXISTING TABLES
-- =====================================================

-- Add ML columns to existing tables if they don't exist
DO $$ 
BEGIN
    -- Add columns to ad_lifecycle if they don't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ad_lifecycle' AND column_name = 'lifecycle_id') THEN
        ALTER TABLE ad_lifecycle ADD COLUMN lifecycle_id TEXT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ad_lifecycle' AND column_name = 'parent_lifecycle_id') THEN
        ALTER TABLE ad_lifecycle ADD COLUMN parent_lifecycle_id UUID REFERENCES ad_lifecycle(id);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'ad_lifecycle' AND column_name = 'metadata') THEN
        ALTER TABLE ad_lifecycle ADD COLUMN metadata JSONB DEFAULT '{}';
    END IF;
    
    -- Add ML columns to performance_metrics if they don't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_metrics' AND column_name = 'lifecycle_id') THEN
        ALTER TABLE performance_metrics ADD COLUMN lifecycle_id TEXT;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_metrics' AND column_name = 'performance_quality_score') THEN
        ALTER TABLE performance_metrics ADD COLUMN performance_quality_score INTEGER DEFAULT 0 CHECK (performance_quality_score >= 0 AND performance_quality_score <= 100);
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_metrics' AND column_name = 'stability_score') THEN
        ALTER TABLE performance_metrics ADD COLUMN stability_score DECIMAL(5,2) DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_metrics' AND column_name = 'momentum_score') THEN
        ALTER TABLE performance_metrics ADD COLUMN momentum_score DECIMAL(5,2) DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_metrics' AND column_name = 'fatigue_index') THEN
        ALTER TABLE performance_metrics ADD COLUMN fatigue_index DECIMAL(5,2) DEFAULT 0;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_metrics' AND column_name = 'hour_of_day') THEN
        ALTER TABLE performance_metrics ADD COLUMN hour_of_day INTEGER;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_metrics' AND column_name = 'day_of_week') THEN
        ALTER TABLE performance_metrics ADD COLUMN day_of_week INTEGER;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_metrics' AND column_name = 'is_weekend') THEN
        ALTER TABLE performance_metrics ADD COLUMN is_weekend BOOLEAN DEFAULT FALSE;
    END IF;
    
    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'performance_metrics' AND column_name = 'ad_age_days') THEN
        ALTER TABLE performance_metrics ADD COLUMN ad_age_days INTEGER DEFAULT 0;
    END IF;
END $$;

-- =====================================================
-- 3. CREATE INDEXES (IF NOT EXISTS)
-- =====================================================

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_ad_id ON performance_metrics(ad_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_lifecycle ON performance_metrics(lifecycle_id);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_stage ON performance_metrics(stage);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_window ON performance_metrics(window_type);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_date ON performance_metrics(date_start, date_end);

-- ML predictions indexes
CREATE INDEX IF NOT EXISTS idx_ml_predictions_ad_id ON ml_predictions(ad_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model ON ml_predictions(model_id);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_stage ON ml_predictions(stage);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_created ON ml_predictions(created_at);

-- Learning events indexes
CREATE INDEX IF NOT EXISTS idx_learning_events_type ON learning_events(event_type);
CREATE INDEX IF NOT EXISTS idx_learning_events_ad_id ON learning_events(ad_id);
CREATE INDEX IF NOT EXISTS idx_learning_events_created ON learning_events(created_at);

-- Time series indexes
CREATE INDEX IF NOT EXISTS idx_time_series_ad_id ON time_series_data(ad_id);
CREATE INDEX IF NOT EXISTS idx_time_series_metric ON time_series_data(metric_name);
CREATE INDEX IF NOT EXISTS idx_time_series_timestamp ON time_series_data(timestamp);

-- Creative intelligence indexes
CREATE INDEX IF NOT EXISTS idx_creative_intelligence_creative_id ON creative_intelligence(creative_id);
CREATE INDEX IF NOT EXISTS idx_creative_intelligence_performance ON creative_intelligence(performance_rank);

-- =====================================================
-- 4. CREATE VIEWS (IF NOT EXISTS)
-- =====================================================

-- Drop existing views if they exist
DROP VIEW IF EXISTS current_active_ads;
DROP VIEW IF EXISTS ml_model_performance;

-- Current active ads with latest metrics
CREATE VIEW current_active_ads AS
SELECT 
    al.ad_id,
    al.stage,
    al.lifecycle_id,
    al.created_at,
    pm.spend,
    pm.impressions,
    pm.clicks,
    pm.purchases,
    pm.ctr,
    pm.cpa,
    pm.roas,
    pm.performance_quality_score,
    pm.stability_score,
    pm.fatigue_index
FROM ad_lifecycle al
LEFT JOIN performance_metrics pm ON al.ad_id = pm.ad_id 
    AND pm.window_type = '1d' 
    AND pm.date_end = CURRENT_DATE
WHERE al.status = 'active';

-- ML model performance tracking
CREATE VIEW ml_model_performance AS
SELECT 
    mm.model_type,
    mm.stage,
    mm.version,
    mm.is_active,
    mm.performance_metrics,
    COUNT(mp.id) as prediction_count,
    AVG(mp.confidence_score) as avg_confidence
FROM ml_models mm
LEFT JOIN ml_predictions mp ON mm.id = mp.model_id
GROUP BY mm.id, mm.model_type, mm.stage, mm.version, mm.is_active, mm.performance_metrics;

-- =====================================================
-- 5. CREATE FUNCTIONS (IF NOT EXISTS)
-- =====================================================

-- Function to calculate rolling averages
CREATE OR REPLACE FUNCTION calculate_rolling_average(
    ad_id_param TEXT,
    metric_param TEXT,
    window_days INTEGER,
    end_date DATE DEFAULT CURRENT_DATE
) RETURNS DECIMAL 
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    result DECIMAL;
BEGIN
    SELECT AVG(
        CASE metric_param
            WHEN 'ctr' THEN ctr
            WHEN 'cpa' THEN cpa
            WHEN 'roas' THEN roas
            WHEN 'spend' THEN spend
            ELSE 0
        END
    ) INTO result
    FROM performance_metrics
    WHERE ad_id = ad_id_param
        AND window_type = '1d'
        AND date_end BETWEEN end_date - INTERVAL '1 day' * window_days AND end_date;
    
    RETURN COALESCE(result, 0);
END;
$$ LANGUAGE plpgsql;

-- Function to detect performance trends
CREATE OR REPLACE FUNCTION detect_trend(
    ad_id_param TEXT,
    metric_param TEXT,
    window_days INTEGER DEFAULT 7
) RETURNS TEXT 
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    recent_avg DECIMAL;
    older_avg DECIMAL;
    trend_strength DECIMAL;
BEGIN
    -- Calculate recent average
    SELECT calculate_rolling_average(ad_id_param, metric_param, window_days/2, CURRENT_DATE)
    INTO recent_avg;
    
    -- Calculate older average
    SELECT calculate_rolling_average(ad_id_param, metric_param, window_days/2, CURRENT_DATE - INTERVAL '1 day' * (window_days/2))
    INTO older_avg;
    
    -- Calculate trend strength
    trend_strength := (recent_avg - older_avg) / NULLIF(older_avg, 0);
    
    RETURN CASE
        WHEN trend_strength > 0.1 THEN 'improving'
        WHEN trend_strength < -0.1 THEN 'declining'
        ELSE 'stable'
    END;
END;
$$ LANGUAGE plpgsql;

-- Update timestamps function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER 
SECURITY DEFINER
SET search_path = public
AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- 6. CREATE TRIGGERS (IF NOT EXISTS)
-- =====================================================

-- Drop existing triggers if they exist
DROP TRIGGER IF EXISTS update_ad_lifecycle_updated_at ON ad_lifecycle;
DROP TRIGGER IF EXISTS update_ml_models_updated_at ON ml_models;
DROP TRIGGER IF EXISTS update_creative_intelligence_updated_at ON creative_intelligence;

-- Apply to relevant tables
CREATE TRIGGER update_ad_lifecycle_updated_at 
    BEFORE UPDATE ON ad_lifecycle 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_ml_models_updated_at 
    BEFORE UPDATE ON ml_models 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_creative_intelligence_updated_at 
    BEFORE UPDATE ON creative_intelligence 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- 7. ENABLE RLS ON ALL TABLES
-- =====================================================

-- Enable RLS on all tables
ALTER TABLE ad_lifecycle ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_models ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE learning_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE adaptive_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_patterns ENABLE ROW LEVEL SECURITY;
ALTER TABLE time_series_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE creative_intelligence ENABLE ROW LEVEL SECURITY;
ALTER TABLE system_health ENABLE ROW LEVEL SECURITY;
ALTER TABLE account_health ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_summaries ENABLE ROW LEVEL SECURITY;
ALTER TABLE feature_engineering ENABLE ROW LEVEL SECURITY;
ALTER TABLE temporal_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE creative_similarity ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 8. CREATE RLS POLICIES
-- =====================================================

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Service role full access" ON ad_lifecycle;
DROP POLICY IF EXISTS "Service role full access" ON performance_metrics;
DROP POLICY IF EXISTS "Service role full access" ON ml_models;
DROP POLICY IF EXISTS "Service role full access" ON ml_predictions;
DROP POLICY IF EXISTS "Service role full access" ON learning_events;
DROP POLICY IF EXISTS "Service role full access" ON adaptive_rules;
DROP POLICY IF EXISTS "Service role full access" ON performance_patterns;
DROP POLICY IF EXISTS "Service role full access" ON time_series_data;
DROP POLICY IF EXISTS "Service role full access" ON creative_intelligence;
DROP POLICY IF EXISTS "Service role full access" ON system_health;
DROP POLICY IF EXISTS "Service role full access" ON account_health;
DROP POLICY IF EXISTS "Service role full access" ON daily_summaries;
DROP POLICY IF EXISTS "Service role full access" ON feature_engineering;
DROP POLICY IF EXISTS "Service role full access" ON temporal_analysis;
DROP POLICY IF EXISTS "Service role full access" ON creative_similarity;

DROP POLICY IF EXISTS "Authenticated users can read" ON ad_lifecycle;
DROP POLICY IF EXISTS "Authenticated users can read" ON performance_metrics;
DROP POLICY IF EXISTS "Authenticated users can read" ON ml_models;
DROP POLICY IF EXISTS "Authenticated users can read" ON ml_predictions;
DROP POLICY IF EXISTS "Authenticated users can read" ON learning_events;
DROP POLICY IF EXISTS "Authenticated users can read" ON adaptive_rules;
DROP POLICY IF EXISTS "Authenticated users can read" ON performance_patterns;
DROP POLICY IF EXISTS "Authenticated users can read" ON time_series_data;
DROP POLICY IF EXISTS "Authenticated users can read" ON creative_intelligence;
DROP POLICY IF EXISTS "Authenticated users can read" ON system_health;
DROP POLICY IF EXISTS "Authenticated users can read" ON account_health;
DROP POLICY IF EXISTS "Authenticated users can read" ON daily_summaries;
DROP POLICY IF EXISTS "Authenticated users can read" ON feature_engineering;
DROP POLICY IF EXISTS "Authenticated users can read" ON temporal_analysis;
DROP POLICY IF EXISTS "Authenticated users can read" ON creative_similarity;

-- Service role policies (full access)
CREATE POLICY "Service role full access" ON ad_lifecycle FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON performance_metrics FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON ml_models FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON ml_predictions FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON learning_events FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON adaptive_rules FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON performance_patterns FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON time_series_data FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON creative_intelligence FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON system_health FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON account_health FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON daily_summaries FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON feature_engineering FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON temporal_analysis FOR ALL TO service_role USING (true);
CREATE POLICY "Service role full access" ON creative_similarity FOR ALL TO service_role USING (true);

-- Authenticated user policies (read-only)
CREATE POLICY "Authenticated users can read" ON ad_lifecycle FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON performance_metrics FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON ml_models FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON ml_predictions FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON learning_events FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON adaptive_rules FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON performance_patterns FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON time_series_data FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON creative_intelligence FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON system_health FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON account_health FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON daily_summaries FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON feature_engineering FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON temporal_analysis FOR SELECT TO authenticated USING (true);
CREATE POLICY "Authenticated users can read" ON creative_similarity FOR SELECT TO authenticated USING (true);

-- =====================================================
-- 9. INSERT DEFAULT DATA (IF NOT EXISTS)
-- =====================================================

-- Insert default adaptive rules
INSERT INTO adaptive_rules (rule_name, stage, rule_type, current_value, learning_rate) 
SELECT * FROM (VALUES
    ('cpa_threshold', 'testing', 'threshold', 35.75, 0.1),
    ('cpa_threshold', 'validation', 'threshold', 30.25, 0.1),
    ('cpa_threshold', 'scaling', 'threshold', 27.50, 0.1),
    ('ctr_minimum', 'testing', 'threshold', 0.008, 0.05),
    ('roas_minimum', 'validation', 'threshold', 1.3, 0.1),
    ('roas_minimum', 'scaling', 'threshold', 2.0, 0.1),
    ('quality_score_minimum', 'testing', 'threshold', 30, 0.05),
    ('quality_score_minimum', 'validation', 'threshold', 75, 0.05),
    ('stability_threshold', 'validation', 'threshold', 70, 0.1),
    ('fatigue_threshold', 'scaling', 'threshold', 0.5, 0.1)
) AS v(rule_name, stage, rule_type, current_value, learning_rate)
WHERE NOT EXISTS (SELECT 1 FROM adaptive_rules WHERE rule_name = v.rule_name AND stage = v.stage);

-- Insert default system health
INSERT INTO system_health (stage, health_score, stability_score, confidence_score, efficiency_score)
SELECT * FROM (VALUES
    ('testing', 85, 0.8, 0.7, 0.75),
    ('validation', 90, 0.85, 0.8, 0.8),
    ('scaling', 95, 0.9, 0.85, 0.85)
) AS v(stage, health_score, stability_score, confidence_score, efficiency_score)
WHERE NOT EXISTS (SELECT 1 FROM system_health WHERE stage = v.stage);

-- =====================================================
-- 10. VERIFICATION QUERIES
-- =====================================================

-- Verify all tables exist
SELECT 'Tables created successfully' as status, count(*) as table_count
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN (
    'ad_lifecycle', 'performance_metrics', 'ml_models', 'ml_predictions',
    'learning_events', 'adaptive_rules', 'performance_patterns', 'time_series_data',
    'creative_intelligence', 'system_health', 'account_health', 'daily_summaries',
    'feature_engineering', 'temporal_analysis', 'creative_similarity'
);

-- Verify RLS is enabled
SELECT 'RLS enabled' as status, count(*) as rls_enabled_count
FROM pg_tables 
WHERE schemaname = 'public' 
AND rowsecurity = true;

-- Verify policies exist
SELECT 'Policies created' as status, count(*) as policy_count
FROM pg_policies 
WHERE schemaname = 'public';

-- Verify functions exist
SELECT 'Functions created' as status, count(*) as function_count
FROM pg_proc 
WHERE proname IN ('calculate_rolling_average', 'detect_trend', 'update_updated_at_column');

-- Verify views exist
SELECT 'Views created' as status, count(*) as view_count
FROM information_schema.views 
WHERE table_schema = 'public' 
AND table_name IN ('current_active_ads', 'ml_model_performance');
