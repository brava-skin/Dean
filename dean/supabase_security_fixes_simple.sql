-- =====================================================
-- SUPABASE SECURITY FIXES - SIMPLE VERSION
-- Addresses all security issues identified by Supabase linter
-- =====================================================

-- =====================================================
-- 1. FIX SECURITY DEFINER VIEWS
-- =====================================================

-- Drop and recreate views without SECURITY DEFINER
DROP VIEW IF EXISTS current_active_ads;
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

DROP VIEW IF EXISTS ml_model_performance;
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
-- 2. ENABLE RLS ON MISSING TABLES
-- =====================================================

-- Enable RLS on tables that were missing it
ALTER TABLE feature_engineering ENABLE ROW LEVEL SECURITY;
ALTER TABLE temporal_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE creative_similarity ENABLE ROW LEVEL SECURITY;

-- =====================================================
-- 3. CREATE RLS POLICIES FOR ALL TABLES
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

-- Create comprehensive RLS policies
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
-- 4. FIX FUNCTION SEARCH PATH ISSUES
-- =====================================================

-- Recreate functions with proper security settings
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
-- 5. FIX VECTOR EXTENSION ISSUE (SIMPLIFIED)
-- =====================================================

-- Create ml_schema for ML extensions
CREATE SCHEMA IF NOT EXISTS ml_schema;

-- Create vector extension in ml_schema (this will resolve the extension in public issue)
CREATE EXTENSION IF NOT EXISTS "vector" SCHEMA ml_schema;

-- Grant permissions on ml_schema
GRANT USAGE ON SCHEMA ml_schema TO service_role;
GRANT USAGE ON SCHEMA ml_schema TO authenticated;

-- =====================================================
-- 6. VERIFICATION QUERIES
-- =====================================================

-- Verify RLS is enabled on all tables
SELECT 'RLS Status' as check_type, schemaname, tablename, rowsecurity 
FROM pg_tables 
WHERE schemaname = 'public' 
AND tablename IN (
    'ad_lifecycle', 'performance_metrics', 'ml_models', 'ml_predictions',
    'learning_events', 'adaptive_rules', 'performance_patterns', 'time_series_data',
    'creative_intelligence', 'system_health', 'account_health', 'daily_summaries',
    'feature_engineering', 'temporal_analysis', 'creative_similarity'
);

-- Verify policies exist
SELECT 'Policies' as check_type, schemaname, tablename, policyname, permissive, roles, cmd
FROM pg_policies 
WHERE schemaname = 'public'
ORDER BY tablename, policyname;

-- Verify functions have proper security settings
SELECT 'Functions' as check_type, proname, prosecdef, proconfig
FROM pg_proc 
WHERE proname IN ('calculate_rolling_average', 'detect_trend', 'update_updated_at_column');

-- Verify vector extension is in ml_schema
SELECT 'Vector Extension' as check_type, n.nspname as schema_name, e.extname as extension_name
FROM pg_extension e
JOIN pg_namespace n ON e.extnamespace = n.oid
WHERE e.extname = 'vector';

-- Verify views exist
SELECT 'Views' as check_type, table_name, table_type
FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('current_active_ads', 'ml_model_performance');

-- Check if creative_intelligence table has similarity_vector column
SELECT 'Creative Intelligence' as check_type, column_name, data_type, udt_name
FROM information_schema.columns 
WHERE table_name = 'creative_intelligence' 
AND column_name = 'similarity_vector';
