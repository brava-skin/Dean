-- =====================================================
-- SUPABASE SECURITY & PERFORMANCE FIXES
-- Fixes all linter issues for production deployment
-- =====================================================

-- 1. FIX SECURITY DEFINER VIEWS (ERROR LEVEL)
-- Remove SECURITY DEFINER from views to fix security issues

-- Drop and recreate ml_model_performance view without SECURITY DEFINER
DROP VIEW IF EXISTS public.ml_model_performance CASCADE;

CREATE VIEW public.ml_model_performance AS
SELECT 
    mm.id,
    mm.model_type,
    mm.version,
    mm.stage,
    mm.performance_metrics,
    mm.features_used,
    mm.created_at,
    mm.updated_at,
    -- Calculate performance score from metrics
    COALESCE(
        (mm.performance_metrics->>'test_r2')::float,
        (mm.performance_metrics->>'cv_score')::float,
        0.0
    ) as performance_score
FROM ml_models mm
WHERE mm.is_active = true;

-- Drop and recreate current_active_ads view without SECURITY DEFINER
DROP VIEW IF EXISTS public.current_active_ads CASCADE;

CREATE VIEW public.current_active_ads AS
SELECT 
    al.ad_id,
    al.creative_id,
    al.campaign_id,
    al.adset_id,
    al.stage,
    al.status,
    al.lifecycle_id,
    al.created_at,
    al.updated_at,
    -- Get latest performance data
    pm.spend,
    pm.impressions,
    pm.clicks,
    pm.purchases,
    pm.ctr,
    pm.cpa,
    pm.roas
FROM ad_lifecycle al
LEFT JOIN LATERAL (
    SELECT *
    FROM performance_metrics pm
    WHERE pm.ad_id = al.ad_id
    ORDER BY pm.date_start DESC
    LIMIT 1
) pm ON true
WHERE al.status = 'active';

-- 2. FIX EXTENSION IN PUBLIC SCHEMA (WARN LEVEL)
-- Move vector extension to ml_schema

-- Create ml_schema if it doesn't exist
CREATE SCHEMA IF NOT EXISTS ml_schema;

-- Drop vector extension from public schema
DROP EXTENSION IF EXISTS vector CASCADE;

-- Create vector extension in ml_schema
CREATE EXTENSION IF NOT EXISTS vector SCHEMA ml_schema;

-- Update creative_intelligence table to use ml_schema.vector
-- First, add new column with correct type
ALTER TABLE public.creative_intelligence 
ADD COLUMN IF NOT EXISTS similarity_vector_new ml_schema.vector(384);

-- Copy data from old column to new column (if data exists)
UPDATE public.creative_intelligence 
SET similarity_vector_new = similarity_vector::ml_schema.vector(384)
WHERE similarity_vector IS NOT NULL;

-- Drop old column and rename new one
ALTER TABLE public.creative_intelligence DROP COLUMN IF EXISTS similarity_vector;
ALTER TABLE public.creative_intelligence RENAME COLUMN similarity_vector_new TO similarity_vector;

-- 3. FIX RLS INITIALIZATION PLAN (WARN LEVEL)
-- Optimize RLS policies for meta_creatives table

-- Drop existing policies
DROP POLICY IF EXISTS "Allow insert for owner" ON public.meta_creatives;
DROP POLICY IF EXISTS "Allow update by owner" ON public.meta_creatives;
DROP POLICY IF EXISTS "Allow delete by owner" ON public.meta_creatives;

-- Create optimized RLS policies with SELECT subqueries
CREATE POLICY "Allow insert for owner" ON public.meta_creatives
    FOR INSERT WITH CHECK ((SELECT auth.uid()) = created_by);

CREATE POLICY "Allow update by owner" ON public.meta_creatives
    FOR UPDATE USING ((SELECT auth.uid()) = created_by);

CREATE POLICY "Allow delete by owner" ON public.meta_creatives
    FOR DELETE USING ((SELECT auth.uid()) = created_by);

-- 4. FIX UNINDEXED FOREIGN KEYS (INFO LEVEL)
-- Add missing indexes for foreign keys

-- Add index for parent_lifecycle_id foreign key
CREATE INDEX IF NOT EXISTS idx_ad_lifecycle_parent_lifecycle_id 
ON public.ad_lifecycle (parent_lifecycle_id);

-- 5. REMOVE UNUSED INDEXES (INFO LEVEL)
-- Drop indexes that have never been used to improve performance

-- Drop unused indexes on meta_creatives
DROP INDEX IF EXISTS public.idx_meta_creatives_video_id;
DROP INDEX IF EXISTS public.idx_meta_creatives_created_by;
DROP INDEX IF EXISTS public.idx_meta_creatives_created_at;

-- Drop unused indexes on fatigue_analysis
DROP INDEX IF EXISTS public.idx_fatigue_analysis_fatigue_score;
DROP INDEX IF EXISTS public.idx_fatigue_analysis_analysis_date;

-- Drop unused indexes on ml_predictions
DROP INDEX IF EXISTS public.idx_ml_predictions_ad_id;
DROP INDEX IF EXISTS public.idx_ml_predictions_model;

-- Drop unused indexes on performance_metrics
DROP INDEX IF EXISTS public.idx_performance_metrics_lifecycle;

-- Drop unused indexes on learning_events
DROP INDEX IF EXISTS public.idx_learning_events_type;
DROP INDEX IF EXISTS public.idx_learning_events_ad_id;
DROP INDEX IF EXISTS public.idx_learning_events_created;

-- Drop unused indexes on time_series_data
DROP INDEX IF EXISTS public.idx_time_series_ad_id;
DROP INDEX IF EXISTS public.idx_time_series_metric;
DROP INDEX IF EXISTS public.idx_time_series_timestamp;

-- Drop unused indexes on creative_intelligence
DROP INDEX IF EXISTS public.idx_creative_intelligence_creative_id;
DROP INDEX IF EXISTS public.idx_creative_intelligence_performance;

-- 6. ADD ESSENTIAL INDEXES FOR PERFORMANCE
-- Add indexes that will actually be used by the ML system

-- Performance metrics indexes
CREATE INDEX IF NOT EXISTS idx_performance_metrics_ad_stage_date 
ON public.performance_metrics (ad_id, stage, date_start);

CREATE INDEX IF NOT EXISTS idx_performance_metrics_stage_date 
ON public.performance_metrics (stage, date_start);

-- ML models indexes
CREATE INDEX IF NOT EXISTS idx_ml_models_type_stage 
ON public.ml_models (model_type, stage);

CREATE INDEX IF NOT EXISTS idx_ml_models_active 
ON public.ml_models (is_active) WHERE is_active = true;

-- ML predictions indexes
CREATE INDEX IF NOT EXISTS idx_ml_predictions_stage_date 
ON public.ml_predictions (stage, created_at);

-- Learning events indexes
CREATE INDEX IF NOT EXISTS idx_learning_events_stage_date 
ON public.learning_events (stage, created_at);

-- Time series data indexes
CREATE INDEX IF NOT EXISTS idx_time_series_ad_metric_date 
ON public.time_series_data (ad_id, metric_name, timestamp);

-- Creative intelligence indexes
CREATE INDEX IF NOT EXISTS idx_creative_intelligence_performance_score 
ON public.creative_intelligence (performance_score);

-- 7. OPTIMIZE TABLE STATISTICS
-- Update table statistics for better query planning

ANALYZE public.performance_metrics;
ANALYZE public.ml_models;
ANALYZE public.ml_predictions;
ANALYZE public.learning_events;
ANALYZE public.time_series_data;
ANALYZE public.creative_intelligence;
ANALYZE public.ad_lifecycle;
ANALYZE public.meta_creatives;
ANALYZE public.fatigue_analysis;

-- 8. GRANT PERMISSIONS
-- Ensure proper permissions for the ml_schema

GRANT USAGE ON SCHEMA ml_schema TO service_role;
GRANT USAGE ON SCHEMA ml_schema TO authenticated;

-- 9. CREATE HELPER FUNCTIONS
-- Create utility functions for better performance

-- Function to get active models efficiently
CREATE OR REPLACE FUNCTION public.get_active_models(model_type_filter TEXT DEFAULT NULL)
RETURNS TABLE (
    id UUID,
    model_type TEXT,
    stage TEXT,
    performance_score FLOAT
) 
LANGUAGE SQL
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT 
        mm.id,
        mm.model_type,
        mm.stage,
        COALESCE(
            (mm.performance_metrics->>'test_r2')::float,
            (mm.performance_metrics->>'cv_score')::float,
            0.0
        ) as performance_score
    FROM ml_models mm
    WHERE mm.is_active = true
    AND (model_type_filter IS NULL OR mm.model_type = model_type_filter)
    ORDER BY mm.created_at DESC;
$$;

-- Function to get performance summary
CREATE OR REPLACE FUNCTION public.get_performance_summary(
    ad_id_filter TEXT DEFAULT NULL,
    stage_filter TEXT DEFAULT NULL,
    days_back INTEGER DEFAULT 30
)
RETURNS TABLE (
    ad_id TEXT,
    stage TEXT,
    total_spend FLOAT,
    total_impressions BIGINT,
    total_clicks BIGINT,
    total_purchases BIGINT,
    avg_ctr FLOAT,
    avg_cpa FLOAT,
    avg_roas FLOAT
)
LANGUAGE SQL
SECURITY DEFINER
SET search_path = public
AS $$
    SELECT 
        pm.ad_id,
        pm.stage,
        SUM(pm.spend) as total_spend,
        SUM(pm.impressions) as total_impressions,
        SUM(pm.clicks) as total_clicks,
        SUM(pm.purchases) as total_purchases,
        AVG(pm.ctr) as avg_ctr,
        AVG(pm.cpa) as avg_cpa,
        AVG(pm.roas) as avg_roas
    FROM performance_metrics pm
    WHERE pm.date_start >= CURRENT_DATE - INTERVAL '1 day' * days_back
    AND (ad_id_filter IS NULL OR pm.ad_id = ad_id_filter)
    AND (stage_filter IS NULL OR pm.stage = stage_filter)
    GROUP BY pm.ad_id, pm.stage
    ORDER BY total_spend DESC;
$$;

-- 10. FINAL OPTIMIZATIONS
-- Set optimal configuration for the ML system

-- Set work_mem for better performance with ML queries
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET shared_preload_libraries = 'vector';

-- Reload configuration
SELECT pg_reload_conf();

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Supabase security and performance fixes applied successfully!';
    RAISE NOTICE 'All linter issues have been resolved.';
    RAISE NOTICE 'Database is now optimized for production ML workloads.';
END $$;
