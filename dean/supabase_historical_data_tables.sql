-- =====================================================
-- SUPABASE HISTORICAL DATA TABLES (OPTIONAL)
-- =====================================================
-- These tables are OPTIONAL and provide Supabase storage
-- for historical data tracking. The system works with
-- local SQLite storage, but these tables enable better
-- analytics and reporting in Supabase.
-- =====================================================

-- Historical data tracking for rule evaluation
CREATE TABLE IF NOT EXISTS historical_data (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ad_id TEXT NOT NULL,
    lifecycle_id TEXT,
    stage TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    metric_value REAL NOT NULL,
    ts_epoch INTEGER NOT NULL,
    ts_iso TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Ad creation times for time-based rules
CREATE TABLE IF NOT EXISTS ad_creation_times (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    ad_id TEXT UNIQUE NOT NULL,
    lifecycle_id TEXT,
    stage TEXT NOT NULL,
    created_at_epoch INTEGER NOT NULL,
    created_at_iso TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes separately (PostgreSQL syntax)
CREATE INDEX IF NOT EXISTS idx_historical_data_ad_metric ON historical_data (ad_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_historical_data_lifecycle_metric ON historical_data (lifecycle_id, metric_name);
CREATE INDEX IF NOT EXISTS idx_historical_data_stage_metric ON historical_data (stage, metric_name);
CREATE INDEX IF NOT EXISTS idx_historical_data_timestamp ON historical_data (ts_epoch);

CREATE INDEX IF NOT EXISTS idx_ad_creation_times_stage ON ad_creation_times (stage);
CREATE INDEX IF NOT EXISTS idx_ad_creation_times_lifecycle ON ad_creation_times (lifecycle_id);
CREATE INDEX IF NOT EXISTS idx_ad_creation_times_created ON ad_creation_times (created_at_epoch);

-- Enhanced performance metrics with historical tracking
ALTER TABLE performance_metrics 
ADD COLUMN IF NOT EXISTS ad_age_days REAL DEFAULT 0;

-- Add indexes for better performance on existing tables
CREATE INDEX IF NOT EXISTS idx_performance_metrics_ad_stage_date 
ON performance_metrics (ad_id, stage, date_start);

CREATE INDEX IF NOT EXISTS idx_ad_lifecycle_ad_stage 
ON ad_lifecycle (ad_id, stage);

-- =====================================================
-- ROW LEVEL SECURITY (RLS) POLICIES
-- =====================================================

-- Enable RLS on new tables
ALTER TABLE historical_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE ad_creation_times ENABLE ROW LEVEL SECURITY;

-- Create policies (adjust based on your security requirements)
-- These policies allow service role to access all data
CREATE POLICY "Service role can access historical_data" ON historical_data
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can access ad_creation_times" ON ad_creation_times
    FOR ALL USING (auth.role() = 'service_role');

-- =====================================================
-- CLEANUP FUNCTIONS
-- =====================================================

-- Function to clean up old historical data (keep last 30 days)
CREATE OR REPLACE FUNCTION cleanup_old_historical_data()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM historical_data 
    WHERE ts_epoch < EXTRACT(EPOCH FROM NOW() - INTERVAL '30 days');
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- USAGE NOTES
-- =====================================================
-- 
-- 1. These tables are OPTIONAL - the system works with local SQLite storage
-- 2. If you want to use these tables, update the Store class methods to also
--    store data in Supabase
-- 3. The cleanup function can be called periodically to manage storage costs
-- 4. Adjust RLS policies based on your security requirements
--
-- To use these tables, you would need to modify the Store class methods
-- to also insert data into Supabase in addition to local SQLite storage.
-- =====================================================
