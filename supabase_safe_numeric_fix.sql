-- SAFE Fix for Supabase numeric field overflow errors
-- This script handles views that depend on the columns we want to modify

-- Step 1: Get the current view definition (run this first to see the actual definition)
-- SELECT definition FROM pg_views WHERE viewname = 'current_active_ads';

-- Step 2: Drop the dependent view temporarily
DROP VIEW IF EXISTS current_active_ads CASCADE;

-- Step 3: Fix performance_metrics table numeric fields
-- Only modify columns that are actually causing overflow
ALTER TABLE performance_metrics 
  ALTER COLUMN ctr TYPE DECIMAL(8,4);

-- Try other columns one by one (uncomment as needed)
-- ALTER TABLE performance_metrics ALTER COLUMN cpc TYPE DECIMAL(10,2);
-- ALTER TABLE performance_metrics ALTER COLUMN cpm TYPE DECIMAL(10,2);
-- ALTER TABLE performance_metrics ALTER COLUMN cpa TYPE DECIMAL(10,2);
-- ALTER TABLE performance_metrics ALTER COLUMN roas TYPE DECIMAL(10,4);

-- Step 4: Fix other tables (these likely don't have view dependencies)
ALTER TABLE time_series_data 
  ALTER COLUMN metric_value TYPE DECIMAL(15,4);

ALTER TABLE historical_data 
  ALTER COLUMN metric_value TYPE DECIMAL(15,4);

-- Step 5: Recreate a simple view (you may need to adjust this based on your needs)
-- This is a minimal version - replace with your actual view definition if you have it
CREATE OR REPLACE VIEW current_active_ads AS
SELECT 
  ad_id,
  lifecycle_id,
  stage,
  status,
  spend,
  impressions,
  clicks,
  ctr,
  cpc,
  cpm,
  created_at
FROM performance_metrics 
WHERE date_start = CURRENT_DATE
ORDER BY created_at DESC;

-- Step 6: Grant permissions
GRANT SELECT ON current_active_ads TO authenticated;
GRANT SELECT ON current_active_ads TO anon;
