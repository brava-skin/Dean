-- Fix Supabase numeric field overflow errors
-- Issue: Fields with precision(5,4) can only store values up to 9.9999
-- Solution: Increase precision to handle larger values
-- Note: Some columns are used by views, so we need to handle them carefully

-- Step 1: Drop the view that depends on the columns we want to modify
DROP VIEW IF EXISTS current_active_ads CASCADE;

-- Step 2: Fix performance_metrics table numeric fields
ALTER TABLE performance_metrics 
  ALTER COLUMN ctr TYPE DECIMAL(8,4),
  ALTER COLUMN cpc TYPE DECIMAL(10,2),
  ALTER COLUMN cpm TYPE DECIMAL(10,2),
  ALTER COLUMN cpa TYPE DECIMAL(10,2),
  ALTER COLUMN roas TYPE DECIMAL(10,4);

-- Step 3: Recreate the view with the updated column types
-- (This is a basic recreation - you may need to adjust based on your actual view definition)
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
  purchases,
  add_to_cart,
  created_at,
  updated_at
FROM performance_metrics 
WHERE status = 'ACTIVE' 
  AND date_start = CURRENT_DATE;

-- Step 4: Fix time_series_data table if it has similar issues
ALTER TABLE time_series_data 
  ALTER COLUMN metric_value TYPE DECIMAL(15,4);

-- Step 5: Fix historical_data table if it has similar issues  
ALTER TABLE historical_data 
  ALTER COLUMN metric_value TYPE DECIMAL(15,4);

-- Step 6: Grant necessary permissions on the recreated view
GRANT SELECT ON current_active_ads TO authenticated;
GRANT SELECT ON current_active_ads TO anon;
