-- Fix Supabase numeric field overflow errors
-- Issue: Fields with precision(5,4) can only store values up to 9.9999
-- Solution: Increase precision to handle larger values

-- Fix performance_metrics table numeric fields
ALTER TABLE performance_metrics 
  ALTER COLUMN ctr TYPE DECIMAL(8,4),
  ALTER COLUMN cpc TYPE DECIMAL(10,2),
  ALTER COLUMN cpm TYPE DECIMAL(10,2),
  ALTER COLUMN cpa TYPE DECIMAL(10,2),
  ALTER COLUMN roas TYPE DECIMAL(10,4);

-- Fix time_series_data table if it has similar issues
ALTER TABLE time_series_data 
  ALTER COLUMN metric_value TYPE DECIMAL(15,4);

-- Fix historical_data table if it has similar issues  
ALTER TABLE historical_data 
  ALTER COLUMN metric_value TYPE DECIMAL(15,4);

-- Add any other tables that might have numeric overflow issues
-- (Add more ALTER statements as needed based on actual schema)

-- Update any existing data that might have been truncated
-- (This is optional - existing data should be fine)
