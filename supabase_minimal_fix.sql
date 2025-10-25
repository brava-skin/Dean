-- MINIMAL Fix for Supabase numeric field overflow errors
-- This script only fixes tables that don't have view dependencies
-- The code-level safe_float() function will handle the overflow prevention

-- Fix time_series_data table metric values
ALTER TABLE time_series_data 
  ALTER COLUMN metric_value TYPE DECIMAL(15,4);

-- Fix historical_data table metric values  
ALTER TABLE historical_data 
  ALTER COLUMN metric_value TYPE DECIMAL(15,4);

-- Note: We're NOT modifying performance_metrics columns (ctr, cpc, cpm, etc.)
-- because they're used by views. Instead, the safe_float() function in the code
-- will bound values to prevent overflow:
-- - CTR will be capped at 9.9999% 
-- - CPC/CPM will be capped at reasonable values
-- - This prevents the overflow while maintaining compatibility
