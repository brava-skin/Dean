-- Fix performance_metrics upsert constraint
-- Add UNIQUE constraint on (ad_id, window_type, date_start) for ON CONFLICT to work

-- Drop existing constraint if it exists
ALTER TABLE performance_metrics 
DROP CONSTRAINT IF EXISTS performance_metrics_ad_id_window_type_date_start_key;

-- Add UNIQUE constraint
ALTER TABLE performance_metrics 
ADD CONSTRAINT performance_metrics_ad_id_window_type_date_start_key 
UNIQUE (ad_id, window_type, date_start);

-- Verify the constraint was created
SELECT 
    conname AS constraint_name,
    contype AS constraint_type,
    pg_get_constraintdef(oid) AS constraint_definition
FROM pg_constraint
WHERE conrelid = 'performance_metrics'::regclass
AND conname = 'performance_metrics_ad_id_window_type_date_start_key';

