-- Fix missing model_name column in ml_models table
-- This column is required but was missing from the schema

-- Add the missing model_name column
ALTER TABLE ml_models ADD COLUMN IF NOT EXISTS model_name VARCHAR(255);

-- Make it NOT NULL with a default value for existing records
UPDATE ml_models SET model_name = CONCAT(model_type, '_', stage, '_v', version) WHERE model_name IS NULL;

-- Now make it NOT NULL
ALTER TABLE ml_models ALTER COLUMN model_name SET NOT NULL;

-- Verify the fix
SELECT 
    'ml_models' as table_name,
    COUNT(*) as total_rows,
    COUNT(model_name) as non_null_model_names,
    COUNT(*) - COUNT(model_name) as null_model_names
FROM ml_models;

-- Show sample data
SELECT model_type, stage, version, model_name, created_at 
FROM ml_models 
ORDER BY created_at DESC 
LIMIT 5;
