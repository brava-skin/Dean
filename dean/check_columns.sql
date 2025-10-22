-- Check all column names in Supabase database
-- This script will show you the actual schema structure

-- 1. List all tables and their columns
SELECT 
    table_name,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns 
WHERE table_schema = 'public'
ORDER BY table_name, ordinal_position;

-- 2. Check specific tables that might have issues
SELECT '=== PERFORMANCE_METRICS ===' as table_check;
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'performance_metrics' AND table_schema = 'public'
ORDER BY ordinal_position;

SELECT '=== ML_MODELS ===' as table_check;
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ml_models' AND table_schema = 'public'
ORDER BY ordinal_position;

SELECT '=== LEARNING_EVENTS ===' as table_check;
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'learning_events' AND table_schema = 'public'
ORDER BY ordinal_position;

SELECT '=== CREATIVE_INTELLIGENCE ===' as table_check;
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'creative_intelligence' AND table_schema = 'public'
ORDER BY ordinal_position;

SELECT '=== AD_LIFECYCLE ===' as table_check;
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ad_lifecycle' AND table_schema = 'public'
ORDER BY ordinal_position;

SELECT '=== ML_PREDICTIONS ===' as table_check;
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ml_predictions' AND table_schema = 'public'
ORDER BY ordinal_position;

-- 3. Check for any views
SELECT '=== VIEWS ===' as view_check;
SELECT table_name, table_type
FROM information_schema.tables 
WHERE table_schema = 'public' AND table_type = 'VIEW';

-- 4. Check for any functions
SELECT '=== FUNCTIONS ===' as function_check;
SELECT routine_name, routine_type
FROM information_schema.routines 
WHERE routine_schema = 'public';
