-- Add only the missing fatigue_analysis table to existing Supabase database
-- This script only adds what's missing, avoiding "relation already exists" errors

-- Check if fatigue_analysis table exists, if not create it
DO $$
BEGIN
    -- Only create the table if it doesn't exist
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'fatigue_analysis') THEN
        -- Fatigue analysis and decay tracking
        CREATE TABLE fatigue_analysis (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            ad_id TEXT NOT NULL,
            lifecycle_id TEXT,
            fatigue_score DECIMAL(5,4) NOT NULL CHECK (fatigue_score >= 0 AND fatigue_score <= 1),
            decay_rate DECIMAL(10,6),
            half_life_days DECIMAL(8,2),
            performance_trend TEXT CHECK (performance_trend IN ('improving', 'stable', 'declining', 'volatile')),
            fatigue_factors JSONB,
            recommended_actions JSONB,
            analysis_date TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );

        -- Enable RLS
        ALTER TABLE fatigue_analysis ENABLE ROW LEVEL SECURITY;

        -- Add RLS policies (only if they don't exist)
        IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'fatigue_analysis' AND policyname = 'Service role full access') THEN
            CREATE POLICY "Service role full access" ON fatigue_analysis FOR ALL TO service_role USING (true);
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename = 'fatigue_analysis' AND policyname = 'Authenticated users can read') THEN
            CREATE POLICY "Authenticated users can read" ON fatigue_analysis FOR SELECT TO authenticated USING (true);
        END IF;

        -- Add indexes for performance (only if they don't exist)
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_fatigue_analysis_ad_id') THEN
            CREATE INDEX idx_fatigue_analysis_ad_id ON fatigue_analysis(ad_id);
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_fatigue_analysis_fatigue_score') THEN
            CREATE INDEX idx_fatigue_analysis_fatigue_score ON fatigue_analysis(fatigue_score);
        END IF;
        
        IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname = 'idx_fatigue_analysis_analysis_date') THEN
            CREATE INDEX idx_fatigue_analysis_analysis_date ON fatigue_analysis(analysis_date);
        END IF;

        RAISE NOTICE 'fatigue_analysis table created successfully';
    ELSE
        RAISE NOTICE 'fatigue_analysis table already exists, skipping creation';
    END IF;
END $$;
