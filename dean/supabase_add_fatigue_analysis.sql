-- Add fatigue_analysis table to existing Supabase database
-- Run this script to add the missing fatigue_analysis table

-- Fatigue analysis and decay tracking
CREATE TABLE IF NOT EXISTS fatigue_analysis (
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

-- Add RLS policies
CREATE POLICY "Service role full access" ON fatigue_analysis FOR ALL TO service_role USING (true);
CREATE POLICY "Authenticated users can read" ON fatigue_analysis FOR SELECT TO authenticated USING (true);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_fatigue_analysis_ad_id ON fatigue_analysis(ad_id);
CREATE INDEX IF NOT EXISTS idx_fatigue_analysis_fatigue_score ON fatigue_analysis(fatigue_score);
CREATE INDEX IF NOT EXISTS idx_fatigue_analysis_analysis_date ON fatigue_analysis(analysis_date);
