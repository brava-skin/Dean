-- Creative Intelligence Schema for Dean
-- Enhanced creative management with performance tracking, ML analysis, and AI generation

-- Creative Library Table
CREATE TABLE IF NOT EXISTS creative_library (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    creative_id TEXT UNIQUE NOT NULL,
    creative_type TEXT NOT NULL, -- 'primary_text', 'headline', 'description'
    content TEXT NOT NULL,
    category TEXT, -- 'global', 'product_specific', 'seasonal', etc.
    tags TEXT[], -- Array of tags for categorization
    performance_score FLOAT DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_by TEXT DEFAULT 'system', -- 'system', 'user', 'ai_generated'
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Creative Performance Tracking
CREATE TABLE IF NOT EXISTS creative_performance (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    creative_id TEXT NOT NULL,
    ad_id TEXT NOT NULL,
    stage TEXT NOT NULL, -- 'testing', 'validation', 'scaling'
    date_start DATE NOT NULL,
    date_end DATE NOT NULL,
    
    -- Performance Metrics
    impressions INTEGER DEFAULT 0,
    clicks INTEGER DEFAULT 0,
    spend FLOAT DEFAULT 0.0,
    purchases INTEGER DEFAULT 0,
    add_to_cart INTEGER DEFAULT 0,
    initiate_checkout INTEGER DEFAULT 0,
    
    -- Calculated Metrics
    ctr FLOAT DEFAULT 0.0,
    cpc FLOAT DEFAULT 0.0,
    cpm FLOAT DEFAULT 0.0,
    roas FLOAT DEFAULT 0.0,
    cpa FLOAT DEFAULT 0.0,
    
    -- Creative-specific metrics
    engagement_rate FLOAT DEFAULT 0.0,
    conversion_rate FLOAT DEFAULT 0.0,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Foreign key relationships
    FOREIGN KEY (creative_id) REFERENCES creative_library(creative_id),
    UNIQUE(creative_id, ad_id, date_start)
);

-- Creative Combinations (for mix-and-match tracking)
CREATE TABLE IF NOT EXISTS creative_combinations (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    combination_id TEXT UNIQUE NOT NULL, -- Generated ID for this specific combination
    primary_text_id TEXT NOT NULL,
    headline_id TEXT NOT NULL,
    description_id TEXT NOT NULL,
    
    -- Performance tracking
    total_ads INTEGER DEFAULT 0,
    total_spend FLOAT DEFAULT 0.0,
    total_purchases INTEGER DEFAULT 0,
    avg_roas FLOAT DEFAULT 0.0,
    avg_cpa FLOAT DEFAULT 0.0,
    
    -- Success metrics
    success_rate FLOAT DEFAULT 0.0,
    performance_rank INTEGER,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (primary_text_id) REFERENCES creative_library(creative_id),
    FOREIGN KEY (headline_id) REFERENCES creative_library(creative_id),
    FOREIGN KEY (description_id) REFERENCES creative_library(creative_id)
);

-- AI-Generated Creatives
CREATE TABLE IF NOT EXISTS ai_generated_creatives (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    creative_id TEXT UNIQUE NOT NULL,
    source_creative_id TEXT, -- Reference to original creative that inspired this
    generation_prompt TEXT NOT NULL,
    generation_model TEXT DEFAULT 'gpt-4',
    generation_parameters JSONB DEFAULT '{}'::jsonb,
    
    -- Generated content
    content TEXT NOT NULL,
    creative_type TEXT NOT NULL,
    category TEXT,
    tags TEXT[],
    
    -- Performance tracking
    performance_score FLOAT DEFAULT 0.0,
    usage_count INTEGER DEFAULT 0,
    is_approved BOOLEAN DEFAULT FALSE,
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (source_creative_id) REFERENCES creative_library(creative_id)
);

-- Creative Similarity Analysis
CREATE TABLE IF NOT EXISTS creative_similarity (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    creative_id_1 TEXT NOT NULL,
    creative_id_2 TEXT NOT NULL,
    similarity_score FLOAT NOT NULL,
    similarity_type TEXT DEFAULT 'semantic', -- 'semantic', 'performance', 'structural'
    analysis_model TEXT DEFAULT 'sentence-transformers',
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (creative_id_1) REFERENCES creative_library(creative_id),
    FOREIGN KEY (creative_id_2) REFERENCES creative_library(creative_id),
    UNIQUE(creative_id_1, creative_id_2, similarity_type)
);

-- Creative Performance Patterns (ML insights)
CREATE TABLE IF NOT EXISTS creative_patterns (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    pattern_type TEXT NOT NULL, -- 'high_performing', 'trending', 'seasonal', 'audience_specific'
    pattern_data JSONB NOT NULL,
    confidence_score FLOAT DEFAULT 0.0,
    sample_size INTEGER DEFAULT 0,
    
    -- Pattern details
    description TEXT,
    insights TEXT[],
    recommendations TEXT[],
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Creative A/B Test Results
CREATE TABLE IF NOT EXISTS creative_ab_tests (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    test_id TEXT UNIQUE NOT NULL,
    test_name TEXT NOT NULL,
    test_type TEXT NOT NULL, -- 'headline', 'description', 'primary_text', 'combination'
    
    -- Test configuration
    control_creative_id TEXT NOT NULL,
    variant_creative_id TEXT NOT NULL,
    test_start_date DATE NOT NULL,
    test_end_date DATE,
    
    -- Results
    control_performance JSONB DEFAULT '{}'::jsonb,
    variant_performance JSONB DEFAULT '{}'::jsonb,
    statistical_significance FLOAT DEFAULT 0.0,
    winner TEXT, -- 'control', 'variant', 'inconclusive'
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    FOREIGN KEY (control_creative_id) REFERENCES creative_library(creative_id),
    FOREIGN KEY (variant_creative_id) REFERENCES creative_library(creative_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_creative_library_type ON creative_library(creative_type);
CREATE INDEX IF NOT EXISTS idx_creative_library_performance ON creative_library(performance_score DESC);
CREATE INDEX IF NOT EXISTS idx_creative_performance_date ON creative_performance(date_start, date_end);
CREATE INDEX IF NOT EXISTS idx_creative_performance_creative ON creative_performance(creative_id);
CREATE INDEX IF NOT EXISTS idx_creative_performance_stage ON creative_performance(stage);
CREATE INDEX IF NOT EXISTS idx_creative_combinations_performance ON creative_combinations(avg_roas DESC);
CREATE INDEX IF NOT EXISTS idx_ai_generated_approved ON ai_generated_creatives(is_approved);
CREATE INDEX IF NOT EXISTS idx_creative_similarity_score ON creative_similarity(similarity_score DESC);
CREATE INDEX IF NOT EXISTS idx_creative_patterns_type ON creative_patterns(pattern_type);

-- Triggers for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_creative_library_updated_at BEFORE UPDATE ON creative_library FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_creative_performance_updated_at BEFORE UPDATE ON creative_performance FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_creative_combinations_updated_at BEFORE UPDATE ON creative_combinations FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_ai_generated_creatives_updated_at BEFORE UPDATE ON ai_generated_creatives FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_creative_patterns_updated_at BEFORE UPDATE ON creative_patterns FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_creative_ab_tests_updated_at BEFORE UPDATE ON creative_ab_tests FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
