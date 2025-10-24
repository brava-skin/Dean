"""
Creative Intelligence System for Dean
Advanced creative management with performance tracking, ML analysis, and AI generation
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging

# Optional imports for advanced features
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from supabase import create_client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False


@dataclass
class CreativePerformance:
    """Creative performance metrics."""
    creative_id: str
    impressions: int = 0
    clicks: int = 0
    spend: float = 0.0
    purchases: int = 0
    add_to_cart: int = 0
    initiate_checkout: int = 0
    ctr: float = 0.0
    cpc: float = 0.0
    cpm: float = 0.0
    roas: float = 0.0
    cpa: float = 0.0
    engagement_rate: float = 0.0
    conversion_rate: float = 0.0


@dataclass
class CreativeInsight:
    """Creative performance insight."""
    insight_type: str
    description: str
    confidence: float
    recommendations: List[str]
    metadata: Dict[str, Any]


class CreativeIntelligenceSystem:
    """Advanced creative intelligence system with ML and AI capabilities."""
    
    def __init__(self, supabase_client=None, openai_api_key=None):
        self.supabase_client = supabase_client
        self.openai_api_key = openai_api_key
        self.logger = logging.getLogger(__name__)
        
        # Initialize AI models
        self.sentence_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                self.logger.warning(f"Failed to load sentence transformer: {e}")
        
        # Initialize OpenAI
        if OPENAI_AVAILABLE and openai_api_key:
            openai.api_key = openai_api_key
    
    def load_copy_bank_to_supabase(self, copy_bank_path: str = "dean/data/copy_bank.json") -> bool:
        """Load copy bank data into Supabase creative library."""
        try:
            with open(copy_bank_path, 'r') as f:
                copy_bank = json.load(f)
            
            if not self.supabase_client:
                self.logger.error("No Supabase client available")
                return False
            
            # Load primary texts
            for i, text in enumerate(copy_bank.get("global", {}).get("primary_texts", [])):
                creative_id = f"primary_text_{i+1}"
                self._upsert_creative(
                    creative_id=creative_id,
                    creative_type="primary_text",
                    content=text,
                    category="global",
                    created_by="system"
                )
            
            # Load headlines
            for i, headline in enumerate(copy_bank.get("global", {}).get("headlines", [])):
                creative_id = f"headline_{i+1}"
                self._upsert_creative(
                    creative_id=creative_id,
                    creative_type="headline",
                    content=headline,
                    category="global",
                    created_by="system"
                )
            
            # Load descriptions
            for i, description in enumerate(copy_bank.get("global", {}).get("descriptions", [])):
                creative_id = f"description_{i+1}"
                self._upsert_creative(
                    creative_id=creative_id,
                    creative_type="description",
                    content=description,
                    category="global",
                    created_by="system"
                )
            
            self.logger.info("Successfully loaded copy bank to Supabase")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load copy bank to Supabase: {e}")
            return False
    
    def _upsert_creative(self, creative_id: str, creative_type: str, content: str, 
                        category: str = "global", created_by: str = "system") -> bool:
        """Upsert creative to Supabase."""
        try:
            creative_data = {
                "creative_id": creative_id,
                "creative_type": creative_type,
                "content": content,
                "category": category,
                "created_by": created_by,
                "metadata": {}
            }
            
            result = self.supabase_client.table('creative_library').upsert(
                creative_data,
                on_conflict='creative_id'
            ).execute()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to upsert creative {creative_id}: {e}")
            return False
    
    def track_creative_performance(self, ad_id: str, creative_ids: Dict[str, str], 
                                 performance_data: Dict[str, Any], stage: str) -> bool:
        """Track performance of specific creatives used in an ad."""
        try:
            if not self.supabase_client:
                return False
            
            # Track performance for each creative type
            for creative_type, creative_id in creative_ids.items():
                if not creative_id:
                    continue
                
                performance_record = {
                    "creative_id": creative_id,
                    "ad_id": ad_id,
                    "stage": stage,
                    "date_start": performance_data.get("date_start", datetime.now().strftime("%Y-%m-%d")),
                    "date_end": performance_data.get("date_end", datetime.now().strftime("%Y-%m-%d")),
                    "impressions": int(performance_data.get("impressions", 0)),
                    "clicks": int(performance_data.get("clicks", 0)),
                    "spend": float(performance_data.get("spend", 0)),
                    "purchases": int(performance_data.get("purchases", 0)),
                    "add_to_cart": int(performance_data.get("add_to_cart", 0)),
                    "initiate_checkout": int(performance_data.get("initiate_checkout", 0)),
                    "ctr": float(performance_data.get("ctr", 0)),
                    "cpc": float(performance_data.get("cpc", 0)),
                    "cpm": float(performance_data.get("cpm", 0)),
                    "roas": float(performance_data.get("roas", 0)),
                    "cpa": float(performance_data.get("cpa", 0)),
                    "engagement_rate": float(performance_data.get("engagement_rate", 0)),
                    "conversion_rate": float(performance_data.get("conversion_rate", 0))
                }
                
                self.supabase_client.table('creative_performance').upsert(
                    performance_record,
                    on_conflict='creative_id,ad_id,date_start'
                ).execute()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to track creative performance: {e}")
            return False
    
    def analyze_creative_performance(self, days_back: int = 30) -> List[CreativeInsight]:
        """Analyze creative performance patterns and generate insights."""
        try:
            if not self.supabase_client:
                return []
            
            # Get performance data
            cutoff_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            
            performance_data = self.supabase_client.table('creative_performance').select(
                'creative_id,impressions,clicks,spend,purchases,add_to_cart,initiate_checkout,ctr,cpc,cpm,roas,cpa'
            ).gte('date_start', cutoff_date).execute()
            
            if not performance_data.data:
                return []
            
            insights = []
            
            # Analyze top performers
            df = pd.DataFrame(performance_data.data)
            if not df.empty:
                # Group by creative_id and aggregate
                creative_stats = df.groupby('creative_id').agg({
                    'impressions': 'sum',
                    'clicks': 'sum',
                    'spend': 'sum',
                    'purchases': 'sum',
                    'add_to_cart': 'sum',
                    'initiate_checkout': 'sum',
                    'ctr': 'mean',
                    'cpc': 'mean',
                    'cpm': 'mean',
                    'roas': 'mean',
                    'cpa': 'mean'
                }).reset_index()
                
                # Calculate performance scores
                creative_stats['performance_score'] = (
                    creative_stats['roas'] * 0.4 +
                    (creative_stats['ctr'] / 100) * 0.3 +
                    (creative_stats['purchases'] / creative_stats['clicks'].replace(0, 1)) * 0.3
                )
                
                # Top performers insight
                top_performers = creative_stats.nlargest(5, 'performance_score')
                if not top_performers.empty:
                    insights.append(CreativeInsight(
                        insight_type="top_performers",
                        description=f"Top performing creatives with avg ROAS {top_performers['roas'].mean():.2f}",
                        confidence=0.8,
                        recommendations=[
                            f"Scale up creative {row['creative_id']} (ROAS: {row['roas']:.2f})"
                            for _, row in top_performers.iterrows()
                        ],
                        metadata={"top_creatives": top_performers['creative_id'].tolist()}
                    ))
                
                # CTR analysis
                high_ctr = creative_stats[creative_stats['ctr'] > creative_stats['ctr'].quantile(0.8)]
                if not high_ctr.empty:
                    insights.append(CreativeInsight(
                        insight_type="high_ctr",
                        description=f"High CTR creatives detected (avg CTR: {high_ctr['ctr'].mean():.2f}%)",
                        confidence=0.7,
                        recommendations=[
                            f"Analyze creative {row['creative_id']} for engagement patterns"
                            for _, row in high_ctr.iterrows()
                        ],
                        metadata={"high_ctr_creatives": high_ctr['creative_id'].tolist()}
                    ))
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to analyze creative performance: {e}")
            return []
    
    def generate_ai_creatives(self, source_creative_id: str, creative_type: str, 
                            count: int = 3) -> List[str]:
        """Generate AI-powered creatives based on top performers."""
        try:
            if not OPENAI_AVAILABLE or not self.openai_api_key:
                self.logger.warning("OpenAI not available for creative generation")
                return []
            
            # Get source creative content
            if self.supabase_client:
                source_creative = self.supabase_client.table('creative_library').select(
                    'content,creative_type,category'
                ).eq('creative_id', source_creative_id).execute()
                
                if not source_creative.data:
                    return []
                
                source_content = source_creative.data[0]['content']
                source_category = source_creative.data[0].get('category', 'general')
            else:
                return []
            
            # Generate prompt for OpenAI
            prompt = f"""
            Based on this high-performing {creative_type} for {source_category}:
            "{source_content}"
            
            Generate {count} similar but distinct {creative_type}s that:
            1. Maintain the same tone and style
            2. Target the same audience
            3. Include similar key benefits
            4. Are optimized for social media advertising
            5. Have strong call-to-action elements
            
            Return only the {creative_type}s, one per line, without numbering or labels.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert copywriter specializing in high-converting social media advertisements."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            
            generated_creatives = response.choices[0].message.content.strip().split('\n')
            generated_creatives = [c.strip() for c in generated_creatives if c.strip()]
            
            # Store generated creatives in Supabase
            for i, creative_content in enumerate(generated_creatives):
                ai_creative_id = f"ai_{creative_type}_{source_creative_id}_{i+1}_{int(time.time())}"
                self._upsert_creative(
                    creative_id=ai_creative_id,
                    creative_type=creative_type,
                    content=creative_content,
                    category=source_category,
                    created_by="ai_generated"
                )
                
                # Store in AI generated creatives table
                ai_creative_data = {
                    "creative_id": ai_creative_id,
                    "source_creative_id": source_creative_id,
                    "generation_prompt": prompt,
                    "generation_model": "gpt-4",
                    "content": creative_content,
                    "creative_type": creative_type,
                    "category": source_category,
                    "generation_parameters": {"temperature": 0.7, "max_tokens": 500}
                }
                
                self.supabase_client.table('ai_generated_creatives').insert(ai_creative_data).execute()
            
            return generated_creatives
            
        except Exception as e:
            self.logger.error(f"Failed to generate AI creatives: {e}")
            return []
    
    def find_similar_creatives(self, creative_id: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find similar creatives using semantic similarity."""
        try:
            if not self.sentence_model or not self.supabase_client:
                return []
            
            # Get source creative
            source_creative = self.supabase_client.table('creative_library').select(
                'content,creative_type'
            ).eq('creative_id', creative_id).execute()
            
            if not source_creative.data:
                return []
            
            source_content = source_creative.data[0]['content']
            source_type = source_creative.data[0]['creative_type']
            
            # Get all creatives of the same type
            all_creatives = self.supabase_client.table('creative_library').select(
                'creative_id,content'
            ).eq('creative_type', source_type).neq('creative_id', creative_id).execute()
            
            if not all_creatives.data:
                return []
            
            # Calculate similarities
            source_embedding = self.sentence_model.encode([source_content])
            similar_creatives = []
            
            for creative in all_creatives.data:
                content_embedding = self.sentence_model.encode([creative['content']])
                similarity = np.dot(source_embedding[0], content_embedding[0])
                
                if similarity >= threshold:
                    similar_creatives.append((creative['creative_id'], float(similarity)))
            
            # Sort by similarity
            similar_creatives.sort(key=lambda x: x[1], reverse=True)
            
            return similar_creatives
            
        except Exception as e:
            self.logger.error(f"Failed to find similar creatives: {e}")
            return []
    
    def get_top_creatives(self, creative_type: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing creatives by type."""
        try:
            if not self.supabase_client:
                return []
            
            # Get creatives with performance data
            query = f"""
            SELECT 
                cl.creative_id,
                cl.content,
                cl.performance_score,
                AVG(cp.roas) as avg_roas,
                AVG(cp.ctr) as avg_ctr,
                SUM(cp.impressions) as total_impressions,
                SUM(cp.purchases) as total_purchases
            FROM creative_library cl
            LEFT JOIN creative_performance cp ON cl.creative_id = cp.creative_id
            WHERE cl.creative_type = '{creative_type}'
            GROUP BY cl.creative_id, cl.content, cl.performance_score
            ORDER BY avg_roas DESC, cl.performance_score DESC
            LIMIT {limit}
            """
            
            result = self.supabase_client.rpc('execute_sql', {'query': query}).execute()
            
            if result.data:
                return result.data
            else:
                # Fallback to simple query
                result = self.supabase_client.table('creative_library').select(
                    'creative_id,content,performance_score'
                ).eq('creative_type', creative_type).order('performance_score', desc=True).limit(limit).execute()
                
                return result.data
            
        except Exception as e:
            self.logger.error(f"Failed to get top creatives: {e}")
            return []
    
    def update_creative_performance_scores(self) -> bool:
        """Update performance scores for all creatives based on recent data."""
        try:
            if not self.supabase_client:
                return False
            
            # Get performance data for last 30 days
            cutoff_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            
            performance_data = self.supabase_client.table('creative_performance').select(
                'creative_id,roas,ctr,purchases,clicks'
            ).gte('date_start', cutoff_date).execute()
            
            if not performance_data.data:
                return False
            
            # Calculate performance scores
            df = pd.DataFrame(performance_data.data)
            creative_scores = df.groupby('creative_id').agg({
                'roas': 'mean',
                'ctr': 'mean',
                'purchases': 'sum',
                'clicks': 'sum'
            }).reset_index()
            
            creative_scores['performance_score'] = (
                creative_scores['roas'] * 0.4 +
                (creative_scores['ctr'] / 100) * 0.3 +
                (creative_scores['purchases'] / creative_scores['clicks'].replace(0, 1)) * 0.3
            )
            
            # Update scores in database
            for _, row in creative_scores.iterrows():
                self.supabase_client.table('creative_library').update({
                    'performance_score': float(row['performance_score']),
                    'updated_at': datetime.now().isoformat()
                }).eq('creative_id', row['creative_id']).execute()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update creative performance scores: {e}")
            return False


def create_creative_intelligence_system(supabase_client=None, openai_api_key=None) -> CreativeIntelligenceSystem:
    """Create and initialize the creative intelligence system."""
    return CreativeIntelligenceSystem(supabase_client, openai_api_key)
