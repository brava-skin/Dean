"""
DEAN ML EXTRAS
Additional advanced ML capabilities:
- Automated Feature Engineering (FeatureTools)
- Bayesian Optimization
- Competitor Analysis (Meta Ad Library)
- Portfolio Optimization
- Seasonality Detection & Timing
- Budget Allocation Optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.preprocessing import StandardScaler

# Optional imports
try:
    import featuretools as ft
    FEATURETOOLS_AVAILABLE = True
except ImportError:
    FEATURETOOLS_AVAILABLE = False
    ft = None

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    BAYESOPT_AVAILABLE = True
except ImportError:
    BAYESOPT_AVAILABLE = False

logger = logging.getLogger(__name__)

# =====================================================
# AUTOMATED FEATURE ENGINEERING
# =====================================================

class AutoFeatureEngineer:
    """Automated feature discovery using FeatureTools."""
    
    def __init__(self):
        if not FEATURETOOLS_AVAILABLE:
            self.available = False
            self.logger = logging.getLogger(f"{__name__}.AutoFeatureEngineer")
            self.logger.warning("FeatureTools not available - auto feature engineering disabled")
            return
        
        self.available = True
        self.logger = logging.getLogger(f"{__name__}.AutoFeatureEngineer")
    
    def generate_features(self, df: pd.DataFrame, entity_id: str = 'ad_id',
                         target_col: Optional[str] = None,
                         max_depth: int = 2) -> pd.DataFrame:
        """
        Automatically generate features using deep feature synthesis.
        
        Args:
            df: DataFrame with raw data
            entity_id: Column to use as entity identifier
            target_col: Target column to exclude from feature generation
            max_depth: Maximum depth for feature synthesis
        
        Returns:
            DataFrame with original + engineered features
        """
        if not self.available:
            return df
        
        try:
            # Create entity set
            es = ft.EntitySet(id='ads')
            
            # Add dataframe
            es = es.add_dataframe(
                dataframe_name='performance',
                dataframe=df,
                index=entity_id,
                time_index='created_at' if 'created_at' in df.columns else None
            )
            
            # Generate features (exclude target)
            ignore_columns = [target_col] if target_col else []
            
            feature_matrix, feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name='performance',
                max_depth=max_depth,
                ignore_columns=ignore_columns,
                verbose=False
            )
            
            self.logger.info(f"Generated {len(feature_defs)} automated features")
            
            return feature_matrix
            
        except Exception as e:
            self.logger.error(f"Error generating automated features: {e}")
            return df

# =====================================================
# BAYESIAN OPTIMIZATION
# =====================================================

class BayesianOptimizer:
    """Bayesian optimization for hyperparameter tuning."""
    
    def __init__(self):
        if not BAYESOPT_AVAILABLE:
            self.available = False
            self.logger = logging.getLogger(f"{__name__}.BayesianOptimizer")
            self.logger.warning("scikit-optimize not available - Bayesian optimization disabled")
            return
        
        self.available = True
        self.logger = logging.getLogger(f"{__name__}.BayesianOptimizer")
    
    def optimize_thresholds(self, objective_func, threshold_ranges: Dict[str, Tuple[float, float]],
                           n_calls: int = 50) -> Dict[str, float]:
        """
        Optimize kill/promote thresholds using Bayesian optimization.
        
        Args:
            objective_func: Function to minimize (e.g., negative ROAS)
            threshold_ranges: Dict of threshold_name -> (min, max)
            n_calls: Number of optimization iterations
        
        Returns:
            Optimal thresholds
        """
        if not self.available:
            # Return midpoint of ranges
            return {name: (r[0] + r[1]) / 2 for name, r in threshold_ranges.items()}
        
        try:
            # Define search space
            space = [Real(r[0], r[1], name=name) for name, r in threshold_ranges.items()]
            
            # Run Bayesian optimization
            result = gp_minimize(
                objective_func,
                space,
                n_calls=n_calls,
                random_state=42,
                verbose=False
            )
            
            # Extract optimal parameters
            optimal_thresholds = dict(zip(threshold_ranges.keys(), result.x))
            
            self.logger.info(f"Optimal thresholds found: {optimal_thresholds}")
            self.logger.info(f"Best objective value: {result.fun:.4f}")
            
            return optimal_thresholds
            
        except Exception as e:
            self.logger.error(f"Error in Bayesian optimization: {e}")
            return {name: (r[0] + r[1]) / 2 for name, r in threshold_ranges.items()}

# =====================================================
# COMPETITOR ANALYSIS
# =====================================================

class CompetitorAnalyzer:
    """Analyze competitor ads from Meta Ad Library."""
    
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.logger = logging.getLogger(f"{__name__}.CompetitorAnalyzer")
    
    def search_competitor_ads(self, search_terms: List[str], 
                             country: str = 'ALL',
                             limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search Meta Ad Library for competitor ads.
        
        Args:
            search_terms: Keywords to search for
            country: Country code
            limit: Max ads to retrieve
        
        Returns:
            List of competitor ad data
        """
        try:
            import requests
            
            competitor_ads = []
            
            for term in search_terms:
                # Use correct API endpoint for ALL ad types (not just political)
                url = f"https://graph.facebook.com/v18.0/ads_archive"
                params = {
                    'access_token': self.access_token,
                    'search_terms': term,
                    'ad_reached_countries': country,
                    'ad_type': 'ALL',  # FIX: Changed from POLITICAL_AND_ISSUE_ADS to ALL
                    'fields': 'id,ad_creative_bodies,ad_creative_link_titles,ad_delivery_start_time,impressions,spend',
                    'limit': limit,
                    'search_page_ids': ''  # Empty to search all pages
                }
                
                response = requests.get(url, params=params, timeout=30)
                
                if response.status_code == 200:
                    data = response.json()
                    ads = data.get('data', [])
                    competitor_ads.extend(ads)
                    
                    self.logger.info(f"Found {len(ads)} competitor ads for '{term}'")
                else:
                    self.logger.warning(f"Failed to fetch competitor ads: {response.status_code}")
            
            return competitor_ads
            
        except Exception as e:
            self.logger.error(f"Error searching competitor ads: {e}")
            return []
    
    def analyze_competitor_trends(self, competitor_ads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in competitor advertising."""
        try:
            if not competitor_ads:
                return {}
            
            df = pd.DataFrame(competitor_ads)
            
            analysis = {
                'total_competitors': len(df),
                'avg_impressions': df.get('impressions', pd.Series()).mean() if 'impressions' in df else 0,
                'common_keywords': self._extract_common_keywords(df),
                'market_saturation': self._calculate_saturation(df)
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing competitor trends: {e}")
            return {}
    
    def _extract_common_keywords(self, df: pd.DataFrame) -> List[str]:
        """Extract most common keywords from competitor ads."""
        try:
            if 'ad_creative_bodies' not in df:
                return []
            
            # Simple keyword extraction
            all_text = ' '.join(df['ad_creative_bodies'].fillna(''))
            words = all_text.lower().split()
            
            # Count frequency
            from collections import Counter
            word_counts = Counter(words)
            
            # Get top 10
            common = [word for word, count in word_counts.most_common(10) if len(word) > 3]
            
            return common
            
        except Exception as e:
            self.logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _calculate_saturation(self, df: pd.DataFrame) -> float:
        """Calculate market saturation score."""
        try:
            # Simple saturation metric: number of active competitors
            # 0.0 = low saturation, 1.0 = high saturation
            
            num_competitors = len(df)
            
            # Normalize to 0-1 (assuming 100+ competitors = saturated)
            saturation = min(1.0, num_competitors / 100.0)
            
            return saturation
            
        except Exception as e:
            self.logger.error(f"Error calculating saturation: {e}")
            return 0.5

# =====================================================
# PORTFOLIO OPTIMIZATION
# =====================================================

class PortfolioOptimizer:
    """Optimize budget allocation across ads using linear programming."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PortfolioOptimizer")
    
    def optimize_budget_allocation(self, ad_data: List[Dict[str, Any]],
                                  total_budget: float,
                                  constraints: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Optimize budget allocation to maximize ROAS.
        
        Args:
            ad_data: List of ads with predicted_roas, current_budget, min_budget, max_budget
            total_budget: Total budget to allocate
            constraints: Additional constraints
        
        Returns:
            Dict of ad_id -> optimal_budget
        """
        try:
            n_ads = len(ad_data)
            
            if n_ads == 0:
                return {}
            
            # Objective: maximize total ROAS
            # Decision variables: budget for each ad
            
            # Extract data
            ad_ids = [ad['ad_id'] for ad in ad_data]
            predicted_roas = np.array([ad.get('predicted_roas', 1.0) for ad in ad_data])
            min_budgets = np.array([ad.get('min_budget', 5.0) for ad in ad_data])
            max_budgets = np.array([ad.get('max_budget', 100.0) for ad in ad_data])
            
            # VALIDATION: Check if constraints are feasible
            total_min_budget = np.sum(min_budgets)
            total_max_budget = np.sum(max_budgets)
            
            if total_budget < total_min_budget:
                self.logger.warning(f"Total budget ({total_budget}) < sum of min budgets ({total_min_budget}). Using proportional allocation.")
                return self._proportional_allocation(ad_data, total_budget)
            
            if total_budget > total_max_budget:
                self.logger.warning(f"Total budget ({total_budget}) > sum of max budgets ({total_max_budget}). Capping at max.")
                total_budget = total_max_budget
            
            # Objective function (negative because we minimize)
            def objective(budgets):
                return -np.sum(budgets * predicted_roas)
            
            # Constraints
            cons = []
            
            # Budget constraint: sum of budgets = total_budget
            cons.append({
                'type': 'eq',
                'fun': lambda x: np.sum(x) - total_budget
            })
            
            # Bounds for each ad
            bounds = list(zip(min_budgets, max_budgets))
            
            # Initial guess: equal distribution
            x0 = np.full(n_ads, total_budget / n_ads)
            x0 = np.clip(x0, min_budgets, max_budgets)
            
            # Optimize
            result = optimize.minimize(
                objective,
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=cons
            )
            
            if result.success:
                optimal_budgets = dict(zip(ad_ids, result.x))
                self.logger.info(f"Portfolio optimization successful. Expected ROAS: {-result.fun:.2f}")
                return optimal_budgets
            else:
                self.logger.warning("Portfolio optimization failed, using proportional allocation")
                return self._proportional_allocation(ad_data, total_budget)
            
        except Exception as e:
            self.logger.error(f"Error optimizing portfolio: {e}")
            return self._proportional_allocation(ad_data, total_budget)
    
    def _proportional_allocation(self, ad_data: List[Dict[str, Any]], 
                                total_budget: float) -> Dict[str, float]:
        """Fallback: allocate proportional to ROAS."""
        try:
            ad_ids = [ad['ad_id'] for ad in ad_data]
            roas_values = np.array([ad.get('predicted_roas', 1.0) for ad in ad_data])
            
            # Allocate proportional to ROAS
            total_roas = np.sum(roas_values)
            if total_roas > 0:
                budgets = (roas_values / total_roas) * total_budget
            else:
                budgets = np.full(len(ad_data), total_budget / len(ad_data))
            
            return dict(zip(ad_ids, budgets))
            
        except Exception as e:
            self.logger.error(f"Error in proportional allocation: {e}")
            return {}

# =====================================================
# SEASONALITY & TIMING OPTIMIZATION
# =====================================================

class SeasonalityAnalyzer:
    """Detect and leverage seasonality patterns."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SeasonalityAnalyzer")
    
    def detect_day_of_week_patterns(self, df: pd.DataFrame, 
                                   metric_col: str = 'roas') -> Dict[str, float]:
        """Detect which days perform best."""
        try:
            if 'date_start' not in df:
                return {}
            
            df = df.copy()
            df['day_of_week'] = pd.to_datetime(df['date_start']).dt.day_name()
            
            # Average metric by day
            day_performance = df.groupby('day_of_week')[metric_col].mean().to_dict()
            
            self.logger.info(f"Day of week patterns: {day_performance}")
            
            return day_performance
            
        except Exception as e:
            self.logger.error(f"Error detecting day patterns: {e}")
            return {}
    
    def detect_hour_of_day_patterns(self, df: pd.DataFrame,
                                   metric_col: str = 'ctr') -> Dict[int, float]:
        """Detect which hours perform best."""
        try:
            if 'created_at' not in df:
                return {}
            
            df = df.copy()
            df['hour'] = pd.to_datetime(df['created_at']).dt.hour
            
            # Average metric by hour
            hour_performance = df.groupby('hour')[metric_col].mean().to_dict()
            
            self.logger.info(f"Hour of day patterns detected")
            
            return hour_performance
            
        except Exception as e:
            self.logger.error(f"Error detecting hour patterns: {e}")
            return {}
    
    def get_optimal_launch_time(self, hour_patterns: Dict[int, float]) -> int:
        """Get best hour to launch new ads."""
        try:
            if not hour_patterns:
                return 9  # Default: 9 AM
            
            # Return hour with highest performance
            best_hour = max(hour_patterns, key=hour_patterns.get)
            
            self.logger.info(f"Optimal launch time: {best_hour}:00")
            
            return best_hour
            
        except Exception as e:
            self.logger.error(f"Error getting optimal time: {e}")
            return 9
    
    def should_increase_budget_now(self, day_patterns: Dict[str, float],
                                  hour_patterns: Dict[int, float],
                                  current_time: datetime) -> Tuple[bool, float]:
        """Determine if now is a good time to increase budgets."""
        try:
            day_name = current_time.strftime('%A')
            hour = current_time.hour
            
            # Get performance for current time
            day_perf = day_patterns.get(day_name, 1.0)
            hour_perf = hour_patterns.get(hour, 1.0)
            
            # Average performance across all times
            avg_day = np.mean(list(day_patterns.values())) if day_patterns else 1.0
            avg_hour = np.mean(list(hour_patterns.values())) if hour_patterns else 1.0
            
            # Calculate relative performance
            relative_perf = (day_perf / avg_day) * (hour_perf / avg_hour)
            
            # If current time is 20%+ better than average, recommend increase
            should_increase = relative_perf > 1.2
            multiplier = min(2.0, relative_perf)  # Cap at 2x
            
            return should_increase, multiplier
            
        except Exception as e:
            self.logger.error(f"Error checking budget timing: {e}")
            return False, 1.0

# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_auto_feature_engineer() -> AutoFeatureEngineer:
    """Create automated feature engineer."""
    return AutoFeatureEngineer()

def create_bayesian_optimizer() -> BayesianOptimizer:
    """Create Bayesian optimizer."""
    return BayesianOptimizer()

def create_competitor_analyzer(access_token: str) -> CompetitorAnalyzer:
    """Create competitor analyzer."""
    return CompetitorAnalyzer(access_token)

def create_portfolio_optimizer() -> PortfolioOptimizer:
    """Create portfolio optimizer."""
    return PortfolioOptimizer()

def create_seasonality_analyzer() -> SeasonalityAnalyzer:
    """Create seasonality analyzer."""
    return SeasonalityAnalyzer()

