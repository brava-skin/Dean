"""
DEAN ML SYSTEM ENHANCEMENTS
Critical improvements to ML intelligence system

This module provides enhanced ML capabilities including:
- Model validation and A/B testing
- Data progress tracking  
- Anomaly detection
- Hyperparameter tuning
- Time-series forecasting
- Creative similarity analysis
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from supabase import Client

from infrastructure.utils import now_utc

logger = logging.getLogger(__name__)

# =====================================================
# MODEL VALIDATION & A/B TESTING
# =====================================================

@dataclass
class ModelValidationResult:
    """Results from model validation."""
    model_type: str
    stage: str
    accuracy: float
    mae: float
    r2_score: float
    prediction_vs_actual: List[Tuple[float, float]]
    validation_date: datetime
    is_performing_well: bool

class ModelValidator:
    """Validates ML model predictions against actual outcomes."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.logger = logging.getLogger(f"{__name__}.ModelValidator")
    
    def validate_predictions(self, model_type: str, stage: str, days_back: int = 7) -> ModelValidationResult:
        """Compare predictions vs actual outcomes."""
        try:
            # Get predictions from last N days
            start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            # FIX: ml_predictions doesn't have model_type column, only stage and prediction_type
            # We need to filter by stage and check the linked model's type
            predictions_response = self.client.table('ml_predictions').select('*').eq(
                'stage', stage
            ).gte('created_at', start_date).execute()
            
            if not predictions_response.data or len(predictions_response.data) < 5:
                return None
            
            # Get actual performance data
            pred_vs_actual = []
            total_error = 0
            
            for pred in predictions_response.data:
                ad_id = pred['ad_id']
                predicted_value = pred['predicted_value']
                prediction_time = datetime.fromisoformat(pred['created_at'].replace('Z', '+00:00'))
                
                # Get actual performance 24h after prediction
                actual_date = (prediction_time + timedelta(hours=24)).strftime('%Y-%m-%d')
                
                actual_response = self.client.table('performance_metrics').select('*').eq(
                    'ad_id', ad_id
                ).eq('date_start', actual_date).execute()
                
                if actual_response.data:
                    actual = actual_response.data[0]
                    actual_value = actual.get('cpa' if 'predictor' in model_type else 'roas', 0)
                    
                    pred_vs_actual.append((predicted_value, actual_value))
                    total_error += abs(predicted_value - actual_value)
            
            if not pred_vs_actual:
                return None
            
            # Calculate metrics
            predictions = np.array([p[0] for p in pred_vs_actual])
            actuals = np.array([p[1] for p in pred_vs_actual])
            
            mae = np.mean(np.abs(predictions - actuals))
            mse = np.mean((predictions - actuals) ** 2)
            r2 = 1 - (mse / (np.var(actuals) + 1e-6))
            
            # Accuracy: predictions within 20% of actual
            within_20pct = sum(1 for p, a in pred_vs_actual if abs(p - a) / (abs(a) + 1e-6) < 0.2)
            accuracy = within_20pct / len(pred_vs_actual)
            
            is_performing_well = accuracy > 0.6 and r2 > 0.3
            
            result = ModelValidationResult(
                model_type=model_type,
                stage=stage,
                accuracy=accuracy,
                mae=mae,
                r2_score=r2,
                prediction_vs_actual=pred_vs_actual,
                validation_date=now_utc(),
                is_performing_well=is_performing_well
            )
            
            # Store validation result
            self._store_validation_result(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error validating model: {e}")
            return None
    
    def _store_validation_result(self, result: ModelValidationResult) -> None:
        """Store validation result in database."""
        try:
            data = {
                'model_type': result.model_type,
                'stage': result.stage,
                'accuracy': min(max(float(result.accuracy or 0), -999.99), 999.99),
                'mae': min(max(float(result.mae or 0), -999999.99), 999999.99),
                'r2_score': min(max(float(result.r2_score or 0), -999.99), 999.99),
                'is_performing_well': result.is_performing_well,
                'validation_date': result.validation_date.isoformat(),
                'sample_size': len(result.prediction_vs_actual)
            }
            
            self.client.table('model_validations').insert(data).execute()
            
        except Exception as e:
            self.logger.error(f"Error storing validation result: {e}")
    
    def validate_all_models(self, days_back: int = 7) -> Dict[str, Dict[str, Any]]:
        """
        Validate all active models across all stages.
        
        Args:
            days_back: Number of days to look back for validation data
        
        Returns:
            Dict of model_name -> validation metrics
        """
        results = {}
        
        try:
            # Model types to validate
            model_configs = [
                ('performance_predictor', 'testing'),
                ('performance_predictor', 'validation'),
                ('performance_predictor', 'scaling'),
                ('roas_predictor', 'testing'),
                ('roas_predictor', 'validation'),
                ('roas_predictor', 'scaling'),
            ]
            
            for model_type, stage in model_configs:
                model_name = f"{model_type}_{stage}"
                
                try:
                    validation_result = self.validate_predictions(model_type, stage, days_back)
                    
                    if validation_result:
                        results[model_name] = {
                            'accuracy': validation_result.accuracy,
                            'mae': validation_result.mae,
                            'r2_score': validation_result.r2_score,
                            'is_performing_well': validation_result.is_performing_well,
                            'sample_size': len(validation_result.prediction_vs_actual)
                        }
                        
                        self.logger.info(
                            f"Validated {model_name}: "
                            f"Accuracy={validation_result.accuracy:.2%}, "
                            f"R²={validation_result.r2_score:.3f}"
                        )
                    else:
                        results[model_name] = {
                            'accuracy': 0.0,
                            'mae': 0.0,
                            'r2_score': 0.0,
                            'is_performing_well': False,
                            'sample_size': 0,
                            'status': 'insufficient_data'
                        }
                        
                except Exception as e:
                    self.logger.warning(f"Could not validate {model_name}: {e}")
                    results[model_name] = {
                        'accuracy': 0.0,
                        'error': str(e),
                        'status': 'error'
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in validate_all_models: {e}")
            return {}

# =====================================================
# DATA PROGRESS TRACKING
# =====================================================

class DataProgressTracker:
    """Track data accumulation progress for ML readiness."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.logger = logging.getLogger(f"{__name__}.DataProgressTracker")
    
    def get_ml_readiness(self, stage: str) -> Dict[str, Any]:
        """Check ML system readiness for a stage."""
        try:
            # Count data points
            response = self.client.table('performance_metrics').select('ad_id, date_start').eq(
                'stage', stage
            ).execute()
            
            if not response.data:
                days_available = 0
                unique_ads = 0
            else:
                df = pd.DataFrame(response.data)
                df['date_start'] = pd.to_datetime(df['date_start'])
                
                # Calculate days based on recent active campaign data
                # Filter to recent data (last 30 days) to avoid old test data inflating the count
                today = pd.Timestamp.now().normalize()
                recent_cutoff = today - pd.Timedelta(days=30)
                recent_data = df[df['date_start'] >= recent_cutoff]
                
                if len(recent_data) > 0:
                    # Use recent data span
                    days_available = (recent_data['date_start'].max() - recent_data['date_start'].min()).days + 1
                else:
                    # Fallback: days since most recent data point
                    days_available = max(1, (today - df['date_start'].max()).days + 1)
                
                # Ensure reasonable bounds (1-365 days)
                days_available = max(1, min(days_available, 365))
                
                unique_ads = df['ad_id'].nunique()
            
            # ML readiness levels
            min_days = 5
            recommended_days = 10
            optimal_days = 30
            
            if days_available < min_days:
                status = 'collecting'
                message = f"Day {days_available}/{min_days} - ML predictions will be available soon"
                ready = False
            elif days_available < recommended_days:
                status = 'ready'
                message = f"Day {days_available}/{recommended_days} - ML active, improving accuracy"
                ready = True
            elif days_available < optimal_days:
                status = 'improving'
                message = f"Day {days_available}/{optimal_days} - ML accuracy improving"
                ready = True
            else:
                status = 'optimal'
                message = f"Day {days_available}+ - Full ML intelligence operational"
                ready = True
            
            return {
                'stage': stage,
                'status': status,
                'message': message,
                'ready': ready,
                'days_available': days_available,
                'unique_ads': unique_ads,
                'samples': len(response.data) if response.data else 0,  # FIX: Add 'samples' key for ml_pipeline
                'data_points': len(response.data) if response.data else 0,
                'progress_pct': min(100, int((days_available / optimal_days) * 100))
            }
            
        except Exception as e:
            self.logger.error(f"Error checking ML readiness: {e}")
            return {'ready': False, 'status': 'error', 'message': str(e)}

# =====================================================
# ANOMALY DETECTION
# =====================================================

class AnomalyDetector:
    """Detect anomalies in ad performance to distinguish tracking issues from poor performance."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.logger = logging.getLogger(f"{__name__}.AnomalyDetector")
    
    def detect_anomalies(self, ad_id: str, days_back: int = 14) -> Dict[str, Any]:
        """Detect performance anomalies."""
        try:
            # Get performance history
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            response = self.client.table('performance_metrics').select('*').eq(
                'ad_id', ad_id
            ).gte('date_start', start_date).order('date_start').execute()
            
            if not response.data or len(response.data) < 7:
                return {'anomalies_detected': False, 'reason': 'insufficient_data'}
            
            df = pd.DataFrame(response.data)
            
            anomalies = {}
            metrics_to_check = ['ctr', 'cpa', 'roas', 'impressions', 'clicks']
            
            for metric in metrics_to_check:
                if metric not in df.columns:
                    continue
                
                values = pd.to_numeric(df[metric], errors='coerce').fillna(0).values
                
                if len(values) < 7:
                    continue
                
                # Use IQR method for anomaly detection
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                iqr = q3 - q1
                lower_bound = q1 - 2.5 * iqr
                upper_bound = q3 + 2.5 * iqr
                
                # Find anomalies
                anomaly_indices = np.where((values < lower_bound) | (values > upper_bound))[0]
                
                if len(anomaly_indices) > 0:
                    # Check if it's tracking issue or performance issue
                    latest_values = values[-3:]  # Last 3 days
                    latest_anomalies = sum(1 for v in latest_values if v < lower_bound or v > upper_bound)
                    
                    if latest_anomalies >= 2:
                        # Multiple recent anomalies = likely tracking issue
                        anomaly_type = 'tracking_issue'
                    else:
                        # Isolated anomaly = performance variation
                        anomaly_type = 'performance_variation'
                    
                    anomalies[metric] = {
                        'detected': True,
                        'type': anomaly_type,
                        'count': len(anomaly_indices),
                        'latest_value': float(values[-1]),
                        'expected_range': (float(lower_bound), float(upper_bound))
                    }
            
            has_tracking_issues = any(
                a.get('type') == 'tracking_issue' 
                for a in anomalies.values()
            )
            
            return {
                'anomalies_detected': len(anomalies) > 0,
                'anomalies': anomalies,
                'has_tracking_issues': has_tracking_issues,
                'recommendation': 'Check tracking setup' if has_tracking_issues else 'Normal variation'
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting anomalies: {e}")
            return {'anomalies_detected': False, 'error': str(e)}

# =====================================================
# HYPERPARAMETER TUNING (OPTUNA)
# =====================================================

class HyperparameterTuner:
    """Optimize model hyperparameters using Optuna."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.HyperparameterTuner")
        try:
            import optuna
            self.optuna = optuna
            self.available = True
        except ImportError:
            self.logger.warning("Optuna not available - using default hyperparameters")
            self.optuna = None
            self.available = False
    
    def tune_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                     n_trials: int = 50) -> Dict[str, Any]:
        """Tune XGBoost hyperparameters."""
        if not self.available:
            return self._default_xgb_params()
        
        try:
            def objective(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42,
                    'n_jobs': -1
                }
                
                try:
                    import xgboost as xgb
                    model = xgb.XGBRegressor(**params)
                except ImportError:
                    from sklearn.ensemble import GradientBoostingRegressor
                    model = GradientBoostingRegressor(
                        n_estimators=params['n_estimators'],
                        max_depth=params['max_depth'],
                        learning_rate=params['learning_rate'],
                        random_state=42
                    )
                
                from sklearn.model_selection import cross_val_score
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_absolute_error')
                return -scores.mean()
            
            study = self.optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            self.logger.info(f"Best params found: MAE={study.best_value:.4f}")
            return study.best_params
            
        except Exception as e:
            self.logger.error(f"Error tuning hyperparameters: {e}")
            return self._default_xgb_params()
    
    def _default_xgb_params(self) -> Dict[str, Any]:
        """Default XGBoost parameters."""
        return {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1
        }

# =====================================================
# TIME-SERIES FORECASTING (PROPHET)
# =====================================================

class TimeSeriesForecaster:
    """Time-series forecasting using Prophet."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.logger = logging.getLogger(f"{__name__}.TimeSeriesForecaster")
        try:
            from prophet import Prophet
            self.prophet = Prophet
            self.available = True
        except ImportError:
            self.logger.warning("Prophet not available - time-series forecasting disabled")
            self.prophet = None
            self.available = False
    
    def forecast_metric(self, ad_id: str, metric_name: str, 
                       days_ahead: int = 7, days_back: int = 30) -> Dict[str, Any]:
        """Forecast a metric using Prophet."""
        if not self.available:
            return {}
        
        try:
            # Get historical data
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            response = self.client.table('performance_metrics').select('date_start, ' + metric_name).eq(
                'ad_id', ad_id
            ).gte('date_start', start_date).order('date_start').execute()
            
            if not response.data or len(response.data) < 7:
                return {}
            
            # Prepare data for Prophet
            df = pd.DataFrame(response.data)
            df['ds'] = pd.to_datetime(df['date_start'])
            df['y'] = pd.to_numeric(df[metric_name], errors='coerce').fillna(0)
            df = df[['ds', 'y']]
            
            # Train Prophet model
            model = self.prophet(daily_seasonality=False, weekly_seasonality=True)
            model.fit(df)
            
            # Make forecast
            future = model.make_future_dataframe(periods=days_ahead)
            forecast = model.predict(future)
            
            # Extract forecast values
            forecast_data = forecast.tail(days_ahead)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
            
            # Detect trend
            recent_trend = forecast['trend'].tail(7).mean() - forecast['trend'].head(7).mean()
            trend_direction = 'increasing' if recent_trend > 0 else 'decreasing' if recent_trend < 0 else 'stable'
            
            return {
                'ad_id': ad_id,
                'metric': metric_name,
                'forecast': forecast_data,
                'trend_direction': trend_direction,
                'trend_strength': abs(recent_trend),
                'forecast_confidence': 0.8  # Prophet typically 80% confidence
            }
            
        except Exception as e:
            self.logger.error(f"Error forecasting {metric_name} for {ad_id}: {e}")
            return {}

# =====================================================
# CREATIVE SIMILARITY ANALYSIS
# =====================================================

class CreativeSimilarityAnalyzer:
    """Analyze creative similarity using embeddings."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.logger = logging.getLogger(f"{__name__}.CreativeSimilarityAnalyzer")
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.available = True
        except ImportError:
            self.logger.warning("sentence-transformers not available - creative similarity disabled")
            self.model = None
            self.available = False
    
    def analyze_creative_text(self, creative_text: str) -> np.ndarray:
        """Generate embedding for creative text."""
        if not self.available or not creative_text:
            return np.zeros(384)
        
        try:
            embedding = self.model.encode(creative_text)
            return embedding
        except Exception as e:
            self.logger.error(f"Error encoding creative: {e}")
            return np.zeros(384)
    
    def find_similar_creatives(self, creative_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar high-performing creatives."""
        if not self.available:
            return []
        
        try:
            # Get creative embedding
            response = self.client.table('creative_intelligence').select('similarity_vector, performance_score').eq(
                'creative_id', creative_id
            ).execute()
            
            if not response.data:
                return []
            
            target_vector = response.data[0].get('similarity_vector', [])
            if not target_vector:
                return []
            
            # Find similar creatives with high performance
            all_creatives = self.client.table('creative_intelligence').select('*').gte(
                'performance_score', 0.6
            ).execute()
            
            if not all_creatives.data:
                return []
            
            similarities = []
            target_vec = np.array(target_vector)
            
            for creative in all_creatives.data:
                if creative['creative_id'] == creative_id:
                    continue
                
                vec = np.array(creative.get('similarity_vector', []))
                if len(vec) == 0:
                    continue
                
                # Cosine similarity
                similarity = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec) + 1e-6)
                
                similarities.append({
                    'creative_id': creative['creative_id'],
                    'similarity_score': float(similarity),
                    'performance_score': creative.get('performance_score', 0),
                    'avg_cpa': creative.get('avg_cpa', 0),
                    'avg_roas': creative.get('avg_roas', 0)
                })
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding similar creatives: {e}")
            return []

# =====================================================
# CAUSAL IMPACT ANALYSIS
# =====================================================

class CausalImpactAnalyzer:
    """Causal inference to determine what actually drives purchases."""
    
    def __init__(self, supabase_client: Client):
        self.client = supabase_client
        self.logger = logging.getLogger(f"{__name__}.CausalImpactAnalyzer")
    
    def analyze_feature_causality(self, stage: str, days_back: int = 30) -> Dict[str, Any]:
        """Analyze causal relationships between features and purchases."""
        try:
            # Get performance data
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            response = self.client.table('performance_metrics').select('*').eq(
                'stage', stage
            ).gte('date_start', start_date).execute()
            
            if not response.data or len(response.data) < 20:
                return {}
            
            df = pd.DataFrame(response.data)
            
            # Convert to numeric
            numeric_features = ['ctr', 'cpm', 'cpc', 'impressions', 'clicks', 'add_to_cart', 'initiate_checkout']
            for col in numeric_features:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df['purchases'] = pd.to_numeric(df['purchases'], errors='coerce').fillna(0)
            
            # Causal analysis using correlation with time-lag
            causal_impacts = {}
            
            for feature in numeric_features:
                if feature not in df.columns:
                    continue
                
                # Pearson correlation
                try:
                    correlation, p_value = stats.pearsonr(df[feature], df['purchases'])
                    
                    # Only significant relationships
                    if p_value < 0.05 and abs(correlation) > 0.3:
                        causal_impacts[feature] = {
                            'correlation': float(correlation),
                            'p_value': float(p_value),
                            'impact_strength': abs(correlation),
                            'direction': 'positive' if correlation > 0 else 'negative',
                            'is_significant': True
                        }
                except Exception:
                    pass
            
            # Rank features by causal impact
            ranked_features = sorted(
                causal_impacts.items(),
                key=lambda x: x[1]['impact_strength'],
                reverse=True
            )
            
            return {
                'stage': stage,
                'causal_features': dict(ranked_features[:10]),  # Top 10
                'strongest_predictor': ranked_features[0][0] if ranked_features else None,
                'analysis_date': now_utc().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing causality: {e}")
            return {}

# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_model_validator(supabase_url: str, supabase_key: str) -> ModelValidator:
    """Create model validator."""
    from supabase import create_client
    client = create_client(supabase_url, supabase_key)
    return ModelValidator(client)

def create_data_progress_tracker(supabase_url: str, supabase_key: str) -> DataProgressTracker:
    """Create data progress tracker."""
    from supabase import create_client
    client = create_client(supabase_url, supabase_key)
    return DataProgressTracker(client)

def create_anomaly_detector(supabase_url: str, supabase_key: str) -> AnomalyDetector:
    """Create anomaly detector."""
    from supabase import create_client
    client = create_client(supabase_url, supabase_key)
    return AnomalyDetector(client)

def create_time_series_forecaster(supabase_url: str, supabase_key: str) -> TimeSeriesForecaster:
    """Create time-series forecaster."""
    from supabase import create_client
    client = create_client(supabase_url, supabase_key)
    return TimeSeriesForecaster(client)

def create_creative_similarity_analyzer(supabase_url: str, supabase_key: str) -> CreativeSimilarityAnalyzer:
    """Create creative similarity analyzer."""
    from supabase import create_client
    client = create_client(supabase_url, supabase_key)
    return CreativeSimilarityAnalyzer(client)

def create_causal_impact_analyzer(supabase_url: str, supabase_key: str) -> CausalImpactAnalyzer:
    """Create causal impact analyzer."""
    from supabase import create_client
    client = create_client(supabase_url, supabase_key)
    return CausalImpactAnalyzer(client)

