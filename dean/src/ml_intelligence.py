"""
DEAN SELF-LEARNING META ADS AUTOMATION SYSTEM
Advanced Machine Learning Intelligence Layer

This module implements the core ML intelligence system including:
- XGBoost prediction engines
- Cross-stage transfer learning
- Temporal modeling and trend analysis
- Performance pattern recognition
- Adaptive rule learning
"""

from __future__ import annotations

import json
import logging
import pickle
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False
from supabase import create_client, Client
import joblib

from utils import now_utc, today_ymd_account, yesterday_ymd_account

warnings.filterwarnings('ignore', category=FutureWarning)

logger = logging.getLogger(__name__)

# =====================================================
# CORE ML INTELLIGENCE CLASSES
# =====================================================

@dataclass
class MLConfig:
    """Configuration for ML intelligence system."""
    # Model parameters
    xgb_params: Dict[str, Any] = None
    retrain_frequency_hours: int = 24
    prediction_horizon_hours: int = 24
    confidence_threshold: float = 0.7
    
    # Feature engineering
    rolling_windows: List[int] = None
    feature_importance_threshold: float = 0.01
    max_features: int = 100
    
    # Learning parameters
    learning_rate: float = 0.1
    adaptation_rate: float = 0.05
    memory_decay: float = 0.95
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42,
                'n_jobs': -1
            }
        
        if self.rolling_windows is None:
            self.rolling_windows = [1, 3, 7, 14, 30]

@dataclass
class PredictionResult:
    """Container for ML prediction results."""
    predicted_value: float
    confidence_score: float
    prediction_interval_lower: float
    prediction_interval_upper: float
    feature_importance: Dict[str, float]
    model_version: str
    prediction_horizon_hours: int
    created_at: datetime

@dataclass
class LearningEvent:
    """Container for learning events and insights."""
    event_type: str
    ad_id: str
    lifecycle_id: str
    stage: str
    learning_data: Dict[str, Any]
    confidence_score: float
    impact_score: float
    created_at: datetime

class SupabaseMLClient:
    """Enhanced Supabase client for ML operations."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.logger = logging.getLogger(f"{__name__}.SupabaseMLClient")
    
    def get_performance_data(self, 
                           ad_ids: Optional[List[str]] = None,
                           stages: Optional[List[str]] = None,
                           days_back: int = 30) -> pd.DataFrame:
        """Fetch performance data for ML training."""
        try:
            # Build query
            query = self.client.table('performance_metrics').select('*')
            
            if ad_ids:
                query = query.in_('ad_id', ad_ids)
            if stages:
                query = query.in_('stage', stages)
            
            # Date filter
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            query = query.gte('date_start', start_date)
            
            response = query.execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            
            # Convert types
            numeric_columns = [
                'spend', 'impressions', 'clicks', 'purchases', 'add_to_cart',
                'initiate_checkout', 'revenue', 'ctr', 'cpm', 'cpc', 'cpa',
                'roas', 'aov', 'three_sec_views', 'video_views', 'watch_time',
                'dwell_time', 'frequency', 'atc_rate', 'ic_rate', 'purchase_rate',
                'atc_to_ic_rate', 'ic_to_purchase_rate', 'performance_quality_score',
                'stability_score', 'momentum_score', 'fatigue_index'
            ]
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert dates
            df['date_start'] = pd.to_datetime(df['date_start'])
            df['date_end'] = pd.to_datetime(df['date_end'])
            df['created_at'] = pd.to_datetime(df['created_at'])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching performance data: {e}")
            return pd.DataFrame()
    
    def get_time_series_data(self, 
                           ad_id: str,
                           metric_name: str,
                           days_back: int = 30) -> pd.DataFrame:
        """Fetch time series data for temporal modeling."""
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            response = self.client.table('time_series_data').select('*').eq(
                'ad_id', ad_id
            ).eq('metric_name', metric_name).gte('timestamp', start_date).execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
            
            return df.sort_values('timestamp')
            
        except Exception as e:
            self.logger.error(f"Error fetching time series data: {e}")
            return pd.DataFrame()
    
    def save_prediction(self, prediction: PredictionResult, 
                       ad_id: str, lifecycle_id: str, stage: str,
                       model_id: str) -> str:
        """Save ML prediction to database."""
        try:
            prediction_id = str(uuid.uuid4())
            
            data = {
                'id': prediction_id,
                'ad_id': ad_id,
                'lifecycle_id': lifecycle_id,
                'model_id': model_id,
                'stage': stage,
                'prediction_type': 'performance',
                'predicted_value': float(prediction.predicted_value),
                'confidence_score': float(prediction.confidence_score),
                'prediction_interval_lower': float(prediction.prediction_interval_lower),
                'prediction_interval_upper': float(prediction.prediction_interval_upper),
                'features': prediction.feature_importance,
                'prediction_horizon_hours': prediction.prediction_horizon_hours,
                'created_at': prediction.created_at.isoformat(),
                'expires_at': (prediction.created_at + timedelta(days=7)).isoformat()
            }
            
            response = self.client.table('ml_predictions').insert(data).execute()
            
            if response.data:
                self.logger.info(f"Saved prediction {prediction_id} for ad {ad_id}")
                return prediction_id
            else:
                self.logger.error(f"Failed to save prediction for ad {ad_id}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving prediction: {e}")
            return None
    
    def save_learning_event(self, event: LearningEvent) -> str:
        """Save learning event to database."""
        try:
            event_id = str(uuid.uuid4())
            
            data = {
                'id': event_id,
                'event_type': event.event_type,
                'ad_id': event.ad_id,
                'lifecycle_id': event.lifecycle_id,
                'from_stage': event.stage,
                'to_stage': event.stage,  # Will be updated for promotions
                'learning_data': event.learning_data,
                'confidence_score': float(event.confidence_score),
                'impact_score': float(event.impact_score),
                'created_at': event.created_at.isoformat()
            }
            
            response = self.client.table('learning_events').insert(data).execute()
            
            if response.data:
                self.logger.info(f"Saved learning event {event_id}")
                return event_id
            else:
                self.logger.error(f"Failed to save learning event")
                return None
                
        except Exception as e:
            self.logger.error(f"Error saving learning event: {e}")
            return None

class FeatureEngineer:
    """Advanced feature engineering for ML models."""
    
    def __init__(self, config: MLConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.FeatureEngineer")
    
    def create_rolling_features(self, df: pd.DataFrame, 
                              group_cols: List[str],
                              value_cols: List[str]) -> pd.DataFrame:
        """Create rolling window features."""
        try:
            df = df.copy()
            
            for group_col in group_cols:
                for value_col in value_cols:
                    if value_col not in df.columns:
                        continue
                    
                    for window in self.config.rolling_windows:
                        # Rolling mean
                        df[f'{value_col}_rolling_mean_{window}d'] = df.groupby(group_col)[value_col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).mean()
                        )
                        
                        # Rolling std
                        df[f'{value_col}_rolling_std_{window}d'] = df.groupby(group_col)[value_col].transform(
                            lambda x: x.rolling(window=window, min_periods=1).std()
                        )
                        
                        # Rolling trend (slope)
                        df[f'{value_col}_rolling_trend_{window}d'] = df.groupby(group_col)[value_col].transform(
                            lambda x: x.rolling(window=window, min_periods=2).apply(
                                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0
                            )
                        )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating rolling features: {e}")
            return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between metrics."""
        try:
            df = df.copy()
            
            # CTR * ATC rate interaction
            if 'ctr' in df.columns and 'atc_rate' in df.columns:
                df['ctr_atc_interaction'] = df['ctr'] * df['atc_rate']
            
            # CPA * ROAS efficiency
            if 'cpa' in df.columns and 'roas' in df.columns:
                df['cpa_roas_efficiency'] = df['roas'] / (df['cpa'] + 1e-6)
            
            # Engagement quality score
            engagement_cols = ['ctr', 'atc_rate', 'purchase_rate']
            if all(col in df.columns for col in engagement_cols):
                df['engagement_quality'] = (
                    df['ctr'] * 0.4 + 
                    df['atc_rate'] * 0.3 + 
                    df['purchase_rate'] * 0.3
                )
            
            # Performance stability
            stability_cols = ['stability_score', 'momentum_score', 'fatigue_index']
            if all(col in df.columns for col in stability_cols):
                df['performance_stability'] = (
                    df['stability_score'] * 0.5 - 
                    df['fatigue_index'] * 0.3 + 
                    df['momentum_score'] * 0.2
                )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating interaction features: {e}")
            return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal and seasonal features."""
        try:
            df = df.copy()
            
            if 'date_start' in df.columns:
                df['day_of_week'] = pd.to_datetime(df['date_start']).dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
                df['month'] = pd.to_datetime(df['date_start']).dt.month
                df['quarter'] = pd.to_datetime(df['date_start']).dt.quarter
                
                # Cyclical encoding for seasonality
                df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating temporal features: {e}")
            return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced ML features."""
        try:
            df = df.copy()
            
            # Performance momentum
            if 'ctr' in df.columns:
                df['ctr_momentum'] = df.groupby('ad_id')['ctr'].pct_change()
                df['ctr_acceleration'] = df.groupby('ad_id')['ctr_momentum'].diff()
            
            # Volatility measures
            volatility_cols = ['ctr', 'cpa', 'roas']
            for col in volatility_cols:
                if col in df.columns:
                    df[f'{col}_volatility'] = df.groupby('ad_id')[col].rolling(
                        window=7, min_periods=3
                    ).std()
            
            # Relative performance
            if 'cpa' in df.columns:
                df['cpa_vs_account'] = df['cpa'] / df.groupby('stage')['cpa'].transform('mean')
                df['cpa_percentile'] = df.groupby('stage')['cpa'].rank(pct=True)
            
            # Fatigue indicators
            if 'fatigue_index' in df.columns:
                df['fatigue_trend'] = df.groupby('ad_id')['fatigue_index'].diff()
                df['fatigue_acceleration'] = df.groupby('ad_id')['fatigue_trend'].diff()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating advanced features: {e}")
            return df

class XGBoostPredictor:
    """XGBoost-based prediction engine for performance forecasting."""
    
    def __init__(self, config: MLConfig, supabase_client: SupabaseMLClient):
        self.config = config
        self.supabase = supabase_client
        self.feature_engineer = FeatureEngineer(config)
        self.logger = logging.getLogger(f"{__name__}.XGBoostPredictor")
        
        # Model storage
        self.models: Dict[str, any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_importance: Dict[str, Dict[str, float]] = {}
        
    def prepare_training_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data for ML models."""
        try:
            # Create features
            df_features = self.feature_engineer.create_rolling_features(
                df, ['ad_id'], ['ctr', 'cpa', 'roas', 'spend', 'purchases']
            )
            df_features = self.feature_engineer.create_interaction_features(df_features)
            df_features = self.feature_engineer.create_temporal_features(df_features)
            df_features = self.feature_engineer.create_advanced_features(df_features)
            
            # Select features
            feature_cols = [col for col in df_features.columns 
                           if col not in ['ad_id', 'id', 'date_start', 'date_end', 'created_at', target_col]]
            
            # Remove non-numeric columns
            numeric_cols = df_features[feature_cols].select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in feature_cols if col in numeric_cols]
            
            # Handle missing values
            df_features[feature_cols] = df_features[feature_cols].fillna(0)
            
            # Prepare X and y
            X = df_features[feature_cols].values
            y = df_features[target_col].fillna(0).values
            
            return X, y, feature_cols
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([]), []
    
    def train_model(self, model_type: str, stage: str, target_col: str) -> bool:
        """Train XGBoost model for specific prediction task."""
        try:
            # Get training data
            df = self.supabase.get_performance_data(stages=[stage])
            if df.empty:
                self.logger.warning(f"No data available for training {model_type} model for {stage}")
                return False
            
            # Prepare data
            X, y, feature_cols = self.prepare_training_data(df, target_col)
            if len(X) == 0:
                self.logger.warning(f"No features available for training {model_type} model")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train model
            if XGBOOST_AVAILABLE:
                model = xgb.XGBRegressor(**self.config.xgb_params)
            else:
                # Fallback to GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=self.config.xgb_params.get('n_estimators', 100),
                    max_depth=self.config.xgb_params.get('max_depth', 6),
                    learning_rate=self.config.xgb_params.get('learning_rate', 0.1),
                    random_state=42
                )
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            self.logger.info(f"Trained {model_type} model for {stage}: MAE={mae:.4f}, R²={r2:.4f}")
            
            # Store model and scaler
            model_key = f"{model_type}_{stage}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            # Store feature importance
            feature_importance = dict(zip(feature_cols, model.feature_importances_))
            self.feature_importance[model_key] = feature_importance
            
            # Save to Supabase
            self.save_model_to_supabase(model_type, stage, model, scaler, feature_cols, feature_importance)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training {model_type} model for {stage}: {e}")
            return False
    
    def predict(self, model_type: str, stage: str, ad_id: str, 
                features: Dict[str, float]) -> Optional[PredictionResult]:
        """Make prediction using trained model."""
        try:
            model_key = f"{model_type}_{stage}"
            
            if model_key not in self.models:
                self.logger.warning(f"Model {model_key} not found, attempting to load from Supabase")
                if not self.load_model_from_supabase(model_type, stage):
                    return None
            
            model = self.models[model_key]
            scaler = self.scalers[model_key]
            
            # Prepare features
            feature_vector = np.array([features.get(col, 0) for col in model.feature_names_in_])
            feature_vector_scaled = scaler.transform([feature_vector])
            
            # Make prediction
            prediction = model.predict(feature_vector_scaled)[0]
            
            # Calculate confidence (based on model uncertainty)
            # This is a simplified approach - in production, use proper uncertainty quantification
            confidence = min(0.95, max(0.1, 1.0 - abs(prediction) * 0.01))
            
            # Prediction intervals (simplified)
            std_error = np.std(model.predict(feature_vector_scaled))
            interval_lower = prediction - 1.96 * std_error
            interval_upper = prediction + 1.96 * std_error
            
            result = PredictionResult(
                predicted_value=float(prediction),
                confidence_score=float(confidence),
                prediction_interval_lower=float(interval_lower),
                prediction_interval_upper=float(interval_upper),
                feature_importance=self.feature_importance.get(model_key, {}),
                model_version=f"{model_type}_{stage}_v1",
                prediction_horizon_hours=self.config.prediction_horizon_hours,
                created_at=now_utc()
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return None
    
    def save_model_to_supabase(self, model_type: str, stage: str, 
                              model: any, scaler: StandardScaler,
                              feature_cols: List[str], feature_importance: Dict[str, float]) -> bool:
        """Save trained model to Supabase."""
        try:
            # Serialize model
            model_data = pickle.dumps(model)
            
            # Create model metadata
            metadata = {
                'feature_columns': feature_cols,
                'feature_importance': feature_importance,
                'training_date': now_utc().isoformat(),
                'model_type': model_type,
                'stage': stage
            }
            
            # Performance metrics (simplified)
            performance_metrics = {
                'feature_count': len(feature_cols),
                'model_size_bytes': len(model_data)
            }
            
            data = {
                'model_type': model_type,
                'stage': stage,
                'model_name': f"{model_type}_{stage}_v1",
                'model_data': model_data.hex(),  # Convert to hex for storage
                'model_metadata': metadata,
                'features_used': feature_cols,
                'performance_metrics': performance_metrics,
                'is_active': True,
                'trained_at': now_utc().isoformat()
            }
            
            response = self.supabase.client.table('ml_models').insert(data).execute()
            
            if response.data:
                self.logger.info(f"Saved {model_type} model for {stage} to Supabase")
                return True
            else:
                self.logger.error(f"Failed to save {model_type} model for {stage}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving model to Supabase: {e}")
            return False
    
    def load_model_from_supabase(self, model_type: str, stage: str) -> bool:
        """Load trained model from Supabase."""
        try:
            response = self.supabase.client.table('ml_models').select('*').eq(
                'model_type', model_type
            ).eq('stage', stage).eq('is_active', True).execute()
            
            if not response.data:
                self.logger.warning(f"No active model found for {model_type} in {stage}")
                return False
            
            model_data = response.data[0]
            
            # Deserialize model
            model_bytes = bytes.fromhex(model_data['model_data'])
            model = pickle.loads(model_bytes)
            
            # Create scaler (simplified - in production, store scaler separately)
            scaler = StandardScaler()
            
            # Store in memory
            model_key = f"{model_type}_{stage}"
            self.models[model_key] = model
            self.scalers[model_key] = scaler
            
            if 'feature_importance' in model_data.get('model_metadata', {}):
                self.feature_importance[model_key] = model_data['model_metadata']['feature_importance']
            
            self.logger.info(f"Loaded {model_type} model for {stage} from Supabase")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model from Supabase: {e}")
            return False

class TemporalAnalyzer:
    """Advanced temporal analysis and trend detection."""
    
    def __init__(self, supabase_client: SupabaseMLClient):
        self.supabase = supabase_client
        self.logger = logging.getLogger(f"{__name__}.TemporalAnalyzer")
    
    def analyze_trends(self, ad_id: str, metric_name: str, 
                      days_back: int = 30) -> Dict[str, Any]:
        """Analyze temporal trends for an ad."""
        try:
            # Get time series data
            df = self.supabase.get_time_series_data(ad_id, metric_name, days_back)
            if df.empty:
                return {}
            
            # Calculate trend
            if len(df) < 3:
                return {'trend': 'insufficient_data'}
            
            # Linear trend
            x = np.arange(len(df))
            y = df['metric_value'].values
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Trend classification
            if p_value < 0.05:  # Significant trend
                if slope > 0:
                    trend = 'improving'
                else:
                    trend = 'declining'
            else:
                trend = 'stable'
            
            # Volatility
            volatility = np.std(y) / (np.mean(y) + 1e-6)
            
            # Momentum (recent vs older performance)
            if len(df) >= 7:
                recent_avg = df.tail(3)['metric_value'].mean()
                older_avg = df.head(3)['metric_value'].mean()
                momentum = (recent_avg - older_avg) / (older_avg + 1e-6)
            else:
                momentum = 0
            
            return {
                'trend': trend,
                'slope': float(slope),
                'r_squared': float(r_value ** 2),
                'p_value': float(p_value),
                'volatility': float(volatility),
                'momentum': float(momentum),
                'data_points': len(df),
                'latest_value': float(y[-1]),
                'trend_strength': abs(slope) * r_value
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends for {ad_id}: {e}")
            return {}
    
    def detect_fatigue(self, ad_id: str, days_back: int = 14) -> Dict[str, Any]:
        """Detect performance fatigue patterns."""
        try:
            # Get key metrics
            metrics = ['ctr', 'cpa', 'roas']
            fatigue_signals = {}
            
            for metric in metrics:
                trend_data = self.analyze_trends(ad_id, metric, days_back)
                if trend_data:
                    # Fatigue indicators
                    if trend_data['trend'] == 'declining' and trend_data['p_value'] < 0.1:
                        fatigue_signals[metric] = {
                            'declining': True,
                            'strength': abs(trend_data['slope']),
                            'confidence': 1 - trend_data['p_value']
                        }
                    else:
                        fatigue_signals[metric] = {
                            'declining': False,
                            'strength': 0,
                            'confidence': 0
                        }
            
            # Overall fatigue score
            declining_metrics = sum(1 for signal in fatigue_signals.values() if signal['declining'])
            fatigue_score = declining_metrics / len(metrics)
            
            return {
                'fatigue_score': fatigue_score,
                'fatigue_signals': fatigue_signals,
                'is_fatigued': fatigue_score > 0.5,
                'confidence': np.mean([s['confidence'] for s in fatigue_signals.values()])
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting fatigue for {ad_id}: {e}")
            return {}

class CrossStageLearner:
    """Cross-stage transfer learning system."""
    
    def __init__(self, supabase_client: SupabaseMLClient):
        self.supabase = supabase_client
        self.logger = logging.getLogger(f"{__name__}.CrossStageLearner")
    
    def transfer_insights(self, from_stage: str, to_stage: str, 
                         ad_id: str) -> Dict[str, Any]:
        """Transfer insights between stages."""
        try:
            # Get performance data from source stage
            source_data = self.supabase.get_performance_data(
                ad_ids=[ad_id], stages=[from_stage]
            )
            
            if source_data.empty:
                return {}
            
            # Extract key insights
            latest_performance = source_data.iloc[-1]
            
            insights = {
                'ctr_pattern': latest_performance.get('ctr', 0),
                'cpa_pattern': latest_performance.get('cpa', 0),
                'roas_pattern': latest_performance.get('roas', 0),
                'quality_score': latest_performance.get('performance_quality_score', 0),
                'stability': latest_performance.get('stability_score', 0),
                'momentum': latest_performance.get('momentum_score', 0),
                'fatigue_risk': latest_performance.get('fatigue_index', 0)
            }
            
            # Create learning event
            learning_event = LearningEvent(
                event_type='cross_stage_transfer',
                ad_id=ad_id,
                lifecycle_id=latest_performance.get('lifecycle_id', ''),
                stage=to_stage,
                learning_data={
                    'from_stage': from_stage,
                    'insights': insights,
                    'transfer_confidence': 0.8
                },
                confidence_score=0.8,
                impact_score=0.7,
                created_at=now_utc()
            )
            
            # Save learning event
            self.supabase.save_learning_event(learning_event)
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Error transferring insights: {e}")
            return {}

class MLIntelligenceSystem:
    """Main ML Intelligence System orchestrator."""
    
    def __init__(self, supabase_url: str, supabase_key: str, config: Optional[MLConfig] = None):
        self.config = config or MLConfig()
        self.supabase = SupabaseMLClient(supabase_url, supabase_key)
        self.predictor = XGBoostPredictor(self.config, self.supabase)
        self.temporal_analyzer = TemporalAnalyzer(self.supabase)
        self.cross_stage_learner = CrossStageLearner(self.supabase)
        self.logger = logging.getLogger(f"{__name__}.MLIntelligenceSystem")
    
    def initialize_models(self) -> bool:
        """Initialize all ML models."""
        try:
            self.logger.info("Initializing ML models...")
            
            # Train core prediction models
            models_to_train = [
                ('performance_predictor', 'testing', 'cpa'),
                ('performance_predictor', 'validation', 'cpa'),
                ('performance_predictor', 'scaling', 'cpa'),
                ('roas_predictor', 'testing', 'roas'),
                ('roas_predictor', 'validation', 'roas'),
                ('roas_predictor', 'scaling', 'roas'),
                ('purchase_probability', 'testing', 'purchases'),
                ('purchase_probability', 'validation', 'purchases'),
                ('purchase_probability', 'scaling', 'purchases')
            ]
            
            success_count = 0
            for model_type, stage, target in models_to_train:
                if self.predictor.train_model(model_type, stage, target):
                    success_count += 1
            
            self.logger.info(f"Initialized {success_count}/{len(models_to_train)} models")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            return False
    
    def predict_performance(self, ad_id: str, stage: str, 
                           features: Dict[str, float]) -> Optional[PredictionResult]:
        """Predict ad performance."""
        try:
            # Get CPA prediction
            cpa_prediction = self.predictor.predict('performance_predictor', stage, ad_id, features)
            
            if cpa_prediction:
                # Save prediction to database
                self.supabase.save_prediction(
                    cpa_prediction, ad_id, features.get('lifecycle_id', ''), stage,
                    f"performance_predictor_{stage}"
                )
            
            return cpa_prediction
            
        except Exception as e:
            self.logger.error(f"Error predicting performance: {e}")
            return None
    
    def analyze_ad_intelligence(self, ad_id: str, stage: str) -> Dict[str, Any]:
        """Comprehensive ad intelligence analysis."""
        try:
            # Get current performance data
            df = self.supabase.get_performance_data(ad_ids=[ad_id], stages=[stage])
            if df.empty:
                return {}
            
            latest = df.iloc[-1]
            
            # Temporal analysis
            trend_analysis = {}
            for metric in ['ctr', 'cpa', 'roas']:
                trend_analysis[metric] = self.temporal_analyzer.analyze_trends(ad_id, metric)
            
            # Fatigue detection
            fatigue_analysis = self.temporal_analyzer.detect_fatigue(ad_id)
            
            # Cross-stage insights
            cross_stage_insights = {}
            if stage != 'testing':
                prev_stage = 'testing' if stage == 'validation' else 'validation'
                cross_stage_insights = self.cross_stage_learner.transfer_insights(
                    prev_stage, stage, ad_id
                )
            
            # Performance predictions
            features = {
                'ctr': latest.get('ctr', 0),
                'cpa': latest.get('cpa', 0),
                'roas': latest.get('roas', 0),
                'spend': latest.get('spend', 0),
                'purchases': latest.get('purchases', 0),
                'performance_quality_score': latest.get('performance_quality_score', 0),
                'stability_score': latest.get('stability_score', 0),
                'fatigue_index': latest.get('fatigue_index', 0),
                'lifecycle_id': latest.get('lifecycle_id', '')
            }
            
            predictions = self.predict_performance(ad_id, stage, features)
            
            return {
                'current_performance': latest.to_dict(),
                'trend_analysis': trend_analysis,
                'fatigue_analysis': fatigue_analysis,
                'cross_stage_insights': cross_stage_insights,
                'predictions': predictions.__dict__ if predictions else None,
                'intelligence_score': self.calculate_intelligence_score(
                    latest, trend_analysis, fatigue_analysis, predictions
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing ad intelligence: {e}")
            return {}
    
    def calculate_intelligence_score(self, performance: pd.Series, 
                                   trends: Dict[str, Any], 
                                   fatigue: Dict[str, Any],
                                   predictions: Optional[PredictionResult]) -> float:
        """Calculate overall intelligence score for an ad."""
        try:
            score = 0.0
            
            # Performance quality (40%)
            quality_score = performance.get('performance_quality_score', 0) / 100
            score += quality_score * 0.4
            
            # Trend stability (30%)
            stable_trends = sum(1 for trend in trends.values() 
                              if trend.get('trend') == 'stable' or trend.get('trend') == 'improving')
            trend_score = stable_trends / max(len(trends), 1)
            score += trend_score * 0.3
            
            # Fatigue resistance (20%)
            fatigue_score = 1.0 - fatigue.get('fatigue_score', 0)
            score += fatigue_score * 0.2
            
            # Prediction confidence (10%)
            if predictions:
                confidence_score = predictions.confidence_score
                score += confidence_score * 0.1
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            self.logger.error(f"Error calculating intelligence score: {e}")
            return 0.0

# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_ml_system(supabase_url: str, supabase_key: str, 
                    config: Optional[MLConfig] = None) -> MLIntelligenceSystem:
    """Create and initialize ML intelligence system."""
    system = MLIntelligenceSystem(supabase_url, supabase_key, config)
    system.initialize_models()
    return system

def analyze_ad_performance(ml_system: MLIntelligenceSystem, 
                          ad_id: str, stage: str) -> Dict[str, Any]:
    """Analyze ad performance using ML intelligence."""
    return ml_system.analyze_ad_intelligence(ad_id, stage)

def predict_ad_cpa(ml_system: MLIntelligenceSystem, 
                  ad_id: str, stage: str, 
                  features: Dict[str, float]) -> Optional[float]:
    """Predict ad CPA using ML models."""
    prediction = ml_system.predict_performance(ad_id, stage, features)
    return prediction.predicted_value if prediction else None
