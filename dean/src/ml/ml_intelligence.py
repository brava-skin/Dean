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

from infrastructure.utils import now_utc, today_ymd_account, yesterday_ymd_account

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
    """Enhanced Supabase client for ML operations with historical data integration."""
    
    def __init__(self, supabase_url: str, supabase_key: str):
        self.client: Client = create_client(supabase_url, supabase_key)
        self.logger = logging.getLogger(f"{__name__}.SupabaseMLClient")
    
    def _safe_float(self, value, max_val=999999999.99):
        """Safely convert value to float with bounds checking."""
        try:
            val = float(value or 0)
            # Handle infinity and NaN
            if not (val == val) or val == float('inf') or val == float('-inf'):
                return 0.0
            # Bound the value to prevent overflow
            return min(max(val, -max_val), max_val)
        except (ValueError, TypeError):
            return 0.0
    
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
    
    def get_historical_metrics(self, ad_id: str, metric_names: List[str], 
                             days_back: int = 30) -> pd.DataFrame:
        """Get comprehensive historical metrics for an ad."""
        try:
            start_date = (datetime.now() - timedelta(days=days_back)).isoformat()
            
            # Query historical_data table
            response = self.client.table('historical_data').select('*').eq(
                'ad_id', ad_id
            ).in_('metric_name', metric_names).gte('ts_iso', start_date).order(
                'ts_epoch', desc=False
            ).execute()
            
            if not response.data:
                return pd.DataFrame()
            
            df = pd.DataFrame(response.data)
            df['ts_iso'] = pd.to_datetime(df['ts_iso'])
            
            # Pivot to get metrics as columns
            pivot_df = df.pivot_table(
                index=['ts_iso', 'ad_id', 'stage'], 
                columns='metric_name', 
                values='metric_value', 
                aggfunc='mean'
            ).reset_index()
            
            # Fill missing values
            pivot_df = pivot_df.fillna(0)
            
            return pivot_df
            
        except Exception as e:
            self.logger.error(f"Error getting historical metrics: {e}")
            return pd.DataFrame()
    
    def get_ad_age_features(self, ad_id: str) -> Dict[str, Any]:
        """Get age-based features for an ad."""
        try:
            # Get ad creation time
            response = self.client.table('ad_creation_times').select('*').eq(
                'ad_id', ad_id
            ).execute()
            
            if not response.data:
                return {}
            
            creation_data = response.data[0]
            created_at = datetime.fromisoformat(creation_data['created_at_iso'].replace('Z', '+00:00'))
            
            # Calculate age in different units
            now = now_utc()
            age_delta = now - created_at
            
            age_features = {
                'ad_age_days': age_delta.total_seconds() / (24 * 3600),
                'ad_age_hours': age_delta.total_seconds() / 3600,
                'ad_age_weeks': age_delta.days / 7,
                'created_at_epoch': creation_data['created_at_epoch'],
                'stage': creation_data['stage']
            }
            
            return age_features
            
        except Exception as e:
            self.logger.error(f"Error getting ad age features: {e}")
            return {}
    
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
                'prediction_value': self._safe_float(prediction.predicted_value, 999999999.99),  # Bounded prediction value
                'created_at': prediction.created_at.isoformat()
                # Removed fields that don't exist in actual schema
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
                'stage': event.stage,
                'created_at': event.created_at.isoformat()
                # Removed fields that don't exist in actual schema
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
                        
                        # Rolling std (safe)
                        def safe_rolling_std(series):
                            std_vals = series.rolling(window=window, min_periods=1).std()
                            return std_vals.replace([np.inf, -np.inf, np.nan], 0)
                        
                        df[f'{value_col}_rolling_std_{window}d'] = df.groupby(group_col)[value_col].transform(safe_rolling_std)
                        
                        # Rolling trend (slope) - only for windows >= 2 (safe)
                        if window >= 2:
                            def safe_rolling_trend(series):
                                def calc_trend(y):
                                    try:
                                        if len(y) >= 2:
                                            slope = np.polyfit(range(len(y)), y, 1)[0]
                                            return slope if not (np.isinf(slope) or np.isnan(slope)) else 0
                                        return 0
                                    except:
                                        return 0
                                
                                trend_vals = series.rolling(window=window, min_periods=2).apply(calc_trend)
                                # Replace NaN values with 0
                                return trend_vals.fillna(0)
                            
                            df[f'{value_col}_rolling_trend_{window}d'] = df.groupby(group_col)[value_col].transform(safe_rolling_trend)
            
            # Clean infinity and NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                df[col] = df[col].clip(-1e6, 1e6)
            
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
            
            # CPA * ROAS efficiency (safe division)
            if 'cpa' in df.columns and 'roas' in df.columns:
                # Use safe division to prevent infinity
                cpa_safe = df['cpa'].replace([0, np.inf, -np.inf], np.nan).fillna(1.0)
                df['cpa_roas_efficiency'] = df['roas'] / cpa_safe
                # Cap extreme values
                df['cpa_roas_efficiency'] = df['cpa_roas_efficiency'].clip(-1000, 1000)
            
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
            
            # Clean infinity and NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                df[col] = df[col].clip(-1e6, 1e6)
            
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
            
            # Clean infinity and NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                df[col] = df[col].clip(-1e6, 1e6)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error creating temporal features: {e}")
            return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced ML features."""
        try:
            df = df.copy()
            
            # Performance momentum (safe percentage change)
            if 'ctr' in df.columns:
                # Safe percentage change that handles division by zero
                def safe_pct_change(series):
                    pct = series.pct_change()
                    # Replace infinity and NaN with 0
                    return pct.replace([np.inf, -np.inf, np.nan], 0)
                
                df['ctr_momentum'] = df.groupby('ad_id')['ctr'].transform(safe_pct_change)
                df['ctr_acceleration'] = df.groupby('ad_id')['ctr_momentum'].transform(safe_pct_change)
            
            # Volatility measures (safe rolling std)
            volatility_cols = ['ctr', 'cpa', 'roas']
            for col in volatility_cols:
                if col in df.columns:
                    def safe_rolling_std(series):
                        std_vals = series.rolling(window=7, min_periods=3).std()
                        # Replace NaN and infinity with 0
                        return std_vals.replace([np.inf, -np.inf, np.nan], 0)
                    
                    df[f'{col}_volatility'] = df.groupby('ad_id')[col].transform(safe_rolling_std)
            
            # Relative performance (safe division and ranking)
            if 'cpa' in df.columns:
                # Safe division for relative performance
                stage_mean = df.groupby('stage')['cpa'].transform('mean')
                stage_mean_safe = stage_mean.replace([0, np.inf, -np.inf], np.nan).fillna(1.0)
                df['cpa_vs_account'] = (df['cpa'] / stage_mean_safe).clip(-100, 100)
                
                # Safe ranking that handles identical values
                df['cpa_percentile'] = df.groupby('stage')['cpa'].rank(pct=True, method='average')
                df['cpa_percentile'] = df['cpa_percentile'].fillna(0.5)  # Default to median for NaN
            
            # Fatigue indicators
            if 'fatigue_index' in df.columns:
                df['fatigue_trend'] = df.groupby('ad_id')['fatigue_index'].diff()
                df['fatigue_acceleration'] = df.groupby('ad_id')['fatigue_trend'].diff()
            
            # Clean infinity and NaN values
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
                df[col] = df[col].clip(-1e6, 1e6)
            
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
        """Prepare training data for ML models with feature selection."""
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
            
            # Remove non-numeric columns and ensure feature names are strings
            numeric_cols = df_features[feature_cols].select_dtypes(include=[np.number]).columns
            feature_cols = [str(col) for col in feature_cols if col in numeric_cols]
            
            # Handle missing values and infinity
            df_features[feature_cols] = df_features[feature_cols].fillna(0)
            
            # Clean infinity and extreme values
            for col in feature_cols:
                if col in df_features.columns:
                    # Replace infinity with large but finite values
                    df_features[col] = df_features[col].replace([np.inf, -np.inf], np.nan)
                    df_features[col] = df_features[col].fillna(0)
                    
                    # Clip extreme values to prevent overflow
                    df_features[col] = df_features[col].clip(-1e6, 1e6)
            
            # Prepare X and y
            # Ensure DataFrame has string column names
            df_features.columns = [str(col) for col in df_features.columns]
            feature_cols = [str(col) for col in feature_cols]
            
            X = df_features[feature_cols].values
            y = df_features[target_col].fillna(0).values
            
            # Final data validation
            if np.any(np.isinf(X)) or np.any(np.isnan(X)):
                self.logger.warning("Found infinity or NaN in features, cleaning...")
                X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
            
            if np.any(np.isinf(y)) or np.any(np.isnan(y)):
                self.logger.warning("Found infinity or NaN in target, cleaning...")
                y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Check if we have any valid features
            if len(feature_cols) == 0 or X.shape[1] == 0:
                self.logger.warning("No features available for training")
                return np.array([]), np.array([]), []
            
            # Feature selection (if too many features)
            if len(feature_cols) > self.config.max_features:
                try:
                    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
                    
                    # Use mutual information for feature selection
                    selector = SelectKBest(mutual_info_regression, k=min(self.config.max_features, len(feature_cols)))
                    X_selected = selector.fit_transform(X, y)
                    
                    # Get selected feature names
                    selected_indices = selector.get_support(indices=True)
                    feature_cols = [feature_cols[i] for i in selected_indices]
                    X = X_selected
                    
                    self.logger.info(f"Feature selection: {len(selected_indices)}/{len(numeric_cols)} features selected")
                except Exception as e:
                    self.logger.warning(f"Feature selection failed: {e}, using all features")
            
            return X, y, feature_cols
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([]), []
    
    def train_model(self, model_type: str, stage: str, target_col: str) -> bool:
        """Train XGBoost model for specific prediction task."""
        try:
            # Get training data - try stage-specific first, then all stages if insufficient
            df = self.supabase.get_performance_data(stages=[stage])
            
            if df.empty or len(df) < 5:  # Need at least 5 samples for meaningful training
                self.logger.info(f"Insufficient stage-specific data for {stage} ({len(df) if not df.empty else 0} samples), trying all stages")
                # Fallback to all available data for training
                df = self.supabase.get_performance_data()
                
                if df.empty:
                    self.logger.warning(f"No data available for training {model_type} model for {stage}")
                    return False
                elif len(df) < 5:
                    self.logger.warning(f"Insufficient data for training {model_type} model: {len(df)} samples (need at least 5)")
                    return False
                else:
                    self.logger.info(f"Using cross-stage data for {stage} training: {len(df)} samples")
            
            # Prepare data
            X, y, feature_cols = self.prepare_training_data(df, target_col)
            
            if len(X) == 0:
                self.logger.warning(f"No features available for training {model_type} model")
                return False
            
            # Check if we have enough data for training
            if len(X) < 2:
                self.logger.warning(f"Insufficient data for training: {len(X)} samples (need at least 2)")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train ensemble of models for better predictions
            models_ensemble = {}
            
            # Primary model: XGBoost or GradientBoosting
            if XGBOOST_AVAILABLE:
                primary_model = xgb.XGBRegressor(**self.config.xgb_params)
            else:
                primary_model = GradientBoostingRegressor(
                    n_estimators=self.config.xgb_params.get('n_estimators', 100),
                    max_depth=self.config.xgb_params.get('max_depth', 6),
                    learning_rate=self.config.xgb_params.get('learning_rate', 0.1),
                    random_state=42
                )
            primary_model.fit(X_train_scaled, y_train)
            models_ensemble['primary'] = primary_model
            
            # Ensemble model 1: Random Forest
            try:
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train_scaled, y_train)
                models_ensemble['rf'] = rf_model
            except Exception as e:
                pass
            
            # Ensemble model 2: Gradient Boosting (if not primary)
            if XGBOOST_AVAILABLE:
                try:
                    gb_model = GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        random_state=42
                    )
                    gb_model.fit(X_train_scaled, y_train)
                    models_ensemble['gb'] = gb_model
                except Exception:
                    pass
            
            # Ensemble model 3: Ridge Regression (for baseline)
            try:
                lr_model = Ridge(alpha=1.0, random_state=42)
                lr_model.fit(X_train_scaled, y_train)
                models_ensemble['lr'] = lr_model
            except Exception:
                pass
            
            # Evaluate ensemble
            ensemble_predictions = []
            for name, mdl in models_ensemble.items():
                y_pred_mdl = mdl.predict(X_test_scaled)
                ensemble_predictions.append(y_pred_mdl)
            
            # Average ensemble predictions
            y_pred_ensemble = np.mean(ensemble_predictions, axis=0)
            mae = mean_absolute_error(y_test, y_pred_ensemble)
            r2 = r2_score(y_test, y_pred_ensemble)
            
            # Cross-validation score
            cv_scores = cross_val_score(primary_model, X_train_scaled, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            
            self.logger.info(f"Trained {model_type} ensemble for {stage}: MAE={mae:.4f}, R²={r2:.4f}, CV={cv_mean:.4f}")
            
            # Store all models and scaler
            model_key = f"{model_type}_{stage}"
            self.models[model_key] = primary_model
            for name, mdl in models_ensemble.items():
                if name != 'primary':
                    self.models[f"{model_key}_{name}"] = mdl
            self.scalers[model_key] = scaler
            
            # Store feature importance with CV score
            feature_importance = dict(zip(feature_cols, primary_model.feature_importances_))
            self.feature_importance[model_key] = feature_importance
            self.feature_importance[f"{model_key}_confidence"] = {
                'cv_score': float(cv_mean),
                'test_r2': float(r2),
                'test_mae': float(mae),
                'ensemble_size': len(models_ensemble)
            }
            
            # Save to Supabase (FIX: use primary_model instead of undefined 'model')
            self.save_model_to_supabase(model_type, stage, primary_model, scaler, feature_cols, feature_importance)
            
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
            
            # Prepare features - ensure feature names are strings and match
            try:
                # Get model's expected feature names (ensure they're strings)
                expected_features = [str(col) for col in model.feature_names_in_]
                
                # Create feature vector matching model's expected features
                feature_vector = np.array([
                    float(features.get(str(col), 0)) for col in expected_features
                ])
                feature_vector_scaled = scaler.transform([feature_vector])
                
            except AttributeError:
                # Fallback: if model doesn't have feature_names_in_, use features as-is
                feature_names = list(features.keys())
                feature_vector = np.array([float(features[col]) for col in feature_names])
                feature_vector_scaled = scaler.transform([feature_vector])
            
            # Make prediction with ensemble if available
            predictions_ensemble = []
            predictions_ensemble.append(model.predict(feature_vector_scaled)[0])
            
            # If we have multiple models, use ensemble
            model_key_base = f"{model_type}_{stage}"
            for ensemble_suffix in ['_rf', '_gb', '_lr']:
                ensemble_key = model_key_base + ensemble_suffix
                if ensemble_key in self.models:
                    ensemble_pred = self.models[ensemble_key].predict(feature_vector_scaled)[0]
                    predictions_ensemble.append(ensemble_pred)
            
            # Final prediction (mean of ensemble)
            prediction = np.mean(predictions_ensemble)
            prediction_std = np.std(predictions_ensemble) if len(predictions_ensemble) > 1 else 0
            
            # Calculate confidence (improved with ensemble variance)
            if len(predictions_ensemble) > 1:
                # Confidence based on ensemble agreement
                ensemble_agreement = 1.0 - (prediction_std / (abs(prediction) + 1e-6))
                confidence = min(0.95, max(0.1, ensemble_agreement))
            else:
                # Fallback: use cross-validation score from training
                confidence_key = f"{model_key}_confidence"
                confidence = self.feature_importance.get(confidence_key, {}).get('cv_score', 0.5)
                confidence = min(0.95, max(0.1, confidence))
            
            # Prediction intervals (proper bootstrap-based)
            # Use ensemble std or estimate from model variance
            if prediction_std > 0:
                interval_lower = prediction - 1.96 * prediction_std
                interval_upper = prediction + 1.96 * prediction_std
            else:
                # Fallback: use percentage of prediction value
                interval_width = abs(prediction) * 0.2  # ±20%
                interval_lower = prediction - interval_width
                interval_upper = prediction + interval_width
            
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
            
            # Create model metadata (ensure JSON serializable) - FIX: sanitize inf/nan
            def sanitize_float(val):
                """Convert to JSON-safe float."""
                try:
                    f = float(val)
                    if np.isnan(f) or np.isinf(f):
                        return 0.0
                    return f
                except:
                    return 0.0
            
            metadata = {
                'feature_columns': feature_cols,
                'feature_importance': {k: sanitize_float(v) for k, v in feature_importance.items()},
                'training_date': now_utc().isoformat(),
                'model_type': model_type,
                'stage': stage
            }
            
            # Performance metrics (comprehensive) - ensure JSON serializable
            model_key = f"{model_type}_{stage}"
            confidence_data = self.feature_importance.get(f"{model_key}_confidence", {})
            
            performance_metrics = {
                'feature_count': int(len(feature_cols)),
                'model_size_bytes': int(len(model_data)),
                'cv_score': sanitize_float(confidence_data.get('cv_score', 0)),
                'test_r2': sanitize_float(confidence_data.get('test_r2', 0)),
                'test_mae': sanitize_float(confidence_data.get('test_mae', 0)),
                'ensemble_size': int(confidence_data.get('ensemble_size', 1))
            }
            
            data = {
                'model_type': model_type,
                'stage': stage,
                'version': 1,  # Add version for upsert
                'model_name': f"{model_type}_{stage}_v1",
                'model_data': model_data.hex(),  # Convert to hex for storage
                'model_metadata': metadata,
                'features_used': feature_cols,
                'performance_metrics': performance_metrics,
                'is_active': True,
                'trained_at': now_utc().isoformat()
            }
            
            # FIX: Use upsert instead of insert to handle duplicates
            # First try to update existing record, then insert if not found
            try:
                # Try to update existing record
                update_response = self.supabase.client.table('ml_models').update(data).eq(
                    'model_type', model_type
                ).eq('stage', stage).eq('version', 1).execute()
                
                if update_response.data:
                    response = update_response
                else:
                    # Insert new record if no existing one found
                    response = self.supabase.client.table('ml_models').insert(data).execute()
            except Exception as e:
                # If insert fails due to duplicate, try update again
                if "duplicate key" in str(e).lower() or "unique constraint" in str(e).lower():
                    try:
                        response = self.supabase.client.table('ml_models').update(data).eq(
                            'model_type', model_type
                        ).eq('stage', stage).eq('version', 1).execute()
                    except Exception:
                        # If all else fails, just log and continue
                        self.logger.warning(f"Could not save model {model_type}_{stage}, continuing...")
                        return True  # Return True to avoid breaking the flow
                else:
                    raise e
            
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
                return False
            
            model_data = response.data[0]
            
            # Skip corrupted models entirely - force retrain
            if not model_data.get('model_data'):
                return False
                
            # Check if model is too old (older than 7 days)
            trained_at = model_data.get('trained_at')
            if trained_at:
                try:
                    from datetime import datetime, timezone
                    trained_date = datetime.fromisoformat(trained_at.replace('Z', '+00:00'))
                    if (datetime.now(timezone.utc) - trained_date).days > 7:
                        return False
                except:
                    return False
            
            # Deserialize and load the model
            model_data_hex = model_data.get('model_data', '')
            if not model_data_hex:
                logger.error(f"No model data found for {model_type}_{stage}")
                return False
            
            try:
                model_bytes = bytes.fromhex(model_data_hex)
                model = pickle.loads(model_bytes)
                model_key = f"{model_type}_{stage}"
                self.models[model_key] = model
            except ValueError as e:
                logger.error(f"Invalid hex data for {model_type}_{stage}: {e}")
                return False
            except Exception as e:
                logger.error(f"Failed to deserialize model {model_type}_{stage}: {e}")
                return False
            
            # Load scaler if available
            scaler_data = model_data.get('scaler_data')
            if scaler_data:
                scaler_bytes = bytes.fromhex(scaler_data)
                scaler = pickle.loads(scaler_bytes)
                self.scalers[model_key] = scaler
            else:
                # Create a default scaler
                from sklearn.preprocessing import StandardScaler
                self.scalers[model_key] = StandardScaler()
            
            self.logger.info(f"Successfully loaded {model_type} model for {stage} from Supabase")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model from Supabase: {e}")
            return False

class TemporalAnalyzer:
    """Advanced temporal analysis and trend detection with historical data integration."""
    
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
    
    def analyze_comprehensive_trends(self, ad_id: str, days_back: int = 14) -> Dict[str, Any]:
        """Analyze comprehensive performance trends using historical data."""
        try:
            # Get historical metrics
            metric_names = ['cpm', 'ctr', 'spend', 'impressions', 'clicks', 'purchases']
            df = self.supabase.get_historical_metrics(ad_id, metric_names, days_back)
            
            if df.empty:
                return {'status': 'insufficient_data'}
            
            # Calculate trend analysis for each metric
            trends = {}
            
            for metric in metric_names:
                if metric in df.columns:
                    values = df[metric].values
                    if len(values) >= 3:
                        # Linear trend analysis
                        x = np.arange(len(values))
                        slope, _, r_value, p_value, _ = np.polyfit(x, values, 1)
                        
                        # Trend classification
                        if p_value < 0.05:  # Significant trend
                            if slope > 0:
                                trend_direction = 'improving'
                            else:
                                trend_direction = 'declining'
                        else:
                            trend_direction = 'stable'
                        
                        # Volatility analysis
                        volatility = np.std(values) / (np.mean(values) + 1e-6)
                        
                        # Recent vs older performance
                        if len(values) >= 6:
                            recent_avg = np.mean(values[-3:])
                            older_avg = np.mean(values[:-3])
                            momentum = (recent_avg - older_avg) / (older_avg + 1e-6)
                        else:
                            momentum = 0
                        
                        trends[metric] = {
                            'trend_direction': trend_direction,
                            'slope': float(slope),
                            'r_squared': float(r_value ** 2),
                            'volatility': float(volatility),
                            'momentum': float(momentum),
                            'current_value': float(values[-1]) if len(values) > 0 else 0,
                            'avg_value': float(np.mean(values))
                        }
            
            return {
                'status': 'success',
                'trends': trends,
                'data_points': len(df),
                'analysis_period_days': days_back
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing comprehensive trends: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def detect_performance_anomalies(self, ad_id: str, days_back: int = 7) -> Dict[str, Any]:
        """Detect performance anomalies using historical data."""
        try:
            # Get historical metrics
            metric_names = ['cpm', 'ctr', 'spend', 'impressions', 'clicks']
            df = self.supabase.get_historical_metrics(ad_id, metric_names, days_back)
            
            if df.empty or len(df) < 3:
                return {'status': 'insufficient_data'}
            
            anomalies = {}
            
            for metric in metric_names:
                if metric in df.columns:
                    values = df[metric].values
                    
                    # Statistical anomaly detection
                    mean_val = np.mean(values[:-1])  # Exclude latest value
                    std_val = np.std(values[:-1])
                    
                    if std_val > 0:
                        current_val = values[-1]
                        z_score = abs(current_val - mean_val) / std_val
                        
                        # Anomaly threshold (2 standard deviations)
                        if z_score > 2.0:
                            anomaly_type = 'spike' if current_val > mean_val else 'drop'
                            severity = 'high' if z_score > 3.0 else 'medium'
                            
                            anomalies[metric] = {
                                'type': anomaly_type,
                                'severity': severity,
                                'z_score': float(z_score),
                                'current_value': float(current_val),
                                'expected_value': float(mean_val),
                                'deviation_pct': float(abs(current_val - mean_val) / (mean_val + 1e-6) * 100)
                            }
            
            return {
                'status': 'success',
                'anomalies': anomalies,
                'analysis_period_days': days_back
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting performance anomalies: {e}")
            return {'status': 'error', 'error': str(e)}

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
    
    def initialize_models(self, force_retrain: bool = False) -> bool:
        """Initialize all ML models with intelligent caching."""
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
            cached_count = 0
            
            for model_type, stage, target in models_to_train:
                
                # Check if model exists and is recent (< 24h old)
                if not force_retrain and self._should_use_cached_model(model_type, stage):
                    if self.predictor.load_model_from_supabase(model_type, stage):
                        cached_count += 1
                        success_count += 1
                        self.logger.info(f"✅ Loaded cached {model_type} for {stage}")
                        continue
                    else:
                        # Cached model failed to load, train new one
                        trained = self.predictor.train_model(model_type, stage, target)
                        if trained:
                            success_count += 1
                            self.logger.info(f"🔄 Trained new {model_type} for {stage}")
                        else:
                            self.logger.warning(f"⚠️ Failed to train {model_type} for {stage}")
                else:
                    # No cached model or force retrain, train new one
                    trained = self.predictor.train_model(model_type, stage, target)
                    if trained:
                        success_count += 1
                        self.logger.info(f"🔄 Trained new {model_type} for {stage}")
                    else:
                        self.logger.warning(f"⚠️ Failed to train {model_type} for {stage}")

            self.logger.info(f"Initialized {success_count}/{len(models_to_train)} models ({cached_count} from cache)")
            return success_count > 0

        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
            return False
            return False
    
    def _should_use_cached_model(self, model_type: str, stage: str) -> bool:
        """Check if cached model is still valid."""
        try:
            response = self.supabase.client.table('ml_models').select('trained_at').eq(
                'model_type', model_type
            ).eq('stage', stage).eq('is_active', True).order('trained_at', desc=True).limit(1).execute()
            
            if not response.data:
                return False
            
            trained_at = response.data[0].get('trained_at')
            if not trained_at:
                return False
            
            # Check if model is less than retrain_frequency_hours old
            try:
                # Handle different date formats
                if 'T' in trained_at and '+' in trained_at:
                    trained_time = datetime.fromisoformat(trained_at)
                elif 'T' in trained_at and 'Z' in trained_at:
                    trained_time = datetime.fromisoformat(trained_at.replace('Z', '+00:00'))
                else:
                    trained_time = datetime.fromisoformat(trained_at)
                
                age_hours = (now_utc() - trained_time).total_seconds() / 3600
                return age_hours < self.config.retrain_frequency_hours
            except:
                return False
            
        except Exception as e:
            self.logger.error(f"Error checking cached model: {e}")
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
            
            # Performance predictions - apply same feature engineering as training
            try:
                # Create a DataFrame with the current data for feature engineering
                prediction_df = pd.DataFrame([latest])
                
                # Apply the same feature engineering as training
                df_features = self.predictor.feature_engineer.create_rolling_features(
                    prediction_df, ['ad_id'], ['ctr', 'cpa', 'roas', 'spend', 'purchases']
                )
                df_features = self.predictor.feature_engineer.create_interaction_features(df_features)
                df_features = self.predictor.feature_engineer.create_temporal_features(df_features)
                df_features = self.predictor.feature_engineer.create_advanced_features(df_features)
                
                # Select numeric features (same as training)
                feature_cols = [col for col in df_features.columns 
                               if col not in ['ad_id', 'id', 'date_start', 'date_end', 'created_at']]
                numeric_cols = df_features[feature_cols].select_dtypes(include=[np.number]).columns
                feature_cols = [str(col) for col in feature_cols if col in numeric_cols]
                
                # Create features dictionary with all engineered features
                features = {}
                for col in feature_cols:
                    if col in df_features.columns:
                        val = df_features[col].iloc[0] if not df_features[col].empty else 0
                        features[col] = float(val) if pd.notna(val) else 0.0
                
                predictions = self.predict_performance(ad_id, stage, features)
                
            except Exception as e:
                self.logger.warning(f"Feature engineering failed for prediction, using basic features: {e}")
                # Fallback to basic features
                features = {
                    'ctr': float(latest.get('ctr', 0)),
                    'cpa': float(latest.get('cpa', 0)),
                    'roas': float(latest.get('roas', 0)),
                    'spend': float(latest.get('spend', 0)),
                    'purchases': float(latest.get('purchases', 0)),
                    'performance_quality_score': float(latest.get('performance_quality_score', 0)),
                    'stability_score': float(latest.get('stability_score', 0)),
                    'fatigue_index': float(latest.get('fatigue_index', 0)),
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
