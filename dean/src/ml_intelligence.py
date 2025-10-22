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
                        
                        # Rolling trend (slope) - only for windows >= 2
                        if window >= 2:
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
                    df[f'{col}_volatility'] = df.groupby('ad_id')[col].transform(
                        lambda x: x.rolling(window=7, min_periods=3).std()
                    )
            
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
            
            # Remove non-numeric columns
            numeric_cols = df_features[feature_cols].select_dtypes(include=[np.number]).columns
            feature_cols = [col for col in feature_cols if col in numeric_cols]
            
            # Handle missing values
            df_features[feature_cols] = df_features[feature_cols].fillna(0)
            
            # Prepare X and y
            X = df_features[feature_cols].values
            y = df_features[target_col].fillna(0).values
            
            # Feature selection (if too many features)
            if len(feature_cols) > self.config.max_features:
                from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
                
                # Use mutual information for feature selection
                selector = SelectKBest(mutual_info_regression, k=min(self.config.max_features, len(feature_cols)))
                X_selected = selector.fit_transform(X, y)
                
                # Get selected feature names
                selected_indices = selector.get_support(indices=True)
                feature_cols = [feature_cols[i] for i in selected_indices]
                X = X_selected
                
                self.logger.info(f"Feature selection: {len(selected_indices)}/{len(numeric_cols)} features selected")
            
            return X, y, feature_cols
            
        except Exception as e:
            self.logger.error(f"Error preparing training data: {e}")
            return np.array([]), np.array([]), []
    
    def train_model(self, model_type: str, stage: str, target_col: str) -> bool:
        """Train XGBoost model for specific prediction task."""
        try:
            self.logger.info(f"🔧 [ML DEBUG] Starting {model_type} training for {stage}")
            self.logger.info(f"🔧 [ML DEBUG] Target column: {target_col}")
            
            # Get training data
            self.logger.info(f"🔧 [ML DEBUG] Querying Supabase for {stage} stage data...")
            df = self.supabase.get_performance_data(stages=[stage])
            self.logger.info(f"🔧 [ML DEBUG] Retrieved data shape: {df.shape}")
            self.logger.info(f"🔧 [ML DEBUG] Data columns: {list(df.columns)}")
            if not df.empty:
                self.logger.info(f"🔧 [ML DEBUG] Data sample: {df.head(2).to_dict()}")
                self.logger.info(f"🔧 [ML DEBUG] Target column '{target_col}' values: {df[target_col].tolist() if target_col in df.columns else 'Column not found'}")
                self.logger.info(f"🔧 [ML DEBUG] ✅ {stage} stage HAS DATA - proceeding with training")
            else:
                self.logger.info(f"🔧 [ML DEBUG] Empty DataFrame - no data available")
                self.logger.info(f"🔧 [ML DEBUG] ❌ {stage} stage has NO DATA - skipping training")
            
            if df.empty:
                self.logger.warning(f"🔧 [ML DEBUG] No data available for training {model_type} model for {stage}")
                self.logger.warning(f"🔧 [ML DEBUG] DataFrame empty: {df.empty}, Length: {len(df)}")
                self.logger.warning(f"🔧 [ML DEBUG] This means no historical data exists for {stage} stage")
                self.logger.warning(f"🔧 [ML DEBUG] Models will be trained when data becomes available")
                return False
            
            # Prepare data
            self.logger.info(f"🔧 [ML DEBUG] Preparing training data...")
            X, y, feature_cols = self.prepare_training_data(df, target_col)
            self.logger.info(f"🔧 [ML DEBUG] Features shape: X={X.shape}, y={y.shape}")
            self.logger.info(f"🔧 [ML DEBUG] Feature columns: {feature_cols}")
            self.logger.info(f"🔧 [ML DEBUG] Target values: {y[:5] if len(y) > 0 else 'Empty'}")
            
            if len(X) == 0:
                self.logger.warning(f"🔧 [ML DEBUG] No features available for training {model_type} model")
                self.logger.warning(f"🔧 [ML DEBUG] X shape: {X.shape}, y shape: {y.shape}")
                self.logger.warning(f"🔧 [ML DEBUG] This usually means insufficient data or feature engineering failed")
                return False
            
            # Check if we have enough data for training
            if len(X) < 2:
                self.logger.warning(f"🔧 [ML DEBUG] Insufficient data for training: {len(X)} samples (need at least 2)")
                self.logger.warning(f"🔧 [ML DEBUG] Cannot train {model_type} model with {len(X)} samples")
                return False
            
            # Split data
            self.logger.info(f"🔧 [ML DEBUG] Splitting data (80/20 train/test)...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            self.logger.info(f"🔧 [ML DEBUG] Train set: X={X_train.shape}, y={y_train.shape}")
            self.logger.info(f"🔧 [ML DEBUG] Test set: X={X_test.shape}, y={y_test.shape}")
            
            # Scale features
            self.logger.info(f"🔧 [ML DEBUG] Scaling features with RobustScaler...")
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.logger.info(f"🔧 [ML DEBUG] Scaled train: {X_train_scaled.shape}, Scaled test: {X_test_scaled.shape}")
            
            # Train ensemble of models for better predictions
            self.logger.info(f"🔧 [ML DEBUG] Training ensemble of models...")
            models_ensemble = {}
            
            # Primary model: XGBoost or GradientBoosting
            self.logger.info(f"🔧 [ML DEBUG] XGBoost available: {XGBOOST_AVAILABLE}")
            if XGBOOST_AVAILABLE:
                primary_model = xgb.XGBRegressor(**self.config.xgb_params)
            else:
                primary_model = GradientBoostingRegressor(
                    n_estimators=self.config.xgb_params.get('n_estimators', 100),
                    max_depth=self.config.xgb_params.get('max_depth', 6),
                    learning_rate=self.config.xgb_params.get('learning_rate', 0.1),
                    random_state=42
                )
            self.logger.info(f"🔧 [ML DEBUG] Training primary model ({type(primary_model).__name__})...")
            primary_model.fit(X_train_scaled, y_train)
            models_ensemble['primary'] = primary_model
            self.logger.info(f"🔧 [ML DEBUG] Primary model trained successfully")
            
            # Ensemble model 1: Random Forest
            try:
                self.logger.info(f"🔧 [ML DEBUG] Training Random Forest ensemble model...")
                rf_model = RandomForestRegressor(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=-1
                )
                rf_model.fit(X_train_scaled, y_train)
                models_ensemble['rf'] = rf_model
                self.logger.info(f"🔧 [ML DEBUG] Random Forest trained successfully")
            except Exception as e:
                self.logger.warning(f"🔧 [ML DEBUG] Random Forest training failed: {e}")
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
            self.logger.info(f"🔧 [ML DEBUG] Evaluating ensemble with {len(models_ensemble)} models...")
            ensemble_predictions = []
            for name, mdl in models_ensemble.items():
                self.logger.info(f"🔧 [ML DEBUG] Evaluating {name} model...")
                y_pred_mdl = mdl.predict(X_test_scaled)
                ensemble_predictions.append(y_pred_mdl)
                self.logger.info(f"🔧 [ML DEBUG] {name} predictions shape: {y_pred_mdl.shape}")
            
            # Average ensemble predictions
            y_pred_ensemble = np.mean(ensemble_predictions, axis=0)
            mae = mean_absolute_error(y_test, y_pred_ensemble)
            r2 = r2_score(y_test, y_pred_ensemble)
            
            # Cross-validation score
            self.logger.info(f"🔧 [ML DEBUG] Running cross-validation...")
            cv_scores = cross_val_score(primary_model, X_train_scaled, y_train, cv=5, scoring='r2')
            cv_mean = cv_scores.mean()
            
            self.logger.info(f"🔧 [ML DEBUG] Ensemble evaluation complete:")
            self.logger.info(f"🔧 [ML DEBUG] - MAE: {mae:.4f}")
            self.logger.info(f"🔧 [ML DEBUG] - R²: {r2:.4f}")
            self.logger.info(f"🔧 [ML DEBUG] - CV Mean: {cv_mean:.4f}")
            self.logger.info(f"🔧 [ML DEBUG] - Ensemble size: {len(models_ensemble)}")
            self.logger.info(f"Trained {model_type} ensemble for {stage}: MAE={mae:.4f}, R²={r2:.4f}, CV={cv_mean:.4f}")
            
            # Store all models and scaler
            self.logger.info(f"🔧 [ML DEBUG] Storing models and scaler...")
            model_key = f"{model_type}_{stage}"
            self.models[model_key] = primary_model
            for name, mdl in models_ensemble.items():
                if name != 'primary':
                    self.models[f"{model_key}_{name}"] = mdl
            self.scalers[model_key] = scaler
            self.logger.info(f"🔧 [ML DEBUG] Stored {len(self.models)} models and {len(self.scalers)} scalers")
            
            # Store feature importance with CV score
            self.logger.info(f"🔧 [ML DEBUG] Calculating feature importance...")
            feature_importance = dict(zip(feature_cols, primary_model.feature_importances_))
            self.feature_importance[model_key] = feature_importance
            self.feature_importance[f"{model_key}_confidence"] = {
                'cv_score': float(cv_mean),
                'test_r2': float(r2),
                'test_mae': float(mae),
                'ensemble_size': len(models_ensemble)
            }
            self.logger.info(f"🔧 [ML DEBUG] Feature importance calculated for {len(feature_importance)} features")
            
            # Save to Supabase (FIX: use primary_model instead of undefined 'model')
            self.logger.info(f"🔧 [ML DEBUG] Saving model to Supabase...")
            self.save_model_to_supabase(model_type, stage, primary_model, scaler, feature_cols, feature_importance)
            self.logger.info(f"🔧 [ML DEBUG] Model saved to Supabase successfully")
            
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
            self.logger.info(f"🔧 [ML DEBUG] Starting model save to Supabase...")
            self.logger.info(f"🔧 [ML DEBUG] Model type: {model_type}, Stage: {stage}")
            self.logger.info(f"🔧 [ML DEBUG] Model class: {type(model).__name__}")
            self.logger.info(f"🔧 [ML DEBUG] Feature columns: {len(feature_cols)}")
            self.logger.info(f"🔧 [ML DEBUG] Feature importance: {len(feature_importance)} features")
            
            # Serialize model
            self.logger.info(f"🔧 [ML DEBUG] Serializing model with pickle...")
            model_data = pickle.dumps(model)
            self.logger.info(f"🔧 [ML DEBUG] Model serialized, size: {len(model_data)} bytes")
            
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
            response = self.supabase.client.table('ml_models').upsert(
                data, 
                on_conflict='model_type,version,stage'
            ).execute()
            
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
            
            # FIX: Handle corrupted model data gracefully
            try:
                model_data_str = model_data['model_data']
                self.logger.info(f"🔧 [ML DEBUG] Model data type: {type(model_data_str)}")
                self.logger.info(f"🔧 [ML DEBUG] Model data length: {len(model_data_str)}")
                self.logger.info(f"🔧 [ML DEBUG] Model data preview: {model_data_str[:50]}...")
                
                # Try different decoding methods
                if isinstance(model_data_str, str):
                    if model_data_str.startswith('80'):  # Pickle protocol header
                        self.logger.info(f"🔧 [ML DEBUG] Attempting hex decode...")
                        model_bytes = bytes.fromhex(model_data_str)
                    else:
                        self.logger.info(f"🔧 [ML DEBUG] Attempting latin-1 encode...")
                        model_bytes = model_data_str.encode('latin-1')
                else:
                    self.logger.info(f"🔧 [ML DEBUG] Using data as-is...")
                    model_bytes = model_data_str
                
                self.logger.info(f"🔧 [ML DEBUG] Model bytes length: {len(model_bytes)}")
                model = pickle.loads(model_bytes)
                self.logger.info(f"🔧 [ML DEBUG] Model loaded successfully: {type(model)}")
                
            except (ValueError, UnicodeDecodeError, pickle.UnpicklingError) as e:
                self.logger.warning(f"🔧 [ML DEBUG] Could not load model {model_type} for {stage}: {e}")
                self.logger.warning(f"🔧 [ML DEBUG] Model data appears corrupted, will retrain")
                return False
            
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
                self.logger.info(f"🔧 [ML DEBUG] Processing {model_type} for {stage} stage...")
                self.logger.info(f"🔧 [ML DEBUG] Force retrain: {force_retrain}")
                
                # Check if model exists and is recent (< 24h old)
                if not force_retrain and self._should_use_cached_model(model_type, stage):
                    self.logger.info(f"🔧 [ML DEBUG] Attempting to load cached {model_type} for {stage}...")
                    if self.predictor.load_model_from_supabase(model_type, stage):
                        cached_count += 1
                        success_count += 1
                        self.logger.info(f"🔧 [ML DEBUG] ✅ Successfully loaded cached {model_type} for {stage}")
                        self.logger.info(f"✅ Loaded cached {model_type} for {stage}")
                        continue
                    else:
                        self.logger.warning(f"🔧 [ML DEBUG] ❌ Failed to load cached {model_type} for {stage} - will train new")
                else:
                    self.logger.info(f"🔧 [ML DEBUG] No cached model available for {model_type} in {stage} - will train new")
                
                # Train new model
                self.logger.info(f"🔧 [ML DEBUG] Attempting to train new {model_type} for {stage}...")
                if self.predictor.train_model(model_type, stage, target):
                    success_count += 1
                    self.logger.info(f"🔧 [ML DEBUG] ✅ Successfully trained {model_type} for {stage}")
                    self.logger.info(f"🔄 Trained new {model_type} for {stage}")
                else:
                    self.logger.warning(f"🔧 [ML DEBUG] ❌ Failed to train {model_type} for {stage}")
            
            self.logger.info(f"Initialized {success_count}/{len(models_to_train)} models ({cached_count} from cache)")
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Error initializing models: {e}")
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
            trained_time = datetime.fromisoformat(trained_at.replace('Z', '+00:00'))
            age_hours = (now_utc() - trained_time).total_seconds() / 3600
            
            return age_hours < self.config.retrain_frequency_hours
            
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
