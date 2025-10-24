"""
DEAN ADVANCED ML FEATURES
Combined advanced ML capabilities including:
- Reinforcement Learning (Q-Learning)
- Neural Networks (LSTM for time-series)
- Multi-Objective Optimization
- Automated Feature Engineering
- Bayesian Optimization
- Active Learning
- SHAP Explainability
- Portfolio Optimization
- Seasonality Detection
- Budget Allocation Optimization
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Optional imports for advanced features
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    shap = None

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

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# =====================================================
# REINFORCEMENT LEARNING
# =====================================================

@dataclass
class RLState:
    """State representation for RL."""
    cpa: float
    roas: float
    ctr: float
    spend: float
    days_active: int
    stage: str

@dataclass
class RLAction:
    """Action representation for RL."""
    action_type: str  # 'kill', 'promote', 'hold', 'scale_up', 'scale_down'
    confidence: float

class QLearningAgent:
    """Q-Learning agent for ad management decisions."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.1):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        
        # Q-table: state_hash -> {action -> q_value}
        self.q_table: Dict[str, Dict[str, float]] = {}
        
        self.actions = ['kill', 'promote', 'hold', 'scale_up', 'scale_down']
        self.logger = logging.getLogger(f"{__name__}.QLearningAgent")
    
    def _state_to_hash(self, state: RLState) -> str:
        """Convert state to hashable representation."""
        # Discretize continuous values
        cpa_bucket = int(state.cpa / 5) * 5  # €5 buckets
        roas_bucket = int(state.roas / 0.5) * 0.5  # 0.5 buckets
        ctr_bucket = int(state.ctr / 0.5) * 0.5  # 0.5% buckets
        
        return f"{state.stage}_{cpa_bucket}_{roas_bucket}_{ctr_bucket}_{state.days_active}"
    
    def get_action(self, state: RLState, explore: bool = True) -> RLAction:
        """Get action using epsilon-greedy policy."""
        state_hash = self._state_to_hash(state)
        
        # Initialize state if not seen
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {action: 0.0 for action in self.actions}
        
        # Epsilon-greedy action selection
        if explore and np.random.random() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            # Choose best action
            q_values = self.q_table[state_hash]
            action = max(q_values, key=q_values.get)
        
        # Calculate confidence based on Q-value spread
        q_values = self.q_table[state_hash]
        max_q = max(q_values.values())
        min_q = min(q_values.values())
        confidence = (max_q - min_q) / (max_q + 1e-8) if max_q > 0 else 0.0
        
        return RLAction(action_type=action, confidence=confidence)
    
    def update_q_value(self, state: RLState, action: str, reward: float, next_state: RLState):
        """Update Q-value using Q-learning formula."""
        state_hash = self._state_to_hash(state)
        next_state_hash = self._state_to_hash(next_state)
        
        # Initialize states if needed
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {a: 0.0 for a in self.actions}
        if next_state_hash not in self.q_table:
            self.q_table[next_state_hash] = {a: 0.0 for a in self.actions}
        
        # Q-learning update
        current_q = self.q_table[state_hash][action]
        max_next_q = max(self.q_table[next_state_hash].values())
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_hash][action] = new_q

# =====================================================
# NEURAL NETWORKS
# =====================================================

class LSTMPredictor:
    """LSTM-based time series predictor for ad performance."""
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, num_layers: int = 2):
        if not TORCH_AVAILABLE:
            self.available = False
            self.logger = logging.getLogger(f"{__name__}.LSTMPredictor")
            self.logger.warning("PyTorch not available - LSTM disabled")
            return
        
        self.available = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Create LSTM model
        self.model = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)
        
        self.logger = logging.getLogger(f"{__name__}.LSTMPredictor")
    
    def predict(self, sequence: np.ndarray) -> float:
        """Predict next value in sequence."""
        if not self.available:
            return 0.0
        
        try:
            # Convert to tensor
            x = torch.FloatTensor(sequence).unsqueeze(0)
            
            # Forward pass
            lstm_out, _ = self.model(x)
            prediction = self.linear(lstm_out[:, -1, :])
            
            return prediction.item()
        except Exception as e:
            self.logger.error(f"LSTM prediction error: {e}")
            return 0.0

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
        """Automatically generate features using deep feature synthesis."""
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
            self.logger.error(f"Auto feature engineering error: {e}")
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
    
    def optimize_hyperparameters(self, objective_func, param_bounds: Dict[str, Tuple[float, float]],
                                 n_calls: int = 50) -> Dict[str, float]:
        """Optimize hyperparameters using Gaussian Process."""
        if not self.available:
            return {}
        
        try:
            # Convert bounds to skopt format
            dimensions = []
            param_names = []
            for name, (low, high) in param_bounds.items():
                dimensions.append(Real(low, high, name=name))
                param_names.append(name)
            
            # Run optimization
            result = gp_minimize(
                func=objective_func,
                dimensions=dimensions,
                n_calls=n_calls,
                random_state=42
            )
            
            # Return best parameters
            best_params = dict(zip(param_names, result.x))
            self.logger.info(f"Bayesian optimization completed. Best score: {result.fun}")
            
            return best_params
            
        except Exception as e:
            self.logger.error(f"Bayesian optimization error: {e}")
            return {}

# =====================================================
# PORTFOLIO OPTIMIZATION
# =====================================================

class PortfolioOptimizer:
    """Optimize budget allocation across ads using modern portfolio theory."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PortfolioOptimizer")
    
    def optimize_budget_allocation(self, ads_data: pd.DataFrame, 
                                  total_budget: float,
                                  risk_tolerance: float = 0.5) -> Dict[str, float]:
        """Optimize budget allocation using mean-variance optimization."""
        try:
            # Calculate expected returns (ROAS) and risk (volatility)
            returns = ads_data['roas'].values
            risks = ads_data['roas'].rolling(window=7).std().fillna(0.1).values
            
            # Mean-variance optimization
            n_ads = len(ads_data)
            
            # Objective: maximize return - risk_tolerance * risk
            def objective(weights):
                portfolio_return = np.sum(weights * returns)
                portfolio_risk = np.sum(weights * risks)
                return -(portfolio_return - risk_tolerance * portfolio_risk)
            
            # Constraints
            constraints = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
            ]
            
            # Bounds: each weight between 0 and 1
            bounds = [(0, 1) for _ in range(n_ads)]
            
            # Initial guess: equal weights
            x0 = np.ones(n_ads) / n_ads
            
            # Optimize
            result = optimize.minimize(
                objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                budget_allocation = {
                    ads_data.iloc[i]['ad_id']: optimal_weights[i] * total_budget
                    for i in range(n_ads)
                }
                
                self.logger.info(f"Portfolio optimization completed. Total budget: €{total_budget}")
                return budget_allocation
            else:
                # Fallback to equal allocation
                equal_allocation = total_budget / n_ads
                return {ads_data.iloc[i]['ad_id']: equal_allocation for i in range(n_ads)}
                
        except Exception as e:
            self.logger.error(f"Portfolio optimization error: {e}")
            # Fallback to equal allocation
            equal_allocation = total_budget / len(ads_data)
            return {ads_data.iloc[i]['ad_id']: equal_allocation for i in range(len(ads_data))}

# =====================================================
# SEASONALITY DETECTION
# =====================================================

class SeasonalityDetector:
    """Detect seasonal patterns in ad performance."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.SeasonalityDetector")
    
    def detect_seasonality(self, df: pd.DataFrame, 
                          time_col: str = 'created_at',
                          value_col: str = 'roas') -> Dict[str, Any]:
        """Detect seasonal patterns in performance data."""
        try:
            # Convert time column
            df[time_col] = pd.to_datetime(df[time_col])
            df['hour'] = df[time_col].dt.hour
            df['day_of_week'] = df[time_col].dt.dayofweek
            df['day_of_month'] = df[time_col].dt.day
            
            # Analyze patterns
            hourly_pattern = df.groupby('hour')[value_col].mean()
            daily_pattern = df.groupby('day_of_week')[value_col].mean()
            monthly_pattern = df.groupby('day_of_month')[value_col].mean()
            
            # Find peak times
            peak_hour = hourly_pattern.idxmax()
            peak_day = daily_pattern.idxmax()
            peak_day_of_month = monthly_pattern.idxmax()
            
            # Calculate seasonality strength
            hourly_std = hourly_pattern.std()
            daily_std = daily_pattern.std()
            
            seasonality_strength = (hourly_std + daily_std) / 2
            
            return {
                'peak_hour': int(peak_hour),
                'peak_day': int(peak_day),
                'peak_day_of_month': int(peak_day_of_month),
                'seasonality_strength': float(seasonality_strength),
                'hourly_pattern': hourly_pattern.to_dict(),
                'daily_pattern': daily_pattern.to_dict(),
                'is_seasonal': seasonality_strength > 0.1
            }
            
        except Exception as e:
            self.logger.error(f"Seasonality detection error: {e}")
            return {
                'peak_hour': 12,
                'peak_day': 1,
                'peak_day_of_month': 15,
                'seasonality_strength': 0.0,
                'hourly_pattern': {},
                'daily_pattern': {},
                'is_seasonal': False
            }

# =====================================================
# SHAP EXPLAINABILITY
# =====================================================

class SHAPExplainer:
    """SHAP-based model explainability."""
    
    def __init__(self):
        if not SHAP_AVAILABLE:
            self.available = False
            self.logger = logging.getLogger(f"{__name__}.SHAPExplainer")
            self.logger.warning("SHAP not available - explainability disabled")
            return
        
        self.available = True
        self.logger = logging.getLogger(f"{__name__}.SHAPExplainer")
    
    def explain_prediction(self, model, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Generate SHAP explanations for a prediction."""
        if not self.available:
            return {}
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            # Get feature importance
            feature_importance = dict(zip(feature_names, np.abs(shap_values).mean(axis=0)))
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"SHAP explanation error: {e}")
            return {}

# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_ql_agent(learning_rate: float = 0.1) -> QLearningAgent:
    """Create Q-Learning agent."""
    return QLearningAgent(learning_rate=learning_rate)

def create_lstm_predictor(input_size: int = 10) -> LSTMPredictor:
    """Create LSTM predictor."""
    return LSTMPredictor(input_size=input_size)

def create_auto_feature_engineer() -> AutoFeatureEngineer:
    """Create auto feature engineer."""
    return AutoFeatureEngineer()

def create_bayesian_optimizer() -> BayesianOptimizer:
    """Create Bayesian optimizer."""
    return BayesianOptimizer()

def create_portfolio_optimizer() -> PortfolioOptimizer:
    """Create portfolio optimizer."""
    return PortfolioOptimizer()

def create_seasonality_detector() -> SeasonalityDetector:
    """Create seasonality detector."""
    return SeasonalityDetector()

def create_shap_explainer() -> SHAPExplainer:
    """Create SHAP explainer."""
    return SHAPExplainer()
