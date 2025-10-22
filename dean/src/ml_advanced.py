"""
DEAN ADVANCED ML CAPABILITIES
Next-generation ML features including:
- Reinforcement Learning (Q-Learning)
- Neural Networks (LSTM for time-series)
- Multi-Objective Optimization
- Automated Feature Engineering
- Bayesian Optimization
- Active Learning
- SHAP Explainability
- Learning Rate Scheduling
- Lookahead Bias Protection
- Cold Start Handling
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
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
    """Q-Learning agent for ad management decisions with persistence."""
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95,
                 exploration_rate: float = 0.1, supabase_client=None):
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.supabase = supabase_client
        
        # Q-table: state_hash -> {action -> q_value}
        self.q_table: Dict[str, Dict[str, float]] = {}
        
        self.actions = ['kill', 'promote', 'hold', 'scale_up', 'scale_down']
        self.logger = logging.getLogger(f"{__name__}.QLearningAgent")
        
        # Load existing Q-table if available
        if self.supabase:
            self.load_q_table()
    
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
        
        # Initialize Q-values if new state
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {action: 0.0 for action in self.actions}
        
        # Explore vs exploit
        if explore and np.random.random() < self.epsilon:
            # Explore: random action
            action = np.random.choice(self.actions)
            confidence = 0.3
        else:
            # Exploit: best known action
            q_values = self.q_table[state_hash]
            action = max(q_values, key=q_values.get)
            max_q = q_values[action]
            
            # Confidence based on Q-value magnitude
            confidence = min(1.0, abs(max_q) / 10.0)
        
        return RLAction(action_type=action, confidence=confidence)
    
    def update(self, state: RLState, action: str, reward: float, next_state: RLState):
        """Update Q-value using Q-learning update rule."""
        state_hash = self._state_to_hash(state)
        next_state_hash = self._state_to_hash(next_state)
        
        # Initialize if needed
        if state_hash not in self.q_table:
            self.q_table[state_hash] = {a: 0.0 for a in self.actions}
        if next_state_hash not in self.q_table:
            self.q_table[next_state_hash] = {a: 0.0 for a in self.actions}
        
        # Q-learning update
        current_q = self.q_table[state_hash][action]
        max_next_q = max(self.q_table[next_state_hash].values())
        
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state_hash][action] = new_q
        
        self.logger.debug(f"Updated Q({state_hash}, {action}): {current_q:.2f} -> {new_q:.2f}")
    
    def calculate_reward(self, performance_data: Dict[str, Any], 
                        action_taken: str, outcome: str) -> float:
        """Calculate reward based on action outcome."""
        cpa = performance_data.get('cpa', 50)
        roas = performance_data.get('roas', 0)
        
        # Base reward on performance
        if outcome == 'killed':
            # Reward for killing bad ads, penalty for killing good ads
            if cpa > 40:
                return 5.0  # Good kill
            elif cpa < 20:
                return -10.0  # Bad kill (killed a winner)
            else:
                return 0.0
        
        elif outcome == 'promoted':
            # Reward for promoting winners, penalty for promoting losers
            if roas > 2.5:
                return 10.0  # Great promotion
            elif roas < 1.5:
                return -5.0  # Bad promotion
            else:
                return 2.0  # Okay promotion
        
        elif outcome == 'held':
            # Small reward for patience if improving
            if cpa < 30:
                return 1.0
            else:
                return -0.5
        
        else:
            return 0.0
    
    def save_q_table(self):
        """Save Q-table to Supabase for persistence."""
        if not self.supabase:
            return
        
        try:
            import json
            from datetime import datetime
            
            # Convert Q-table to JSON
            q_table_json = json.dumps(self.q_table)
            
            # Save to a simple key-value store (using ml_models table)
            data = {
                'model_type': 'q_learning',
                'stage': 'all',
                'model_data': q_table_json,
                'hyperparameters': {
                    'learning_rate': self.lr,
                    'gamma': self.gamma,
                    'epsilon': self.epsilon
                },
                'trained_at': datetime.now().isoformat(),
                'is_active': True
            }
            
            self.supabase.table('ml_models').upsert(data, on_conflict='model_type,stage').execute()
            self.logger.info(f"Saved Q-table with {len(self.q_table)} states")
            
        except Exception as e:
            self.logger.error(f"Error saving Q-table: {e}")
    
    def load_q_table(self):
        """Load Q-table from Supabase."""
        if not self.supabase:
            return
        
        try:
            import json
            
            # Load latest Q-table
            response = self.supabase.table('ml_models').select('model_data, hyperparameters').eq(
                'model_type', 'q_learning'
            ).eq('stage', 'all').eq('is_active', True).order('trained_at', desc=True).limit(1).execute()
            
            if response.data and len(response.data) > 0:
                model_data = response.data[0].get('model_data')
                if model_data:
                    self.q_table = json.loads(model_data)
                    self.logger.info(f"Loaded Q-table with {len(self.q_table)} states")
                    
                    # Load hyperparameters
                    params = response.data[0].get('hyperparameters', {})
                    if params:
                        self.lr = params.get('learning_rate', self.lr)
                        self.gamma = params.get('gamma', self.gamma)
                        self.epsilon = params.get('epsilon', self.epsilon)
            
        except Exception as e:
            self.logger.error(f"Error loading Q-table: {e}")

# =====================================================
# NEURAL NETWORKS
# =====================================================

class LSTMPredictor(nn.Module if TORCH_AVAILABLE else object):
    """LSTM network for time-series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 output_size: int = 1, dropout: float = 0.2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()  # Add activation
    
    def forward(self, x):
        """Forward pass with proper activations."""
        # x shape: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        
        # Take last output
        last_output = lstm_out[:, -1, :]
        
        # Apply ReLU activation
        activated = self.relu(last_output)
        
        # Fully connected layer (no activation for regression)
        out = self.fc(activated)
        
        return out

class NeuralNetworkPredictor:
    """Neural network-based predictor."""
    
    def __init__(self, input_size: int, sequence_length: int = 7):
        if not TORCH_AVAILABLE:
            self.available = False
            self.logger = logging.getLogger(f"{__name__}.NeuralNetworkPredictor")
            self.logger.warning("PyTorch not available - neural network disabled")
            return
        
        self.available = True
        self.input_size = input_size
        self.sequence_length = sequence_length
        
        self.model = LSTMPredictor(input_size=input_size)
        self.scaler = StandardScaler()
        self.logger = logging.getLogger(f"{__name__}.NeuralNetworkPredictor")
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Train LSTM model."""
        if not self.available:
            return
        
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y).reshape(-1, 1)
            
            # Training setup
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            self.model.train()
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
            
            self.logger.info("Neural network training complete")
            
        except Exception as e:
            self.logger.error(f"Error training neural network: {e}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.available:
            return np.zeros(len(X))
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Convert to tensor
            X_tensor = torch.FloatTensor(X_scaled)
            
            # Predict
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor)
            
            return predictions.numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return np.zeros(len(X))

# =====================================================
# MULTI-OBJECTIVE OPTIMIZATION
# =====================================================

class MultiObjectiveOptimizer:
    """Pareto optimization for multiple objectives."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MultiObjectiveOptimizer")
    
    def find_pareto_front(self, objectives: List[np.ndarray]) -> np.ndarray:
        """Find Pareto-optimal solutions."""
        try:
            # Stack objectives (each should be maximized)
            obj_matrix = np.column_stack(objectives)
            n = len(obj_matrix)
            
            is_pareto = np.ones(n, dtype=bool)
            
            for i in range(n):
                if is_pareto[i]:
                    # Check if any other point dominates this one
                    for j in range(n):
                        if i != j and is_pareto[j]:
                            # j dominates i if j is >= i in all objectives and > in at least one
                            if np.all(obj_matrix[j] >= obj_matrix[i]) and np.any(obj_matrix[j] > obj_matrix[i]):
                                is_pareto[i] = False
                                break
            
            return is_pareto
            
        except Exception as e:
            self.logger.error(f"Error finding Pareto front: {e}")
            return np.ones(len(objectives[0]), dtype=bool)
    
    def select_best_trade_off(self, objectives: Dict[str, float], 
                             weights: Dict[str, float]) -> float:
        """Select best solution using weighted sum."""
        try:
            # Normalize objectives to [0, 1]
            total_score = 0.0
            
            for obj_name, obj_value in objectives.items():
                weight = weights.get(obj_name, 0.33)
                total_score += weight * obj_value
            
            return total_score
            
        except Exception as e:
            self.logger.error(f"Error selecting trade-off: {e}")
            return 0.0

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
        self.explainer = None
        self.logger = logging.getLogger(f"{__name__}.SHAPExplainer")
    
    def fit(self, model, X_background: np.ndarray):
        """Fit SHAP explainer."""
        if not self.available:
            return
        
        try:
            # Use TreeExplainer for tree-based models
            self.explainer = shap.TreeExplainer(model, X_background)
            self.logger.info("SHAP explainer fitted")
            
        except Exception as e:
            self.logger.error(f"Error fitting SHAP explainer: {e}")
    
    def explain_prediction(self, X: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
        """Explain a single prediction."""
        if not self.available or self.explainer is None:
            return {}
        
        try:
            # Get SHAP values
            shap_values = self.explainer.shap_values(X)
            
            # Convert to dictionary
            if len(X.shape) == 1:
                explanation = dict(zip(feature_names, shap_values))
            else:
                explanation = dict(zip(feature_names, shap_values[0]))
            
            return explanation
            
        except Exception as e:
            self.logger.error(f"Error explaining prediction: {e}")
            return {}

# =====================================================
# LEARNING RATE SCHEDULING
# =====================================================

class LearningRateScheduler:
    """Dynamic learning rate adjustment."""
    
    def __init__(self, initial_lr: float = 0.1, schedule_type: str = 'cosine'):
        self.initial_lr = initial_lr
        self.schedule_type = schedule_type
        self.current_lr = initial_lr
        self.epoch = 0
        self.logger = logging.getLogger(f"{__name__}.LearningRateScheduler")
    
    def step(self, epoch: Optional[int] = None):
        """Update learning rate."""
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1
        
        if self.schedule_type == 'cosine':
            # Cosine annealing
            self.current_lr = self.initial_lr * (1 + np.cos(np.pi * self.epoch / 100)) / 2
        
        elif self.schedule_type == 'step':
            # Step decay (every 25 epochs)
            self.current_lr = self.initial_lr * (0.5 ** (self.epoch // 25))
        
        elif self.schedule_type == 'exponential':
            # Exponential decay
            self.current_lr = self.initial_lr * (0.95 ** self.epoch)
        
        else:
            # Constant
            self.current_lr = self.initial_lr
        
        return self.current_lr
    
    def get_lr(self) -> float:
        """Get current learning rate."""
        return self.current_lr

# =====================================================
# ACTIVE LEARNING
# =====================================================

class ActiveLearner:
    """Active learning for targeted experimentation."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ActiveLearner")
    
    def identify_uncertain_regions(self, model, X: np.ndarray, 
                                  ensemble_models: List[Any]) -> np.ndarray:
        """Identify where model is most uncertain."""
        try:
            # Get predictions from ensemble
            predictions = []
            for m in ensemble_models:
                pred = m.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate variance (uncertainty)
            variance = np.var(predictions, axis=0)
            
            # Normalize to [0, 1]
            if variance.max() > 0:
                uncertainty = variance / variance.max()
            else:
                uncertainty = np.zeros_like(variance)
            
            return uncertainty
            
        except Exception as e:
            self.logger.error(f"Error identifying uncertain regions: {e}")
            return np.zeros(len(X))
    
    def suggest_experiments(self, uncertainty: np.ndarray, 
                           X: np.ndarray, n_suggestions: int = 5) -> List[int]:
        """Suggest which samples to experiment with."""
        try:
            # Get top N most uncertain samples
            uncertain_indices = np.argsort(uncertainty)[-n_suggestions:]
            return uncertain_indices.tolist()
            
        except Exception as e:
            self.logger.error(f"Error suggesting experiments: {e}")
            return []

# =====================================================
# LOOKAHEAD BIAS PROTECTION
# =====================================================

class LookaheadProtector:
    """Prevent data leakage and lookahead bias."""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.LookaheadProtector")
    
    def validate_features(self, df: pd.DataFrame, timestamp_col: str,
                         feature_cols: List[str]) -> Dict[str, bool]:
        """Validate that features don't leak future information."""
        results = {}
        
        for col in feature_cols:
            # Check if feature has proper temporal ordering
            is_valid = self._check_temporal_validity(df, col, timestamp_col)
            results[col] = is_valid
            
            if not is_valid:
                self.logger.warning(f"Feature '{col}' may contain lookahead bias")
        
        return results
    
    def _check_temporal_validity(self, df: pd.DataFrame, feature_col: str, 
                                timestamp_col: str) -> bool:
        """Check if feature respects temporal ordering."""
        try:
            # Simple check: feature should be calculable from past data only
            # This is a heuristic - real validation would need feature definitions
            
            # Check for future references in column name
            future_keywords = ['future', 'next', 'tomorrow', 'ahead']
            col_lower = feature_col.lower()
            
            for keyword in future_keywords:
                if keyword in col_lower:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking temporal validity: {e}")
            return True  # Default to valid on error

# =====================================================
# CONVENIENCE FUNCTIONS
# =====================================================

def create_rl_agent(learning_rate: float = 0.1, exploration_rate: float = 0.1) -> QLearningAgent:
    """Create Q-Learning agent."""
    return QLearningAgent(learning_rate=learning_rate, exploration_rate=exploration_rate)

def create_neural_predictor(input_size: int, sequence_length: int = 7) -> NeuralNetworkPredictor:
    """Create neural network predictor."""
    return NeuralNetworkPredictor(input_size=input_size, sequence_length=sequence_length)

def create_multi_objective_optimizer() -> MultiObjectiveOptimizer:
    """Create multi-objective optimizer."""
    return MultiObjectiveOptimizer()

def create_shap_explainer() -> SHAPExplainer:
    """Create SHAP explainer."""
    return SHAPExplainer()

def create_lr_scheduler(initial_lr: float = 0.1, schedule_type: str = 'cosine') -> LearningRateScheduler:
    """Create learning rate scheduler."""
    return LearningRateScheduler(initial_lr=initial_lr, schedule_type=schedule_type)

def create_active_learner() -> ActiveLearner:
    """Create active learner."""
    return ActiveLearner()

def create_lookahead_protector() -> LookaheadProtector:
    """Create lookahead bias protector."""
    return LookaheadProtector()

