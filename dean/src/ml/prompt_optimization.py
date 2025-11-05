"""
Advanced Prompt Optimization
RL-based prompt optimization, evolutionary algorithms, prompt library, versioning
"""

from __future__ import annotations

import logging
import random
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class PromptVersion:
    """Prompt version with performance tracking."""
    version: str
    prompt: str
    performance_score: float = 0.0
    roas: float = 0.0
    ctr: float = 0.0
    usage_count: int = 0
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class PromptLibrary:
    """Library of prompts with performance scores."""
    
    def __init__(self):
        self.prompts: Dict[str, PromptVersion] = {}
    
    def add_prompt(
        self,
        prompt: str,
        performance_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Add prompt to library."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:12]
        
        version = PromptVersion(
            version=prompt_hash,
            prompt=prompt,
        )
        
        if performance_data:
            version.performance_score = performance_data.get("performance_score", 0.0)
            version.roas = performance_data.get("roas", 0.0)
            version.ctr = performance_data.get("ctr", 0.0)
        
        self.prompts[prompt_hash] = version
        return prompt_hash
    
    def update_performance(
        self,
        prompt_hash: str,
        roas: float,
        ctr: float,
    ):
        """Update prompt performance."""
        if prompt_hash not in self.prompts:
            return
        
        prompt = self.prompts[prompt_hash]
        prompt.usage_count += 1
        
        # Update averages
        if prompt.usage_count == 1:
            prompt.roas = roas
            prompt.ctr = ctr
        else:
            prompt.roas = (
                (prompt.roas * (prompt.usage_count - 1) + roas) /
                prompt.usage_count
            )
            prompt.ctr = (
                (prompt.ctr * (prompt.usage_count - 1) + ctr) /
                prompt.usage_count
            )
        
        # Update performance score
        prompt.performance_score = prompt.roas * 0.6 + prompt.ctr * 100 * 0.4
    
    def get_top_prompts(self, top_k: int = 10) -> List[PromptVersion]:
        """Get top performing prompts."""
        sorted_prompts = sorted(
            self.prompts.values(),
            key=lambda p: p.performance_score,
            reverse=True,
        )
        return sorted_prompts[:top_k]
    
    def get_prompt_for_context(
        self,
        context: Dict[str, Any],
        strategy: str = "best",
    ) -> Optional[str]:
        """Get prompt based on context."""
        if strategy == "best":
            top = self.get_top_prompts(1)
            return top[0].prompt if top else None
        elif strategy == "random_top5":
            top = self.get_top_prompts(5)
            if top:
                import random
                return random.choice(top).prompt
        return None


class ReinforcementLearningPromptOptimizer:
    """RL-based prompt optimization."""
    
    def __init__(self):
        self.state_space_size = 10  # Simplified
        self.action_space_size = 5  # Prompt modifications
        self.q_table: Dict[tuple, float] = {}
        self.learning_rate = 0.1
        self.epsilon = 0.1
    
    def optimize_prompt(
        self,
        base_prompt: str,
        context: Dict[str, Any],
    ) -> str:
        """Optimize prompt using RL."""
        # Simplified RL implementation
        # In production, would use more sophisticated RL
        
        # Extract state from context
        state = self._extract_state(context)
        
        # Select action (prompt modification)
        action = self._select_action(state)
        
        # Apply action to prompt
        optimized = self._apply_action(base_prompt, action)
        
        return optimized
    
    def _extract_state(self, context: Dict[str, Any]) -> tuple:
        """Extract state from context."""
        # Simplified state representation
        return tuple([0] * self.state_space_size)
    
    def _select_action(self, state: tuple) -> int:
        """Select action using epsilon-greedy."""
        import random
        
        if random.random() < self.epsilon:
            return random.randint(0, self.action_space_size - 1)
        
        # Exploit
        q_values = [
            self.q_table.get((state, a), 0.0)
            for a in range(self.action_space_size)
        ]
        return q_values.index(max(q_values)) if q_values else 0
    
    def _apply_action(self, prompt: str, action: int) -> str:
        """Apply action to prompt."""
        modifiers = [
            ", editorial, cinematic",
            ", minimalist, sophisticated",
            ", luxury, refined",
            ", premium, timeless",
            ", elegant, understated",
        ]
        
        if action < len(modifiers):
            return f"{prompt}{modifiers[action]}"
        
        return prompt
    
    def update(
        self,
        prompt: str,
        reward: float,
        context: Dict[str, Any],
    ):
        """Update Q-table with reward."""
        state = self._extract_state(context)
        action = self._select_action(state)
        
        # Update Q-value
        current_q = self.q_table.get((state, action), 0.0)
        new_q = current_q + self.learning_rate * (reward - current_q)
        self.q_table[(state, action)] = new_q


def create_prompt_library() -> PromptLibrary:
    """Create prompt library."""
    return PromptLibrary()


def create_rl_prompt_optimizer() -> ReinforcementLearningPromptOptimizer:
    """Create RL prompt optimizer."""
    return ReinforcementLearningPromptOptimizer()


# =====================================================
# EVOLUTIONARY ALGORITHMS
# =====================================================

@dataclass
class PromptGene:
    """A prompt gene in the evolution pool."""
    prompt: str
    performance_score: float = 0.0
    roas: float = 0.0
    ctr: float = 0.0
    usage_count: int = 0
    created_at: datetime = None
    generation: int = 0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class PromptEvolutionEngine:
    """Evolutionary algorithm for prompt optimization."""
    
    def __init__(self, population_size: int = 20):
        self.population_size = population_size
        self.gene_pool: List[PromptGene] = []
        self.generation = 0
        self.mutation_rate = 0.1
        self.crossover_rate = 0.3
    
    def initialize_population(
        self,
        base_prompts: List[str],
    ):
        """Initialize gene pool with base prompts."""
        for prompt in base_prompts:
            gene = PromptGene(
                prompt=prompt,
                generation=0,
            )
            self.gene_pool.append(gene)
        
        # Fill remaining with variations
        while len(self.gene_pool) < self.population_size:
            base = random.choice(base_prompts)
            mutated = self._mutate(base)
            gene = PromptGene(
                prompt=mutated,
                generation=0,
            )
            self.gene_pool.append(gene)
    
    def _mutate(self, prompt: str) -> str:
        """Mutate a prompt."""
        words = prompt.split(", ")
        
        # Style modifiers
        style_modifiers = [
            "editorial", "cinematic", "minimalist", "sophisticated",
            "luxury", "refined", "premium", "timeless", "elegant",
        ]
        
        # Lighting modifiers
        lighting_modifiers = [
            "natural daylight", "soft shadows", "warm golden hour",
            "studio lighting", "professional", "dramatic contrast",
            "soft diffused light",
        ]
        
        # Random mutation
        if random.random() < 0.3:
            # Add a style modifier
            modifier = random.choice(style_modifiers)
            if modifier not in prompt.lower():
                words.append(modifier)
        
        if random.random() < 0.3:
            # Add a lighting modifier
            modifier = random.choice(lighting_modifiers)
            if modifier not in prompt.lower():
                words.append(modifier)
        
        # Randomly remove a word (10% chance)
        if len(words) > 5 and random.random() < 0.1:
            words.pop(random.randint(0, len(words) - 1))
        
        return ", ".join(words)
    
    def _crossover(self, parent1: PromptGene, parent2: PromptGene) -> str:
        """Crossover two prompts."""
        words1 = set(parent1.prompt.split(", "))
        words2 = set(parent2.prompt.split(", "))
        
        # Combine words from both parents
        combined = list(words1.union(words2))
        
        # Take random subset
        size = max(len(words1), len(words2))
        if len(combined) > size:
            combined = random.sample(combined, size)
        
        return ", ".join(combined)
    
    def evolve(self) -> List[PromptGene]:
        """Evolve the next generation."""
        if not self.gene_pool:
            return []
        
        # Sort by performance
        sorted_pool = sorted(
            self.gene_pool,
            key=lambda g: g.performance_score,
            reverse=True,
        )
        
        # Keep top 50%
        elite_size = max(1, len(sorted_pool) // 2)
        elite = sorted_pool[:elite_size]
        
        # Create new generation
        new_generation = []
        
        # Keep elite
        for gene in elite:
            new_gene = PromptGene(
                prompt=gene.prompt,
                performance_score=gene.performance_score,
                roas=gene.roas,
                ctr=gene.ctr,
                generation=self.generation + 1,
            )
            new_generation.append(new_gene)
        
        # Crossover
        while len(new_generation) < self.population_size:
            if random.random() < self.crossover_rate and len(elite) >= 2:
                parent1 = random.choice(elite)
                parent2 = random.choice(elite)
                if parent1 != parent2:
                    child_prompt = self._crossover(parent1, parent2)
                    child = PromptGene(
                        prompt=child_prompt,
                        generation=self.generation + 1,
                    )
                    new_generation.append(child)
                    continue
            
            # Mutation
            parent = random.choice(elite)
            mutated_prompt = self._mutate(parent.prompt)
            child = PromptGene(
                prompt=mutated_prompt,
                generation=self.generation + 1,
            )
            new_generation.append(child)
        
        self.gene_pool = new_generation
        self.generation += 1
        
        return self.gene_pool
    
    def update_performance(
        self,
        prompt: str,
        performance_data: Dict[str, Any],
    ):
        """Update performance for a prompt."""
        # Find matching gene
        for gene in self.gene_pool:
            if gene.prompt == prompt:
                gene.performance_score = performance_data.get("performance_score", 0.0)
                gene.roas = performance_data.get("roas", 0.0)
                gene.ctr = performance_data.get("ctr", 0.0)
                gene.usage_count += 1
                break
    
    def get_best_prompts(self, top_k: int = 5) -> List[PromptGene]:
        """Get top performing prompts."""
        sorted_pool = sorted(
            self.gene_pool,
            key=lambda g: g.performance_score,
            reverse=True,
        )
        return sorted_pool[:top_k]
    
    def select_prompt(self, strategy: str = "exploit") -> str:
        """Select a prompt from the gene pool."""
        if not self.gene_pool:
            return ""
        
        if strategy == "exploit":
            # Select best performer
            best = max(self.gene_pool, key=lambda g: g.performance_score)
            return best.prompt
        
        elif strategy == "explore":
            # Select random
            return random.choice(self.gene_pool).prompt
        
        elif strategy == "balanced":
            # Weighted random selection
            scores = [max(g.performance_score, 0.1) for g in self.gene_pool]
            total = sum(scores)
            if total > 0:
                weights = [s / total for s in scores]
                selected = random.choices(self.gene_pool, weights=weights, k=1)[0]
                return selected.prompt
        
        return self.gene_pool[0].prompt


def create_prompt_evolution_engine(population_size: int = 20) -> PromptEvolutionEngine:
    """Create a prompt evolution engine."""
    return PromptEvolutionEngine(population_size=population_size)


__all__ = [
    "PromptLibrary",
    "PromptVersion",
    "ReinforcementLearningPromptOptimizer",
    "PromptEvolutionEngine",
    "PromptGene",
    "create_prompt_library",
    "create_rl_prompt_optimizer",
    "create_prompt_evolution_engine",
]

