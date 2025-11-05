"""
Creative DNA Analysis System
Advanced creative intelligence using embeddings and similarity matching
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not available, using simple embeddings")

try:
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("sklearn not available, using simple cosine similarity")


@dataclass
class CreativeDNA:
    """Creative DNA representation with embeddings and metadata."""
    creative_id: str
    ad_id: str
    image_prompt: str
    text_overlay: str
    ad_copy: Dict[str, str]
    image_embedding: Optional[np.ndarray] = None
    text_embedding: Optional[np.ndarray] = None
    combined_embedding: Optional[np.ndarray] = None
    performance_score: float = 0.0
    roas: float = 0.0
    ctr: float = 0.0
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.created_at is None:
            self.created_at = datetime.now()


class CreativeDNAAnalyzer:
    """Analyzes creative DNA patterns and similarity."""
    
    def __init__(self, supabase_client=None):
        self.supabase_client = supabase_client
        self.embedding_model = None
        self.creative_db: Dict[str, CreativeDNA] = {}
        
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Loaded sentence transformer for creative DNA")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
    
    def _create_simple_embedding(self, text: str) -> np.ndarray:
        """Create simple embedding using hash-based method."""
        # Simple TF-IDF-like embedding
        words = text.lower().split()
        embedding = np.zeros(384)  # Standard size
        
        for i, word in enumerate(words[:384]):
            hash_val = hash(word) % 384
            embedding[hash_val] += 1.0 / (i + 1)
        
        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def create_text_embedding(self, text: str) -> np.ndarray:
        """Create embedding for text."""
        if not text:
            return np.zeros(384)
        
        if self.embedding_model:
            try:
                return self.embedding_model.encode(text, convert_to_numpy=True)
            except Exception:
                pass
        
        return self._create_simple_embedding(text)
    
    def create_creative_dna(
        self,
        creative_id: str,
        ad_id: str,
        image_prompt: str,
        text_overlay: str,
        ad_copy: Dict[str, str],
        performance_data: Optional[Dict[str, Any]] = None,
    ) -> CreativeDNA:
        """Create Creative DNA for a creative."""
        # Combine all text elements
        text_elements = [
            image_prompt,
            text_overlay or "",
            ad_copy.get("headline", ""),
            ad_copy.get("primary_text", ""),
            ad_copy.get("description", ""),
        ]
        combined_text = " ".join(filter(None, text_elements))
        
        # Create embeddings
        text_embedding = self.create_text_embedding(combined_text)
        image_embedding = self.create_text_embedding(image_prompt)  # Use prompt as image proxy
        
        # Combined embedding (weighted average)
        combined_embedding = (text_embedding * 0.6 + image_embedding * 0.4)
        
        # Normalize
        norm = np.linalg.norm(combined_embedding)
        if norm > 0:
            combined_embedding = combined_embedding / norm
        
        # Calculate performance score
        performance_score = 0.0
        roas = 0.0
        ctr = 0.0
        
        if performance_data:
            roas = float(performance_data.get("roas", 0))
            ctr = float(performance_data.get("ctr", 0))
            purchases = float(performance_data.get("purchases", 0))
            spend = float(performance_data.get("spend", 0))
            
            # Composite performance score
            if spend > 0:
                performance_score = (
                    (roas * 0.5) +
                    (ctr * 100 * 0.3) +
                    (purchases / spend * 10 * 0.2)
                )
        
        dna = CreativeDNA(
            creative_id=creative_id,
            ad_id=ad_id,
            image_prompt=image_prompt,
            text_overlay=text_overlay,
            ad_copy=ad_copy,
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            combined_embedding=combined_embedding,
            performance_score=performance_score,
            roas=roas,
            ctr=ctr,
            metadata=performance_data or {},
        )
        
        # Store in memory
        self.creative_db[creative_id] = dna
        
        # Store in Supabase if available
        if self.supabase_client:
            try:
                self._store_dna_in_supabase(dna)
            except Exception as e:
                logger.error(f"Failed to store DNA in Supabase: {e}")
        
        return dna
    
    def _store_dna_in_supabase(self, dna: CreativeDNA):
        """Store Creative DNA in Supabase."""
        try:
            # Store embedding as JSON array
            embedding_data = {
                "creative_id": dna.creative_id,
                "ad_id": dna.ad_id,
                "image_prompt": dna.image_prompt,
                "text_overlay": dna.text_overlay,
                "ad_copy": dna.ad_copy,
                "image_embedding": dna.image_embedding.tolist() if dna.image_embedding is not None else None,
                "text_embedding": dna.text_embedding.tolist() if dna.text_embedding is not None else None,
                "combined_embedding": dna.combined_embedding.tolist() if dna.combined_embedding is not None else None,
                "performance_score": dna.performance_score,
                "roas": dna.roas,
                "ctr": dna.ctr,
                "created_at": dna.created_at.isoformat() if dna.created_at else None,
                "metadata": dna.metadata,
            }
            
            # Try to insert/update in creative_intelligence table
            self.supabase_client.table('creative_intelligence').upsert(
                {
                    "creative_id": dna.creative_id,
                    "ad_id": dna.ad_id,
                    "similarity_vector": embedding_data["combined_embedding"],
                    "metadata": embedding_data,
                }
            ).execute()
        except Exception as e:
            logger.error(f"Error storing DNA in Supabase: {e}")
    
    def find_similar_creatives(
        self,
        creative_id: str,
        top_k: int = 5,
        min_similarity: float = 0.7,
    ) -> List[Tuple[CreativeDNA, float]]:
        """Find similar creatives based on DNA."""
        if creative_id not in self.creative_db:
            return []
        
        query_dna = self.creative_db[creative_id]
        if query_dna.combined_embedding is None:
            return []
        
        similarities = []
        for other_id, other_dna in self.creative_db.items():
            if other_id == creative_id:
                continue
            
            if other_dna.combined_embedding is None:
                continue
            
            # Calculate cosine similarity
            similarity = np.dot(
                query_dna.combined_embedding,
                other_dna.combined_embedding
            )
            
            if similarity >= min_similarity:
                similarities.append((other_dna, float(similarity)))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def find_winning_patterns(self, min_performance_score: float = 0.5) -> Dict[str, Any]:
        """Identify winning creative patterns."""
        winners = [
            dna for dna in self.creative_db.values()
            if dna.performance_score >= min_performance_score
        ]
        
        if not winners:
            return {}
        
        # Analyze common patterns
        common_prompts = {}
        common_text_overlays = {}
        common_ad_copy_patterns = {}
        
        for winner in winners:
            # Analyze prompts
            prompt_keywords = set(winner.image_prompt.lower().split())
            for keyword in prompt_keywords:
                common_prompts[keyword] = common_prompts.get(keyword, 0) + 1
            
            # Analyze text overlays
            if winner.text_overlay:
                common_text_overlays[winner.text_overlay] = common_text_overlays.get(winner.text_overlay, 0) + 1
            
            # Analyze ad copy
            for key, value in winner.ad_copy.items():
                if value:
                    common_ad_copy_patterns[f"{key}:{value[:50]}"] = common_ad_copy_patterns.get(f"{key}:{value[:50]}", 0) + 1
        
        # Get top patterns
        top_prompt_keywords = sorted(common_prompts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_text_overlays = sorted(common_text_overlays.items(), key=lambda x: x[1], reverse=True)[:5]
        top_ad_copy = sorted(common_ad_copy_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return {
            "winning_count": len(winners),
            "top_prompt_keywords": [kw[0] for kw in top_prompt_keywords],
            "top_text_overlays": [to[0] for to in top_text_overlays],
            "top_ad_copy_patterns": [ac[0] for ac in top_ad_copy],
            "average_performance_score": sum(w.performance_score for w in winners) / len(winners),
            "average_roas": sum(w.roas for w in winners) / len(winners),
        }
    
    def predict_creative_success(
        self,
        image_prompt: str,
        text_overlay: str,
        ad_copy: Dict[str, str],
    ) -> Dict[str, Any]:
        """Predict creative success based on similarity to winners."""
        # Create temporary DNA for prediction
        temp_dna = self.create_creative_dna(
            creative_id="temp_prediction",
            ad_id="temp",
            image_prompt=image_prompt,
            text_overlay=text_overlay,
            ad_copy=ad_copy,
        )
        
        # Find similar winners
        winners = [
            dna for dna in self.creative_db.values()
            if dna.performance_score >= 0.5
        ]
        
        if not winners:
            return {
                "predicted_success": 0.5,
                "confidence": 0.0,
                "similar_winners": 0,
            }
        
        # Calculate similarity to winners
        similarities = []
        for winner in winners:
            if winner.combined_embedding is not None and temp_dna.combined_embedding is not None:
                similarity = np.dot(winner.combined_embedding, temp_dna.combined_embedding)
                similarities.append((similarity, winner.performance_score))
        
        if not similarities:
            return {
                "predicted_success": 0.5,
                "confidence": 0.0,
                "similar_winners": 0,
            }
        
        # Weighted prediction based on similarity
        total_weight = 0.0
        weighted_score = 0.0
        
        for similarity, perf_score in similarities:
            weight = similarity ** 2  # Square to emphasize high similarity
            weighted_score += perf_score * weight
            total_weight += weight
        
        predicted_success = weighted_score / total_weight if total_weight > 0 else 0.5
        
        return {
            "predicted_success": float(predicted_success),
            "confidence": float(min(total_weight / len(similarities), 1.0)) if similarities else 0.0,
            "similar_winners": len(similarities),
            "max_similarity": float(max(s[0] for s in similarities)) if similarities else 0.0,
        }


def create_creative_dna_analyzer(supabase_client=None) -> CreativeDNAAnalyzer:
    """Create a Creative DNA Analyzer instance."""
    return CreativeDNAAnalyzer(supabase_client=supabase_client)


__all__ = ["CreativeDNAAnalyzer", "CreativeDNA", "create_creative_dna_analyzer"]

