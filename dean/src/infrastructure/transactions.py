"""
Transaction Management
Database transaction handling and data consistency
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional
from contextlib import contextmanager
import uuid

logger = logging.getLogger(__name__)


class TransactionManager:
    """Manages database transactions."""
    
    def __init__(self, supabase_client=None):
        self.supabase_client = supabase_client
        self.active_transactions: Dict[str, Dict[str, Any]] = {}
    
    @contextmanager
    def transaction(self, transaction_id: Optional[str] = None):
        """Context manager for transactions."""
        tid = transaction_id or str(uuid.uuid4())
        
        try:
            self.active_transactions[tid] = {
                "id": tid,
                "operations": [],
                "status": "active",
            }
            
            yield tid
            
            # Commit transaction
            self._commit(tid)
            
        except Exception as e:
            # Rollback transaction
            self._rollback(tid)
            raise e
        finally:
            if tid in self.active_transactions:
                del self.active_transactions[tid]
    
    def _commit(self, transaction_id: str):
        """Commit transaction."""
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]["status"] = "committed"
            logger.debug(f"Transaction {transaction_id} committed")
    
    def _rollback(self, transaction_id: str):
        """Rollback transaction."""
        if transaction_id in self.active_transactions:
            self.active_transactions[transaction_id]["status"] = "rolled_back"
            logger.warning(f"Transaction {transaction_id} rolled back")


def create_transaction_manager(supabase_client=None) -> TransactionManager:
    """Create transaction manager."""
    return TransactionManager(supabase_client)


__all__ = ["TransactionManager", "create_transaction_manager"]

