"""
Unified Optimization System
Database optimization, resource optimization (memory, CPU, network, images)
"""

from __future__ import annotations

import logging
import os
import sys
import gc
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import io

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available, resource monitoring disabled")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available, image optimization disabled")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =====================================================
# DATABASE OPTIMIZATION
# =====================================================

@dataclass
class ArchivalRule:
    """Rule for archiving old data."""
    table_name: str
    archive_after_days: int
    archive_to_table: Optional[str] = None  # If None, delete instead of archive
    condition_column: str = "created_at"  # Column to check for age
    batch_size: int = 1000
    enabled: bool = True


@dataclass
class IndexRecommendation:
    """Index optimization recommendation."""
    table_name: str
    column_names: List[str]
    index_type: str  # "btree", "hash", "gin", "gist"
    reason: str
    estimated_improvement: str
    priority: int  # 1-10, higher = more important


class DatabaseOptimizer:
    """Database optimization and maintenance system."""
    
    def __init__(self, supabase_client: Any) -> None:
        self.client = supabase_client
        self.archival_rules: List[ArchivalRule] = []
        self.last_optimization: Optional[datetime] = None
        self.optimization_interval_hours: int = 24  # Run daily
    
    def add_archival_rule(self, rule: ArchivalRule) -> None:
        """Add an archival rule."""
        self.archival_rules.append(rule)
        logger.info(f"Added archival rule for {rule.table_name}: archive after {rule.archive_after_days} days")
    
    def analyze_query_performance(
        self,
        query: str,
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Analyze query performance and suggest optimizations."""
        recommendations = []
        
        # Check for missing indexes on WHERE clauses
        if "WHERE" in query.upper():
            import re
            where_pattern = r"WHERE\s+(\w+)\s*="
            matches = re.findall(where_pattern, query, re.IGNORECASE)
            for column in matches:
                recommendations.append({
                    "type": "missing_index",
                    "table": table_name or "unknown",
                    "column": column,
                    "recommendation": f"Create index on {table_name}.{column}",
                })
        
        # Check for full table scans
        if "ORDER BY" in query.upper() and "LIMIT" not in query.upper():
            recommendations.append({
                "type": "missing_limit",
                "recommendation": "Add LIMIT clause to prevent full table scans",
            })
        
        return {
            "query": query,
            "recommendations": recommendations,
            "analyzed_at": datetime.now().isoformat(),
        }
    
    def recommend_indexes(self, table_name: str) -> List[IndexRecommendation]:
        """Recommend indexes for a table based on common query patterns."""
        recommendations = []
        
        # Common patterns for Dean tables
        common_patterns = {
            "performance_metrics": [
                IndexRecommendation(
                    table_name="performance_metrics",
                    column_names=["ad_id", "date_start"],
                    index_type="btree",
                    reason="Common join and filter pattern",
                    estimated_improvement="High",
                    priority=9,
                ),
                IndexRecommendation(
                    table_name="performance_metrics",
                    column_names=["stage", "date_start"],
                    index_type="btree",
                    reason="Stage-based filtering with date range",
                    estimated_improvement="Medium",
                    priority=7,
                ),
            ],
            "ad_lifecycle": [
                IndexRecommendation(
                    table_name="ad_lifecycle",
                    column_names=["ad_id", "stage"],
                    index_type="btree",
                    reason="Primary lookup pattern",
                    estimated_improvement="High",
                    priority=10,
                ),
            ],
            "creative_intelligence": [
                IndexRecommendation(
                    table_name="creative_intelligence",
                    column_names=["creative_id"],
                    index_type="btree",
                    reason="Primary key lookup",
                    estimated_improvement="High",
                    priority=10,
                ),
                IndexRecommendation(
                    table_name="creative_intelligence",
                    column_names=["performance_score"],
                    index_type="btree",
                    reason="Performance-based queries",
                    estimated_improvement="Medium",
                    priority=6,
                ),
            ],
            # Removed: ml_predictions table recommendations (table no longer used)
            "creative_storage": [
                IndexRecommendation(
                    table_name="creative_storage",
                    column_names=["creative_id"],
                    index_type="btree",
                    reason="Primary lookup",
                    estimated_improvement="High",
                    priority=10,
                ),
                IndexRecommendation(
                    table_name="creative_storage",
                    column_names=["status", "last_used_at"],
                    index_type="btree",
                    reason="Cleanup queries",
                    estimated_improvement="Medium",
                    priority=7,
                ),
            ],
        }
        
        return recommendations + common_patterns.get(table_name, [])
    
    def archive_old_data(self, rule: ArchivalRule) -> int:
        """Archive or delete old data based on rule."""
        if not rule.enabled:
            return 0
        
        cutoff_date = datetime.now() - timedelta(days=rule.archive_after_days)
        
        try:
            if not self.client:
                logger.warning("No Supabase client available for archiving")
                return 0
            
            # Get old records
            old_records = self.client.table(rule.table_name).select("*").lt(
                rule.condition_column,
                cutoff_date.isoformat()
            ).limit(rule.batch_size).execute()
            
            if not old_records.data:
                return 0
            
            archived_count = 0
            
            # Archive or delete
            if rule.archive_to_table:
                # Move to archive table
                for record in old_records.data:
                    try:
                        self.client.table(rule.archive_to_table).insert(record).execute()
                        record_id = record.get("id")
                        if record_id:
                            self.client.table(rule.table_name).delete().eq("id", record_id).execute()
                        archived_count += 1
                    except Exception as e:
                        logger.error(f"Error archiving record {record.get('id')}: {e}")
            else:
                # Delete directly
                for record in old_records.data:
                    try:
                        record_id = record.get("id")
                        if record_id:
                            self.client.table(rule.table_name).delete().eq("id", record_id).execute()
                        archived_count += 1
                    except Exception as e:
                        logger.error(f"Error deleting record {record.get('id')}: {e}")
            
            logger.info(f"Archived {archived_count} records from {rule.table_name}")
            return archived_count
            
        except Exception as e:
            logger.error(f"Error archiving data from {rule.table_name}: {e}")
            return 0
    
    def run_automated_cleanup(self) -> Dict[str, int]:
        """Run all automated cleanup jobs."""
        results = {}
        
        for rule in self.archival_rules:
            if rule.enabled:
                count = self.archive_old_data(rule)
                results[rule.table_name] = count
        
        return results
    
    def optimize_queries(self, table_name: str) -> Dict[str, Any]:
        """Optimize queries for a specific table."""
        recommendations = self.recommend_indexes(table_name)
        
        # Create indexes (this would typically be done via SQL)
        created_indexes = []
        for rec in recommendations:
            if rec.priority >= 7:  # Only create high-priority indexes
                try:
                    logger.info(f"Recommend creating index on {rec.table_name}({','.join(rec.column_names)})")
                    created_indexes.append({
                        "table": rec.table_name,
                        "columns": rec.column_names,
                        "type": rec.index_type,
                        "reason": rec.reason,
                    })
                except Exception as e:
                    logger.error(f"Error creating index: {e}")
        
        return {
            "table": table_name,
            "recommendations": [{
                "columns": rec.column_names,
                "type": rec.index_type,
                "reason": rec.reason,
                "priority": rec.priority,
            } for rec in recommendations],
            "created_indexes": created_indexes,
            "optimized_at": datetime.now().isoformat(),
        }
    
    def vacuum_analyze(self, table_name: Optional[str] = None) -> Dict[str, Any]:
        """Run VACUUM ANALYZE on table(s) to optimize storage."""
        logger.info(f"VACUUM ANALYZE recommended for {table_name or 'all tables'}")
        
        return {
            "action": "vacuum_analyze",
            "table": table_name,
            "timestamp": datetime.now().isoformat(),
            "note": "Supabase handles VACUUM automatically, but manual optimization may be needed",
        }
    
    def get_table_statistics(self, table_name: str) -> Dict[str, Any]:
        """Get statistics about a table."""
        try:
            if not self.client:
                return {"error": "No Supabase client available"}
            
            # Get row count
            count_result = self.client.table(table_name).select("id", count="exact").limit(1).execute()
            row_count = count_result.count if hasattr(count_result, 'count') else 0
            
            # Get sample data
            sample = self.client.table(table_name).select("*").limit(10).execute()
            
            return {
                "table_name": table_name,
                "row_count": row_count,
                "sample_size": len(sample.data) if sample.data else 0,
                "analyzed_at": datetime.now().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting table statistics: {e}")
            return {"error": str(e)}
    
    def should_run_optimization(self) -> bool:
        """Check if optimization should run."""
        if not self.last_optimization:
            return True
        
        hours_since = (datetime.now() - self.last_optimization).total_seconds() / 3600
        return hours_since >= self.optimization_interval_hours
    
    def run_full_optimization(self) -> Dict[str, Any]:
        """Run full database optimization."""
        if not self.should_run_optimization():
            return {"skipped": True, "reason": "Too soon since last optimization"}
        
        results = {
            "optimization_started": datetime.now().isoformat(),
            "cleanup": {},
            "optimizations": {},
            "statistics": {},
        }
        
        # Run cleanup
        results["cleanup"] = self.run_automated_cleanup()
        
        # Optimize common tables
        tables_to_optimize = [
            "performance_metrics",
            "ad_lifecycle",
            "creative_intelligence",
            # Removed: ml_predictions (table no longer used)
            "creative_storage",
        ]
        
        for table in tables_to_optimize:
            try:
                results["optimizations"][table] = self.optimize_queries(table)
                results["statistics"][table] = self.get_table_statistics(table)
            except Exception as e:
                logger.error(f"Error optimizing {table}: {e}")
                results["optimizations"][table] = {"error": str(e)}
        
        self.last_optimization = datetime.now()
        results["optimization_completed"] = datetime.now().isoformat()
        
        return results


# =====================================================
# RESOURCE OPTIMIZATION
# =====================================================

class MemoryOptimizer:
    """Memory usage optimization."""
    
    def __init__(self, max_memory_mb: int = 2048, gc_threshold: tuple = (700, 10, 10)):
        self.max_memory_mb = max_memory_mb
        self.gc_threshold = gc_threshold
        self.process = psutil.Process(os.getpid()) if PSUTIL_AVAILABLE else None
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        if not self.process:
            return {"rss_mb": 0.0, "vms_mb": 0.0, "percent": 0.0}
        
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / (1024 * 1024),
            "vms_mb": memory_info.vms / (1024 * 1024),
            "percent": self.process.memory_percent(),
        }
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Optimize memory usage."""
        before = self.get_memory_usage()
        
        # Force garbage collection
        collected = gc.collect()
        
        # Set aggressive GC thresholds
        gc.set_threshold(*self.gc_threshold)
        
        after = self.get_memory_usage()
        
        freed_mb = before["rss_mb"] - after["rss_mb"]
        
        logger.info(f"Memory optimization: freed {freed_mb:.2f} MB, collected {collected} objects")
        
        return {
            "before_mb": before["rss_mb"],
            "after_mb": after["rss_mb"],
            "freed_mb": freed_mb,
            "objects_collected": collected,
            "optimized_at": datetime.now().isoformat(),
        }
    
    def should_optimize_memory(self) -> bool:
        """Check if memory optimization is needed."""
        usage = self.get_memory_usage()
        return usage["rss_mb"] > self.max_memory_mb or usage["percent"] > 80.0
    
    def clear_caches(self, cache_manager: Optional[Any] = None):
        """Clear caches to free memory."""
        if cache_manager:
            try:
                cache_manager.clear_namespace("performance")
                cache_manager.clear_namespace("api")
                logger.info("Cleared performance and API caches")
            except Exception as e:
                logger.warning(f"Failed to clear caches: {e}")


class CPUOptimizer:
    """CPU usage optimization."""
    
    def __init__(self):
        if PSUTIL_AVAILABLE:
            self.process = psutil.Process(os.getpid())
            self.cpu_count = psutil.cpu_count()
        else:
            self.process = None
            self.cpu_count = 1
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Get current CPU usage."""
        if not self.process:
            return {"process_percent": 0.0, "system_percent": 0.0, "cpu_count": 1}
        
        return {
            "process_percent": self.process.cpu_percent(interval=0.1),
            "system_percent": psutil.cpu_percent(interval=0.1),
            "cpu_count": self.cpu_count,
        }
    
    def should_throttle(self, threshold_percent: float = 80.0) -> bool:
        """Check if CPU usage is too high and should throttle."""
        usage = self.get_cpu_usage()
        return usage["system_percent"] > threshold_percent
    
    def optimize_cpu_usage(self) -> Dict[str, Any]:
        """Optimize CPU usage by adjusting priorities."""
        usage = self.get_cpu_usage()
        
        # Lower process priority if CPU is high
        if usage["system_percent"] > 90.0:
            try:
                if hasattr(os, "nice"):
                    os.nice(5)  # Increase nice value (lower priority)
                    logger.info(f"Lowered process priority: nice={os.nice(0)}")
            except (OSError, AttributeError):
                pass  # Not supported on this system
        
        return {
            "cpu_usage": usage,
            "optimized_at": datetime.now().isoformat(),
        }


class NetworkOptimizer:
    """Network request optimization."""
    
    def __init__(self):
        self.request_history: List[Dict[str, Any]] = []
        self.max_history = 1000
    
    def optimize_request(
        self,
        url: str,
        method: str = "GET",
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: float = 30.0,
    ) -> Dict[str, Any]:
        """Optimize network request parameters."""
        optimized_headers = headers or {}
        
        # Add keep-alive for connection reuse
        if "Connection" not in optimized_headers:
            optimized_headers["Connection"] = "keep-alive"
        
        # Add compression
        if "Accept-Encoding" not in optimized_headers:
            optimized_headers["Accept-Encoding"] = "gzip, deflate, br"
        
        # Optimize timeout based on request type
        if method == "GET":
            optimized_timeout = min(timeout, 15.0)
        else:
            optimized_timeout = timeout
        
        return {
            "url": url,
            "method": method,
            "params": params,
            "headers": optimized_headers,
            "timeout": optimized_timeout,
        }
    
    def batch_requests(self, requests: List[Dict[str, Any]], batch_size: int = 10) -> List[List[Dict[str, Any]]]:
        """Batch requests for efficiency."""
        batches = []
        for i in range(0, len(requests), batch_size):
            batches.append(requests[i:i + batch_size])
        return batches


class ImageOptimizer:
    """Image compression and optimization."""
    
    def __init__(
        self,
        max_width: int = 1920,
        max_height: int = 1920,
        quality: int = 85,
        format: str = "JPEG",
    ):
        self.max_width = max_width
        self.max_height = max_height
        self.quality = quality
        self.format = format
        self.pil_available = PIL_AVAILABLE
    
    def optimize_image(
        self,
        image_path: str,
        output_path: Optional[str] = None,
        max_size_mb: Optional[float] = None,
    ) -> Optional[str]:
        """Optimize image file size and dimensions."""
        if not self.pil_available:
            logger.warning("PIL not available, cannot optimize image")
            return None
        
        try:
            with Image.open(image_path) as img:
                original_size = os.path.getsize(image_path) / (1024 * 1024)
                
                # Resize if needed
                if img.width > self.max_width or img.height > self.max_height:
                    img.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
                
                # Convert to RGB if needed (for JPEG)
                if self.format == "JPEG" and img.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", img.size, (255, 255, 255))
                    if img.mode == "P":
                        img = img.convert("RGBA")
                    background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
                    img = background
                elif self.format == "JPEG" and img.mode != "RGB":
                    img = img.convert("RGB")
                
                # Save optimized image
                output = output_path or image_path
                
                # Adjust quality if size target specified
                current_quality = self.quality
                if max_size_mb:
                    for q in range(self.quality, 50, -5):
                        img.save(output, format=self.format, quality=q, optimize=True)
                        size_mb = os.path.getsize(output) / (1024 * 1024)
                        if size_mb <= max_size_mb:
                            current_quality = q
                            break
                
                img.save(output, format=self.format, quality=current_quality, optimize=True)
                
                final_size = os.path.getsize(output) / (1024 * 1024)
                compression_ratio = (1 - final_size / original_size) * 100 if original_size > 0 else 0
                
                logger.info(
                    f"Image optimized: {original_size:.2f} MB -> {final_size:.2f} MB "
                    f"({compression_ratio:.1f}% reduction)"
                )
                
                return output
                
        except Exception as e:
            logger.error(f"Error optimizing image {image_path}: {e}")
            return None
    
    def optimize_image_bytes(
        self,
        image_bytes: bytes,
        max_size_mb: Optional[float] = None,
    ) -> Optional[bytes]:
        """Optimize image from bytes."""
        if not self.pil_available:
            return None
        
        try:
            img = Image.open(io.BytesIO(image_bytes))
            
            # Resize if needed
            if img.width > self.max_width or img.height > self.max_height:
                img.thumbnail((self.max_width, self.max_height), Image.Resampling.LANCZOS)
            
            # Convert to RGB if needed
            if self.format == "JPEG" and img.mode != "RGB":
                background = Image.new("RGB", img.size, (255, 255, 255))
                if img.mode == "RGBA":
                    background.paste(img, mask=img.split()[-1])
                img = background
            
            # Save to bytes
            output = io.BytesIO()
            img.save(output, format=self.format, quality=self.quality, optimize=True)
            
            output_bytes = output.getvalue()
            size_mb = len(output_bytes) / (1024 * 1024)
            
            # Adjust quality if size target specified
            if max_size_mb and size_mb > max_size_mb:
                for q in range(self.quality, 50, -5):
                    output = io.BytesIO()
                    img.save(output, format=self.format, quality=q, optimize=True)
                    output_bytes = output.getvalue()
                    size_mb = len(output_bytes) / (1024 * 1024)
                    if size_mb <= max_size_mb:
                        break
            
            return output_bytes
            
        except Exception as e:
            logger.error(f"Error optimizing image bytes: {e}")
            return None


class ResourceOptimizer:
    """Unified resource optimizer."""
    
    def __init__(
        self,
        enable_memory_optimization: bool = True,
        enable_cpu_optimization: bool = True,
        enable_network_optimization: bool = True,
        enable_image_optimization: bool = True,
    ):
        self.memory_optimizer = MemoryOptimizer() if enable_memory_optimization else None
        self.cpu_optimizer = CPUOptimizer() if enable_cpu_optimization else None
        self.network_optimizer = NetworkOptimizer() if enable_network_optimization else None
        self.image_optimizer = ImageOptimizer() if enable_image_optimization else None
    
    def optimize_all(self, cache_manager: Optional[Any] = None) -> Dict[str, Any]:
        """Run all optimizations."""
        results = {
            "optimized_at": datetime.now().isoformat(),
            "memory": {},
            "cpu": {},
            "network": {},
            "image": {},
        }
        
        # Memory optimization
        if self.memory_optimizer and self.memory_optimizer.should_optimize_memory():
            results["memory"] = self.memory_optimizer.optimize_memory()
            if cache_manager:
                self.memory_optimizer.clear_caches(cache_manager)
        
        # CPU optimization
        if self.cpu_optimizer:
            results["cpu"] = self.cpu_optimizer.optimize_cpu_usage()
        
        # Network optimization
        if self.network_optimizer:
            results["network"] = {
                "optimization_enabled": True,
                "note": "Network optimization applied to requests automatically",
            }
        
        # Image optimization
        if self.image_optimizer:
            results["image"] = {
                "optimization_enabled": self.image_optimizer.pil_available,
                "note": "Image optimization available for use",
            }
        
        return results
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status."""
        status = {}
        
        if self.memory_optimizer:
            status["memory"] = self.memory_optimizer.get_memory_usage()
        
        if self.cpu_optimizer:
            status["cpu"] = self.cpu_optimizer.get_cpu_usage()
        
        return status
    
    def optimize_image_file(self, image_path: str, output_path: Optional[str] = None) -> Optional[str]:
        """Optimize an image file."""
        if self.image_optimizer:
            return self.image_optimizer.optimize_image(image_path, output_path)
        return None
    
    def optimize_image_bytes(self, image_bytes: bytes) -> Optional[bytes]:
        """Optimize image from bytes."""
        if self.image_optimizer:
            return self.image_optimizer.optimize_image_bytes(image_bytes)
        return None


# =====================================================
# FACTORY FUNCTIONS
# =====================================================

def create_db_optimizer(supabase_client) -> DatabaseOptimizer:
    """Factory function to create DatabaseOptimizer."""
    optimizer = DatabaseOptimizer(supabase_client)
    
    # Add default archival rules
    optimizer.add_archival_rule(ArchivalRule(
        table_name="historical_data",
        archive_after_days=90,
        archive_to_table=None,
        condition_column="created_at",
    ))
    
    # Removed: time_series_data and learning_events archival rules (tables no longer used)
    
    return optimizer


def create_resource_optimizer(**kwargs) -> ResourceOptimizer:
    """Factory function to create ResourceOptimizer."""
    return ResourceOptimizer(**kwargs)

