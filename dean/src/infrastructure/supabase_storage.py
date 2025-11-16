from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Union
import logging
import random
import os
import re
from pathlib import Path
from supabase import create_client

from .data_validation import (
    validate_supabase_data,
    validate_and_sanitize_data,
    ValidationError,
    ValidationResult,
    validate_all_timestamps,
)

logger = logging.getLogger(__name__)

def _sample_payload(payload: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(payload, dict):
        return None

    keys_of_interest = [
        "id",
        "ad_id",
        "creative_id",
        "lifecycle_id",
        "stage",
        "window_type",
        "date_start",
        "date_end",
        "model_type",
        "model_name",
    ]

    sample: Dict[str, Any] = {}
    for key in keys_of_interest:
        if key in payload:
            sample[key] = payload[key]

    for metric_key in ("ctr", "cpc", "cpm", "roas"):
        if metric_key in payload:
            sample[metric_key] = payload[metric_key]

    return sample or None


def _log_supabase_error(operation: str, table: str, error: Any, payload: Optional[Dict[str, Any]] = None) -> None:
    code: Optional[str] = None
    details: Optional[str] = None

    if isinstance(error, dict):
        code = error.get("code")
        details = error.get("details") or error.get("hint")
    elif hasattr(error, "args"):
        for arg in error.args:
            if isinstance(arg, dict):
                code = code or arg.get("code")
                details = details or arg.get("details") or arg.get("hint")

    message = str(error)
    suggestions: List[str] = []

    column_match = re.search(r"Could not find the '([^']+)' column", message)
    if column_match:
        missing_column = column_match.group(1)
        suggestions.append(
            f"Supabase schema cache is missing column '{missing_column}' on {table}. Apply the latest migrations or run `supabase db refresh schema`/`supabase db reset --linked`."
        )

    if "Field 'ctr' must be at most 1" in message:
        ctr_value = None
        if payload and isinstance(payload, dict):
            ctr_value = payload.get("ctr")
        if ctr_value is not None:
            suggestions.append(f"CTR must be a fraction <= 1. Offending value: {ctr_value}")
        suggestions.append("Ensure metrics store CTR as clicks/impressions (fraction).")

    if code == "PGRST204" and not suggestions:
        suggestions.append(
            "Supabase schema cache may be stale. Trigger a schema refresh or remove unused fields from the insert payload."
        )

    sample = _sample_payload(payload)

    logger.error(
        "SUPABASE ERROR [%s.%s] code=%s message=%s details=%s suggestions=%s sample=%s",
        table,
        operation,
        code or "unknown",
        message,
        details or "n/a",
        "; ".join(suggestions) or "n/a",
        sample,
    )
    
    if operation in ("insert", "insert_batch") and table != "insert_failures":
        _record_insert_failure(table, payload, message, code, details)


def _record_insert_failure(table: str, payload: Optional[Dict[str, Any]], error_message: str, error_code: Optional[str] = None, error_details: Optional[str] = None) -> None:
    try:
        error_table = os.getenv("SUPABASE_INSERT_ERROR_TABLE", "insert_failures")
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
        
        if not (url and key):
            logger.debug("Cannot record insert failure: Supabase credentials not available")
            return
        
        try:
            client = create_client(url, key)
        except Exception:
            logger.debug("Cannot record insert failure: Failed to create Supabase client")
            return
        
        full_message = error_message
        if error_code:
            full_message = f"[{error_code}] {full_message}"
        if error_details:
            full_message = f"{full_message} | Details: {error_details}"
        
        failure_record = {
            "table_name": table,
            "payload": payload or {},
            "error_message": full_message[:2000],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "retry_count": 0,
        }
        
        try:
            client.table(error_table).insert(failure_record).execute()
            logger.debug(f"✅ Recorded insert failure for {table} in {error_table}")
        except Exception as e:
            logger.debug(f"Failed to record insert failure in {error_table}: {e}")
    except Exception as e:
        logger.debug(f"Error recording insert failure: {e}")


class SupabaseStorage:
    def __init__(self, supabase_client: Any):
        self.client = supabase_client
    
    def record_ad_creation(self, ad_id: str, lifecycle_id: str, stage: str, 
                          created_at: Optional[datetime] = None) -> None:
        if created_at is None:
            created_at = datetime.now(timezone.utc)
        
        try:
            now = datetime.now(timezone.utc)
            if created_at > now:
                created_at = now
            min_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            if created_at < min_date:
                created_at = now - timedelta(days=random.randint(1, 30))
            data = {
                'created_at': created_at.isoformat(),
                'updated_at': now.isoformat(),
            }
            data = validate_all_timestamps(data)
            try:
                self.client.table('ads').update(data).eq('ad_id', ad_id).execute()
                logger.debug(f"Updated ad creation time for {ad_id} in ads table")
            except Exception as e:
                logger.debug(f"Ad {ad_id} not found in ads table yet (will be created later): {e}")
        except Exception as e:
            logger.error(f"Failed to record ad creation time for {ad_id}: {e}")
    
    def get_ad_creation_time(self, ad_id: str) -> Optional[datetime]:
        try:
            response = self.client.table('ads').select('created_at').eq(
                'ad_id', ad_id
            ).execute()
            
            if response.data and response.data[0].get('created_at'):
                created_at_str = response.data[0]['created_at']
                try:
                    return datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except:
                    pass
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get ad creation time for {ad_id}: {e}")
            return None
    
    def cleanup_invalid_creation_times(self) -> None:
        try:
            all_ads = self.client.table('ads').select('ad_id, created_at').execute()
            
            now = datetime.now(timezone.utc)
            min_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
            
            for record in (all_ads.data or []):
                ad_id = record.get('ad_id')
                created_at_str = record.get('created_at')
                
                if not created_at_str:
                    continue
                try:
                    created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                except:
                    continue
                if created_at < min_date or created_at > now:
                    self.record_ad_creation(
                        ad_id=ad_id,
                        lifecycle_id=record.get('lifecycle_id', ''),
                        stage=record.get('stage', 'asc_plus')
                    )
                    logger.info(f"Fixed invalid creation time for ad {ad_id}")
                    
        except Exception as e:
            logger.error(f"Failed to cleanup invalid creation times: {e}")
    
    def get_ad_age_days(self, ad_id: str) -> Optional[float]:
        creation_time = self.get_ad_creation_time(ad_id)
        if not creation_time:
            return None
        now = datetime.now(timezone.utc)
        age = now - creation_time
        return age.total_seconds() / 86400
    
    def store_historical_data(self, ad_id: str, lifecycle_id: str, stage: str, 
                             metric_name: str, metric_value: float, 
                             timestamp: Optional[datetime] = None) -> None:
        logger.debug(f"Historical data storage for {ad_id}:{metric_name} - now handled by performance_metrics table")
    
    def get_historical_data(self, ad_id: str, metric_name: str, 
                           since_days: int = 7) -> List[Dict[str, Any]]:
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=since_days)).date()
            metric_column_map = {
                'ctr': 'ctr',
                'cpc': 'cpc',
                'cpm': 'cpm',
                'roas': 'roas',
                'cpa': 'cpa',
                'atc_rate': 'atc_rate',
                'purchase_rate': 'purchase_rate',
                'spend': 'spend',
                'impressions': 'impressions',
                'clicks': 'clicks',
                'purchases': 'purchases',
                'add_to_cart': 'add_to_cart',
            }
            
            column_name = metric_column_map.get(metric_name, 'impressions')
            response = self.client.table('performance_metrics').select(
                'date_start', 'date_end', column_name
            ).eq('ad_id', ad_id).gte('date_start', cutoff_date.isoformat()).order(
                'date_start', desc=False
            ).execute()
            result = []
            for row in (response.data or []):
                value = row.get(column_name, 0)
                if value is not None:
                    date_start = row.get('date_start', '')
                    try:
                        ts_epoch = int(datetime.fromisoformat(date_start.replace('Z', '+00:00')).timestamp())
                    except:
                        ts_epoch = 0
                    result.append({
                        'metric_value': float(value),
                        'ts_iso': date_start,
                        'ts_epoch': ts_epoch
                    })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get historical data for {ad_id}: {e}")
            return []
    
    def get_historical_metrics(self, ad_id: str, metric_names: List[str], 
                              days_back: int = 30) -> Dict[str, List[Dict[str, Any]]]:
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_back)).date()
            metric_column_map = {
                'ctr': 'ctr', 'cpc': 'cpc', 'cpm': 'cpm', 'roas': 'roas', 'cpa': 'cpa',
                'atc_rate': 'atc_rate', 'purchase_rate': 'purchase_rate',
                'spend': 'spend', 'impressions': 'impressions', 'clicks': 'clicks',
                'purchases': 'purchases', 'add_to_cart': 'add_to_cart',
            }
            response = self.client.table('performance_metrics').select('*').eq(
                'ad_id', ad_id
            ).gte('date_start', cutoff_date.isoformat()).order('date_start', desc=False).execute()
            result = {name: [] for name in metric_names}
            for row in (response.data or []):
                for metric_name in metric_names:
                    column_name = metric_column_map.get(metric_name)
                    if column_name and column_name in row and row[column_name] is not None:
                        date_start = row.get('date_start', '')
                        try:
                            ts_epoch = int(datetime.fromisoformat(date_start.replace('Z', '+00:00')).timestamp())
                        except:
                            ts_epoch = 0
                        result[metric_name].append({
                            'metric_value': float(row[column_name]),
                            'ts_iso': date_start,
                            'ts_epoch': ts_epoch
                        })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get historical metrics for {ad_id}: {e}")
            return {name: [] for name in metric_names}
    
    def cleanup_old_historical_data(self, days_to_keep: int = 90) -> int:
        try:
            cutoff_date = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).date()
            response = self.client.table('performance_metrics').delete().lt(
                'date_start', cutoff_date.isoformat()
            ).execute()
            logger.info(f"Cleaned up performance metrics older than {days_to_keep} days")
            return 0
            
        except Exception as e:
            logger.error(f"Failed to cleanup old performance metrics: {e}")
            return 0


def create_supabase_storage(supabase_client: Any) -> SupabaseStorage:
    return SupabaseStorage(supabase_client)

FRACTION_FIELDS = {
    "ctr",
    "unique_ctr",
    "atc_rate",
    "ic_rate",
    "purchase_rate",
    "atc_to_ic_rate",
    "ic_to_purchase_rate",
    "engagement_rate",
    "conversion_rate",
    "avg_ctr",
}


def _normalize_fraction_fields(data: Optional[Dict[str, Any]]) -> None:
    if not isinstance(data, dict):
        return

    for field in FRACTION_FIELDS:
        if field not in data or data[field] is None:
            continue
        try:
            value = float(data[field])
        except (TypeError, ValueError):
            continue

        if value < 0:
            data[field] = 0.0
        elif value > 1:
            if value <= 100:
                data[field] = min(1.0, round(value / 100.0, 6))
            else:
                data[field] = 1.0


class ValidatedSupabaseClient:
    def __init__(self, url: str, key: str, enable_validation: bool = True):
        self.client = create_client(url, key)
        self.enable_validation = enable_validation
        if enable_validation:
            logger.info("✅ Data validation enabled for all Supabase operations")
        else:
            logger.warning("⚠️ Data validation disabled - invalid data may be stored")
    
    def insert(self, table: str, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
               validate: bool = None) -> Any:
        should_validate = validate if validate is not None else self.enable_validation
        
        if should_validate:
            if isinstance(data, list):
                return self._insert_batch_validated(table, data)
            else:
                return self._insert_single_validated(table, data)
        else:
            return self.client.table(table).insert(data).execute()
    
    def upsert(self, table: str, data: Union[Dict[str, Any], List[Dict[str, Any]]], 
               on_conflict: str = None, validate: bool = None) -> Any:
        should_validate = validate if validate is not None else self.enable_validation
        
        if should_validate:
            if isinstance(data, list):
                return self._upsert_batch_validated(table, data, on_conflict)
            else:
                return self._upsert_single_validated(table, data, on_conflict)
        else:
            if on_conflict:
                query = self.client.table(table).upsert(data, on_conflict=on_conflict)
            else:
                query = self.client.table(table).upsert(data)
            return query.execute()
    
    def update(self, table: str, data: Dict[str, Any], 
               validate: bool = None, **kwargs) -> Any:
        should_validate = validate if validate is not None else self.enable_validation
        if should_validate:
            sanitized_data = self._validate_and_sanitize(table, data)
            query = self.client.table(table).update(sanitized_data)
            for key, value in kwargs.items():
                if key == 'eq' and 'value' in kwargs:
                    query = query.eq(value, kwargs['value'])
                elif key == 'eq2' and 'value2' in kwargs:
                    query = query.eq(value, kwargs['value2'])
                elif key == 'eq3' and 'value3' in kwargs:
                    query = query.eq(value, kwargs['value3'])
                elif key.startswith('value'):
                    continue
                elif hasattr(query, key):
                    query = getattr(query, key)(value)
            
            return query.execute()
        else:
            query = self.client.table(table).update(data)
            for key, value in kwargs.items():
                if key == 'eq' and 'value' in kwargs:
                    query = query.eq(value, kwargs['value'])
                elif key == 'eq2' and 'value2' in kwargs:
                    query = query.eq(value, kwargs['value2'])
                elif key == 'eq3' and 'value3' in kwargs:
                    query = query.eq(value, kwargs['value3'])
                elif key.startswith('value'):
                    continue
                elif hasattr(query, key):
                    query = getattr(query, key)(value)
            
            return query.execute()
    
    def _insert_single_validated(self, table: str, data: Dict[str, Any]) -> Any:
        _normalize_fraction_fields(data)
        try:
            sanitized_data = validate_and_sanitize_data(table, data)
            logger.debug(f"✅ Data validated for {table} insert")
            return self.client.table(table).insert(sanitized_data).execute()
        except ValidationError as e:
            _log_supabase_error("validation", table, e, data)
            raise e
        except Exception as e:
            _log_supabase_error("insert", table, e, data)
            raise e
    
    def _insert_batch_validated(self, table: str, data_list: List[Dict[str, Any]]) -> Any:
        validated_records = []
        failed_records = []
        
        for i, data in enumerate(data_list):
            _normalize_fraction_fields(data)
            try:
                sanitized_data = validate_and_sanitize_data(table, data)
                validated_records.append(sanitized_data)
            except ValidationError as e:
                logger.error(f"❌ Validation failed for {table} insert item {i}: {e}")
                failed_records.append((i, str(e)))
        
        if failed_records:
            logger.error(f"❌ {len(failed_records)} records failed validation for {table}")
            if len(validated_records) == 0:
                raise ValidationError(f"All records failed validation for {table}")
        
        if validated_records:
            logger.info(f"✅ Inserting {len(validated_records)} validated records into {table}")
            try:
                return self.client.table(table).insert(validated_records).execute()
            except Exception as e:
                sample = validated_records[0] if validated_records else None
                _log_supabase_error("insert_batch", table, e, sample)
                for record in validated_records:
                    _record_insert_failure(table, record, str(e))
                raise e
        else:
            return None
    
    def _upsert_single_validated(self, table: str, data: Dict[str, Any], on_conflict: str = None) -> Any:
        _normalize_fraction_fields(data)
        try:
            sanitized_data = validate_and_sanitize_data(table, data)
            logger.debug(f"✅ Data validated for {table} upsert")
            
            if on_conflict:
                query = self.client.table(table).upsert(sanitized_data, on_conflict=on_conflict)
            else:
                query = self.client.table(table).upsert(sanitized_data)
            
            return query.execute()
        except ValidationError as e:
            _log_supabase_error("validation", table, e, data)
            raise e
        except Exception as e:
            _log_supabase_error("upsert", table, e, data)
            raise e
    
    def _upsert_batch_validated(self, table: str, data_list: List[Dict[str, Any]], on_conflict: str = None) -> Any:
        validated_records = []
        failed_records = []
        
        for i, data in enumerate(data_list):
            _normalize_fraction_fields(data)
            try:
                sanitized_data = validate_and_sanitize_data(table, data)
                validated_records.append(sanitized_data)
            except ValidationError as e:
                logger.error(f"❌ Validation failed for {table} upsert item {i}: {e}")
                failed_records.append((i, str(e)))
        
        if failed_records:
            logger.error(f"❌ {len(failed_records)} records failed validation for {table}")
            if len(validated_records) == 0:
                raise ValidationError(f"All records failed validation for {table}")
        
        if validated_records:
            logger.info(f"✅ Upserting {len(validated_records)} validated records into {table}")

            try:
                if on_conflict:
                    query = self.client.table(table).upsert(validated_records, on_conflict=on_conflict)
                else:
                    query = self.client.table(table).upsert(validated_records)

                return query.execute()
            except Exception as e:
                sample = validated_records[0] if validated_records else None
                _log_supabase_error("upsert_batch", table, e, sample)
                raise e
        else:
            return None
    
    def _validate_and_sanitize(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        return validate_and_sanitize_data(table, data)
    
    def validate_data(self, table: str, data: Dict[str, Any]) -> ValidationResult:
        return validate_supabase_data(table, data, strict_mode=False)
    
    def validate_batch_data(self, table: str, data_list: List[Dict[str, Any]]) -> List[ValidationResult]:
        results = []
        for data in data_list:
            result = validate_supabase_data(table, data, strict_mode=False)
            results.append(result)
        return results
    
    def select(self, table: str, *args, **kwargs):
        return self.client.table(table).select(*args, **kwargs)
    
    def delete(self, table: str, **kwargs):
        query = self.client.table(table).delete()
        for key, value in kwargs.items():
            if hasattr(query, key):
                query = getattr(query, key)(value)
        return query.execute()
    
    def rpc(self, function_name: str, params: Dict[str, Any] = None):
        return self.client.rpc(function_name, params)
    
    def table(self, table_name: str):
        return self.client.table(table_name)
    
    def __getattr__(self, name):
        return getattr(self.client, name)


def create_validated_supabase_client(url: str, key: str, enable_validation: bool = True) -> ValidatedSupabaseClient:
    return ValidatedSupabaseClient(url, key, enable_validation)


def get_validated_supabase_client(enable_validation: bool = True) -> Optional[ValidatedSupabaseClient]:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
    if not (url and key):
        logger.error("❌ Supabase credentials not found. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY")
        return None
    try:
        return create_validated_supabase_client(url, key, enable_validation)
    except Exception as e:
        logger.error(f"❌ Failed to create validated Supabase client: {e}")
        return None


CREATIVE_STORAGE_BUCKET = os.getenv("CREATIVE_STORAGE_BUCKET", "creatives")
CREATIVE_UNUSED_DAYS = int(os.getenv("CREATIVE_UNUSED_DAYS", "30"))
CREATIVE_KILLED_DAYS = int(os.getenv("CREATIVE_KILLED_DAYS", "7"))


class CreativeStorageManager:
    def __init__(self, supabase_client):
        self.client = supabase_client
        self.bucket_name = CREATIVE_STORAGE_BUCKET
        
    def upload_creative(
        self,
        creative_id: str,
        image_path: str,
        ad_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        try:
            if not self.client:
                logger.error("Supabase client not available")
                return None
            
            image_path_obj = Path(image_path)
            if not image_path_obj.exists():
                logger.error(f"Image file not found: {image_path}")
                return None
            
            file_size = image_path_obj.stat().st_size
            file_ext = image_path_obj.suffix.lower() or ".png"
            
            mime_types = {
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
            }
            file_type = mime_types.get(file_ext, "image/png")
            
            storage_path = f"{creative_id}{file_ext}"
            
            with open(image_path, "rb") as f:
                file_data = f.read()
            
            try:
                response = self.client.storage.from_(self.bucket_name).upload(
                    storage_path,
                    file_data,
                    file_options={"content-type": file_type, "upsert": "true"},
                )
            except Exception as e:
                logger.warning(f"Upsert upload failed, trying regular upload: {e}")
                response = self.client.storage.from_(self.bucket_name).upload(
                    storage_path,
                    file_data,
                    file_options={"content-type": file_type},
                )
            
            if not response:
                logger.error(f"Failed to upload creative {creative_id} to storage")
                return None
            
            try:
                public_url_response = self.client.storage.from_(self.bucket_name).get_public_url(storage_path)
                if isinstance(public_url_response, dict):
                    public_url = public_url_response.get("publicUrl") or public_url_response.get("url")
                else:
                    public_url = public_url_response
            except Exception as e:
                supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
                if supabase_url:
                    public_url = f"{supabase_url}/storage/v1/object/public/{self.bucket_name}/{storage_path}"
                else:
                    logger.error("Cannot construct public URL: SUPABASE_URL not set")
                    return None
            
            if not public_url:
                logger.error(f"Failed to get public URL for creative {creative_id}")
                return None
            
            if ad_id:
                enhanced_metadata = metadata or {}
                
                if "format" not in enhanced_metadata:
                    enhanced_metadata["format"] = "static_image"
                if "style" not in enhanced_metadata:
                    enhanced_metadata["style"] = ""
                if "message_type" not in enhanced_metadata:
                    enhanced_metadata["message_type"] = ""
                if "target_motivation" not in enhanced_metadata:
                    enhanced_metadata["target_motivation"] = ""
                
                storage_data = {
                    "creative_id": creative_id,
                    "storage_path": storage_path,
                    "storage_url": public_url,
                    "file_size_bytes": file_size,
                    "file_type": file_type,
                    "status": "active",
                    "metadata": enhanced_metadata,
                }
                
                try:
                    self.client.table("ads").update(storage_data).eq("ad_id", ad_id).execute()
                    logger.debug(f"Updated ad {ad_id} with storage info for creative {creative_id}")
                except Exception as e:
                    logger.debug(f"Ad {ad_id} not found yet (will be created later): {e}")
            
            logger.info(f"✅ Uploaded creative {creative_id} to Supabase Storage")
            return public_url
                
        except Exception as e:
            logger.error(f"Error uploading creative {creative_id}: {e}", exc_info=True)
            from integrations.slack import notify
            notify(f"❌ Failed to upload creative {creative_id} to storage: {e}")
            return None
    
    def mark_creative_killed(self, creative_id: str) -> bool:
        try:
            if not self.client:
                return False
            
            now = datetime.now(timezone.utc).isoformat()
            self.client.table("ads").update({
                "status": "killed",
                "killed_at": now,
                "updated_at": now,
            }).eq("creative_id", creative_id).execute()
            
            logger.info(f"✅ Marked ads with creative {creative_id} as killed")
            return True
            
        except Exception as e:
            logger.error(f"Error marking creative {creative_id} as killed: {e}")
            return False
    
    def get_queued_creative(self) -> Optional[Dict[str, Any]]:
        return None
    
    def get_queued_creative_count(self) -> int:
        return 0
    
    def should_pre_generate_creatives(self, target_count: int = 10, buffer_size: int = 3) -> bool:
        queued_count = self.get_queued_creative_count()
        return queued_count < buffer_size
    
    def get_recently_created_creative(self, minutes_back: int = 5) -> Optional[Dict[str, Any]]:
        try:
            if not self.client:
                return None
            
            cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes_back)
            
            result = self.client.table("ads").select(
                "ad_id, creative_id, storage_url, storage_path, metadata, status, created_at"
            ).gte("created_at", cutoff_time.isoformat()).order("created_at", desc=True).limit(1).execute()
            
            if result.data and len(result.data) > 0:
                ad = result.data[0]
                return {
                    "creative_id": ad.get("creative_id"),
                    "storage_url": ad.get("storage_url"),
                    "storage_path": ad.get("storage_path"),
                    "metadata": ad.get("metadata", {}),
                    "status": ad.get("status"),
                    "created_at": ad.get("created_at"),
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting recently created creative: {e}")
            return None
    
    def mark_creative_active(self, creative_id: str, ad_id: Optional[str] = None) -> bool:
        try:
            if not self.client or not ad_id:
                return False
            
            now = datetime.now(timezone.utc).isoformat()
            self.client.table("ads").update({
                "status": "active",
                "updated_at": now,
            }).eq("ad_id", ad_id).eq("creative_id", creative_id).execute()
            
            logger.info(f"✅ Marked ad {ad_id} (creative {creative_id}) as active")
            return True
            
        except Exception as e:
            logger.error(f"Error marking creative {creative_id} as active: {e}")
            return False
    
    def update_usage(self, creative_id: str) -> bool:
        try:
            if not self.client:
                return False
            
            now = datetime.now(timezone.utc).isoformat()
            self.client.table("ads").update({
                "status": "active",
                "updated_at": now,
            }).eq("creative_id", creative_id).execute()
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating usage for creative {creative_id}: {e}")
            return False
    
    def cleanup_unused_creatives(self) -> int:
        try:
            if not self.client:
                return 0
            return 0
        except Exception as e:
            logger.error(f"Error cleaning up unused creatives: {e}", exc_info=True)
            return 0
    
    def cleanup_killed_creatives(self) -> int:
        try:
            if not self.client:
                return 0
            
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=CREATIVE_KILLED_DAYS)
            
            result = self.client.table("ads").select(
                "ad_id, creative_id, storage_path"
            ).eq("status", "killed").lt(
                "killed_at", cutoff_date.isoformat()
            ).execute()
            
            deleted_count = 0
            
            for record in (result.data or []):
                creative_id = record.get("creative_id")
                storage_path = record.get("storage_path")
                
                try:
                    try:
                        if storage_path:
                            self.client.storage.from_(self.bucket_name).remove([storage_path])
                    except Exception as storage_error:
                        logger.debug(f"Storage file not found (may already be deleted): {storage_error}")
                    
                    self.client.table("ads").delete().eq(
                        "ad_id", record.get("ad_id")
                    ).execute()
                    
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted killed ad/creative {creative_id} from storage")
                    
                except Exception as e:
                    logger.error(f"Error deleting killed creative {creative_id}: {e}")
            
            if deleted_count > 0:
                from integrations.slack import notify
                notify(f"🧹 Cleaned up {deleted_count} killed creatives from storage")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up killed creatives: {e}", exc_info=True)
            return 0
    
    def mark_creative_unused(self, creative_id: str) -> bool:
        return False
    
    def get_creative_url(self, creative_id: str) -> Optional[str]:
        try:
            if not self.client:
                return None
            
            result = self.client.table("ads").select(
                "storage_url, storage_path"
            ).eq("creative_id", creative_id).limit(1).execute()
            
            if result.data and len(result.data) > 0:
                return result.data[0].get("storage_url")
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting creative URL for {creative_id}: {e}")
            return None


def create_creative_storage_manager(supabase_client) -> Optional[CreativeStorageManager]:
    if not supabase_client:
        return None
    return CreativeStorageManager(supabase_client)
