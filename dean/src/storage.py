from __future__ import annotations

import json
import os
import random
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Literal, Optional, Tuple

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

# NEW: tz support for account-local counters
try:
    import pytz
except Exception:  # pragma: no cover
    pytz = None  # type: ignore

UTC = timezone.utc
ACCOUNT_TZ_NAME = os.getenv("ACCOUNT_TZ") or os.getenv("ACCOUNT_TIMEZONE") or "Europe/Amsterdam"


def _get_tz(name: Optional[str] = None):
    if pytz is None:
        return None
    try:
        return pytz.timezone((name or ACCOUNT_TZ_NAME) or "Europe/Amsterdam")
    except Exception:
        return pytz.timezone("Europe/Amsterdam")


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _now_account(tz_name: Optional[str] = None) -> datetime:
    tz = _get_tz(tz_name)
    if tz is None:
        return _now_utc()
    return datetime.now(tz)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _epoch(dt: datetime) -> int:
    return int(dt.timestamp())


def _epoch_now() -> int:
    return _epoch(_now_utc())


def _jitter(base: float) -> float:
    return base * (0.8 + 0.4 * random.random())


def _to_json(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _from_json(s: Optional[str]) -> Optional[Any]:
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _seconds_until_local_midnight(tz_name: Optional[str] = None) -> int:
    """
    TTL helper: seconds from *now* until next local midnight in the given tz (default: account tz).
    Falls back to ~24h if pytz is unavailable.
    """
    tz = _get_tz(tz_name)
    if tz is None:
        return 24 * 3600
    now = _now_account(tz_name)
    # local midnight of "tomorrow"
    tomorrow = (now + timedelta(days=1)).date()
    local_midnight = tz.localize(datetime(tomorrow.year, tomorrow.month, tomorrow.day, 0, 0, 0))
    return max(1, int((local_midnight - now).total_seconds()))


try:
    from prometheus_client import Counter, Histogram  # type: ignore

    _prom_enabled = True
    DB_OPS = Counter("store_db_ops_total", "DB operations", ["op"])
    DB_ERRORS = Counter("store_db_errors_total", "DB errors", ["op"])
    DB_LAT = Histogram("store_db_latency_seconds", "DB latencies", ["op"])
except Exception:  # pragma: no cover
    _prom_enabled = False

    class _N:
        def labels(self, *_, **__): return self
        def inc(self, *_): pass
        def observe(self, *_): pass
    DB_OPS = DB_ERRORS = DB_LAT = _N()  # type: ignore


def _retry_sql(retries: int = 5, base_sleep: float = 0.03, max_sleep: float = 0.5) -> Callable:
    def deco(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            cb = getattr(self, "_cb", None)
            if cb is None:
                cb = defaultdict(lambda: {"n": 0, "open_until": 0.0})
                setattr(self, "_cb", cb)
            state = cb[fn.__name__]
            now = time.time()
            if state["open_until"] and now < state["open_until"]:
                raise OperationalError("circuit_open", None, None)
            last_exc: Optional[Exception] = None
            for i in range(retries):
                t0 = time.perf_counter()
                try:
                    DB_OPS.labels(fn.__name__).inc()
                    out = fn(self, *args, **kwargs)
                    DB_LAT.labels(fn.__name__).observe(time.perf_counter() - t0)
                    state["n"] = 0
                    return out
                except OperationalError as e:
                    last_exc = e
                    DB_ERRORS.labels(fn.__name__).inc()
                    time.sleep(min(max_sleep, _jitter(base_sleep * (2 ** i))))
                except Exception:
                    DB_ERRORS.labels(fn.__name__).inc()
                    raise
            state["n"] += 1
            if state["n"] >= 3:
                state["open_until"] = time.time() + 2.0
            assert last_exc is not None
            raise last_exc
        return wrapper
    return deco


@dataclass(frozen=True)
class ActionRecord:
    id: str
    ts_iso: str
    ts_epoch: int
    entity_type: str
    entity_id: str
    action: str
    level: Literal["info", "warn", "error"]
    stage: Optional[str]
    rule_type: Optional[str]
    reason: Optional[str]
    thresholds: Optional[Dict[str, Any]]
    observed: Optional[Dict[str, Any]]
    window: Optional[str]
    attribution: Optional[str]
    actor: Optional[str]
    meta: Optional[Dict[str, Any]]
    dedup_key: Optional[str]
    deleted: int


class Store:
    SCHEMA_VERSION = 5

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self.eng: Engine = create_engine(
            f"sqlite:///{path}",
            connect_args={"check_same_thread": False, "timeout": 30},
            pool_pre_ping=True,
        )
        self._cb: defaultdict = defaultdict(lambda: {"n": 0, "open_until": 0.0})
        self._init_db()

    def close(self) -> None:
        try:
            self.eng.dispose()
        except Exception:
            pass

    def _init_db(self) -> None:
        with self.eng.begin() as c:
            c.exec_driver_sql("PRAGMA journal_mode=WAL;")
            c.exec_driver_sql("PRAGMA synchronous=NORMAL;")
            c.exec_driver_sql("PRAGMA temp_store=MEMORY;")
            c.exec_driver_sql("PRAGMA foreign_keys=ON;")
            c.exec_driver_sql("PRAGMA busy_timeout=30000;")
            c.exec_driver_sql("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL);")
            cur = c.execute(text("SELECT version FROM schema_version")).fetchone()
            if not cur:
                c.execute(text("INSERT INTO schema_version(version) VALUES (:v)"), {"v": self.SCHEMA_VERSION})
            else:
                current = int(cur[0])
                if current < self.SCHEMA_VERSION:
                    self._migrate(c, current, self.SCHEMA_VERSION)
            c.exec_driver_sql("""
              CREATE TABLE IF NOT EXISTS actions(
                id TEXT PRIMARY KEY,
                ts_iso TEXT NOT NULL,
                ts_epoch INTEGER NOT NULL,
                entity_type TEXT NOT NULL,
                entity_id   TEXT NOT NULL,
                action TEXT NOT NULL,
                level  TEXT NOT NULL CHECK(level IN ('info','warn','error')),
                stage  TEXT,
                rule_type TEXT,
                reason   TEXT,
                thresholds TEXT,
                observed   TEXT,
                window     TEXT,
                attribution TEXT,
                actor TEXT,
                meta  TEXT,
                dedup_key TEXT UNIQUE,
                deleted INTEGER NOT NULL DEFAULT 0
              );
            """)
            c.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_actions_time ON actions(ts_epoch);")
            c.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_actions_entity_time ON actions(entity_type, entity_id, ts_epoch DESC);")
            c.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_actions_stage_action ON actions(stage, action, ts_epoch DESC);")
            c.exec_driver_sql("""
              CREATE TABLE IF NOT EXISTS counters(
                ns TEXT NOT NULL,
                key TEXT NOT NULL,
                val INTEGER NOT NULL,
                ttl_epoch INTEGER,
                updated_epoch INTEGER NOT NULL,
                PRIMARY KEY(ns, key)
              );
            """)
            c.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_counters_ttl ON counters(ttl_epoch);")
            c.exec_driver_sql("""
              CREATE TABLE IF NOT EXISTS flags(
                entity_type TEXT NOT NULL,
                entity_id   TEXT NOT NULL,
                k TEXT NOT NULL,
                v TEXT NOT NULL,
                ttl_epoch INTEGER,
                updated_epoch INTEGER NOT NULL,
                PRIMARY KEY(entity_type, entity_id, k)
              );
            """)
            c.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_flags_ttl ON flags(ttl_epoch);")

    def _migrate(self, conn, current: int, target: int) -> None:
        for v in range(current + 1, target + 1):
            if v == 4:
                try:
                    conn.exec_driver_sql("ALTER TABLE actions ADD COLUMN dedup_key TEXT;")
                    conn.exec_driver_sql("CREATE UNIQUE INDEX IF NOT EXISTS idx_actions_dedup ON actions(dedup_key);")
                except Exception:
                    pass
            if v == 5:
                try:
                    conn.exec_driver_sql("ALTER TABLE actions ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0;")
                except Exception:
                    pass
            conn.execute(text("UPDATE schema_version SET version=:v"), {"v": v})

    @contextmanager
    def _begin(self):
        with self.eng.begin() as conn:
            yield conn

    @_retry_sql()
    def log(
        self,
        *,
        entity_type: str,
        entity_id: str,
        action: str,
        level: Literal["info", "warn", "error"] = "info",
        stage: Optional[str] = None,
        rule_type: Optional[str] = None,
        reason: Optional[str] = None,
        thresholds: Optional[Dict[str, Any]] = None,
        observed: Optional[Dict[str, Any]] = None,
        window: Optional[str] = None,
        attribution: Optional[str] = None,
        actor: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
        ts: Optional[datetime] = None,
        dedup_key: Optional[str] = None,
        soft_delete: bool = False,
    ) -> str:
        now = ts or _now_utc()
        aid = str(uuid.uuid7()) if hasattr(uuid, "uuid7") else str(uuid.uuid4())
        with self._begin() as c:
            sql = """
              INSERT INTO actions
              (id, ts_iso, ts_epoch, entity_type, entity_id, action, level, stage, rule_type, reason,
               thresholds, observed, window, attribution, actor, meta, dedup_key, deleted)
              VALUES
              (:id,:iso,:ep,:et,:eid,:act,:lvl,:stg,:rtype,:rea,:thr,:obs,:win,:att,:actr,:meta,:dedup,:del)
              ON CONFLICT(dedup_key) DO NOTHING
            """
            c.execute(text(sql), {
                "id": aid,
                "iso": _iso(now),
                "ep": _epoch(now),
                "et": entity_type,
                "eid": entity_id,
                "act": action,
                "lvl": level,
                "stg": stage,
                "rtype": rule_type,
                "rea": reason,
                "thr": _to_json(thresholds),
                "obs": _to_json(observed),
                "win": window,
                "att": attribution,
                "actr": actor,
                "meta": _to_json(meta),
                "dedup": dedup_key,
                "del": 1 if soft_delete else 0,
            })
        return aid

    @_retry_sql()
    def batch_log(self, records: Iterable[Dict[str, Any]]) -> int:
        rows = list(records)
        if not rows:
            return 0
        now = _now_utc()
        with self._begin() as c:
            sql = """
              INSERT INTO actions
              (id, ts_iso, ts_epoch, entity_type, entity_id, action, level, stage, rule_type, reason,
               thresholds, observed, window, attribution, actor, meta, dedup_key, deleted)
              VALUES
              (:id,:iso,:ep,:et,:eid,:act,:lvl,:stg,:rtype,:rea,:thr,:obs,:win,:att,:actr,:meta,:dedup,:del)
              ON CONFLICT(dedup_key) DO NOTHING
            """
            payload: List[Dict[str, Any]] = []
            for r in rows:
                aid = r.get("id") or (str(uuid.uuid7()) if hasattr(uuid, "uuid7") else str(uuid.uuid4()))
                ts = r.get("ts") or now
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(UTC)
                payload.append({
                    "id": aid,
                    "iso": _iso(ts),
                    "ep": _epoch(ts),
                    "et": r["entity_type"],
                    "eid": r["entity_id"],
                    "act": r["action"],
                    "lvl": r.get("level", "info"),
                    "stg": r.get("stage"),
                    "rtype": r.get("rule_type"),
                    "rea": r.get("reason"),
                    "thr": _to_json(r.get("thresholds")),
                    "obs": _to_json(r.get("observed")),
                    "win": r.get("window"),
                    "att": r.get("attribution"),
                    "actr": r.get("actor"),
                    "meta": _to_json(r.get("meta")),
                    "dedup": r.get("dedup_key"),
                    "del": 1 if r.get("deleted") else 0,
                })
            c.execute(text(sql), payload)
        return len(payload)

    @_retry_sql()
    def soft_delete_action(self, action_id: str) -> None:
        with self._begin() as c:
            c.execute(text("UPDATE actions SET deleted=1 WHERE id=:id"), {"id": action_id})

    @_retry_sql()
    def get_actions(
        self,
        *,
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        stage: Optional[str] = None,
        action: Optional[str] = None,
        level: Optional[Literal["info", "warn", "error"]] = None,
        since_epoch: Optional[int] = None,
        until_epoch: Optional[int] = None,
        include_deleted: bool = False,
        limit: int = 200,
        cursor_after_epoch: Optional[int] = None,
        cursor_after_id: Optional[str] = None,
    ) -> List[ActionRecord]:
        where: List[str] = []
        params: Dict[str, Any] = {"lim": max(1, min(5000, int(limit)))}
        if not include_deleted:
            where.append("deleted=0")
        if entity_type:
            where.append("entity_type=:et"); params["et"] = entity_type
        if entity_id:
            where.append("entity_id=:eid"); params["eid"] = entity_id
        if stage:
            where.append("stage=:stg"); params["stg"] = stage
        if action:
            where.append("action=:act"); params["act"] = action
        if level:
            where.append("level=:lvl"); params["lvl"] = level
        if since_epoch is not None:
            where.append("ts_epoch>=:since"); params["since"] = int(since_epoch)
        if until_epoch is not None:
            where.append("ts_epoch<=:until"); params["until"] = int(until_epoch)
        if cursor_after_epoch is not None:
            where.append("(ts_epoch<:cae OR (ts_epoch=:cae AND id<:caid))")
            params["cae"] = int(cursor_after_epoch)
            params["caid"] = cursor_after_id or ""
        sql = "SELECT * FROM actions"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY ts_epoch DESC, id DESC LIMIT :lim"
        with self._begin() as c:
            rows = c.execute(text(sql), params).fetchall()
        out: List[ActionRecord] = []
        for r in rows:
            out.append(ActionRecord(
                id=r.id, ts_iso=r.ts_iso, ts_epoch=int(r.ts_epoch),
                entity_type=r.entity_type, entity_id=r.entity_id, action=r.action,
                level=r.level, stage=r.stage, rule_type=r.rule_type, reason=r.reason,
                thresholds=_from_json(r.thresholds), observed=_from_json(r.observed),
                window=r.window, attribution=r.attribution, actor=r.actor,
                meta=_from_json(r.meta), dedup_key=r.dedup_key, deleted=int(r.deleted),
            ))
        return out

    @_retry_sql()
    def get_last_action(self, entity_id: str, action: str) -> Optional[ActionRecord]:
        with self._begin() as c:
            r = c.execute(text("""
                SELECT * FROM actions
                WHERE entity_id=:eid AND action=:act AND deleted=0
                ORDER BY ts_epoch DESC, id DESC LIMIT 1
            """), {"eid": entity_id, "act": action}).fetchone()
        if not r:
            return None
        return ActionRecord(
            id=r.id, ts_iso=r.ts_iso, ts_epoch=int(r.ts_epoch),
            entity_type=r.entity_type, entity_id=r.entity_id, action=r.action,
            level=r.level, stage=r.stage, rule_type=r.rule_type, reason=r.reason,
            thresholds=_from_json(r.thresholds), observed=_from_json(r.observed),
            window=r.window, attribution=r.attribution, actor=r.actor,
            meta=_from_json(r.meta), dedup_key=r.dedup_key, deleted=int(r.deleted),
        )

    @_retry_sql()
    def export_actions_jsonl(self, path: str, since_epoch: Optional[int] = None) -> int:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        count = 0
        cursor_epoch: Optional[int] = None
        cursor_id: Optional[str] = None
        with open(path, "a", encoding="utf-8") as f:
            while True:
                batch = self.get_actions(
                    since_epoch=since_epoch,
                    cursor_after_epoch=cursor_epoch,
                    cursor_after_id=cursor_id,
                    limit=2000,
                )
                if not batch:
                    break
                for r in batch:
                    f.write(json.dumps(asdict(r), separators=(",", ":"), ensure_ascii=False) + "\n")
                    count += 1
                cursor_epoch = batch[-1].ts_epoch
                cursor_id = batch[-1].id
        return count

    @_retry_sql()
    def prune_actions_older_than(self, days: int) -> int:
        cutoff = _epoch(_now_utc() - timedelta(days=days))
        with self._begin() as c:
            cur = c.execute(text("DELETE FROM actions WHERE ts_epoch < :cut"), {"cut": cutoff})
            return cur.rowcount or 0

    @_retry_sql()
    def vacuum_analyze(self) -> None:
        with self._begin() as c:
            c.exec_driver_sql("VACUUM;")
            c.exec_driver_sql("ANALYZE;")

    def db_info(self) -> Dict[str, Any]:
        try:
            with self._begin() as c:
                page_size = c.execute(text("PRAGMA page_size")).fetchone()[0]
                page_count = c.execute(text("PRAGMA page_count")).fetchone()[0]
                freelist = c.execute(text("PRAGMA freelist_count")).fetchone()[0]
            return {
                "page_size": page_size,
                "page_count": page_count,
                "freelist": freelist,
                "approx_mb": round(page_size * page_count / (1024 * 1024), 2),
            }
        except Exception:
            return {}

    def ping(self) -> bool:
        try:
            with self._begin() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # ------------------------ Counters ------------------------

    @_retry_sql()
    def incr(self, key: str, delta: int = 1, ns: str = "default") -> int:
        now = _epoch_now()
        with self._begin() as c:
            c.execute(text("""
                INSERT INTO counters(ns,key,val,ttl_epoch,updated_epoch)
                VALUES(:ns,:k,:v,NULL,:now)
                ON CONFLICT(ns,key) DO UPDATE SET val = val + :v, updated_epoch=:now
            """), {"ns": ns, "k": key, "v": delta, "now": now})
            cur = c.execute(text("SELECT val FROM counters WHERE ns=:ns AND key=:k"),
                            {"ns": ns, "k": key}).fetchone()
            return int(cur[0]) if cur else delta

    @_retry_sql()
    def set_counter(self, key: str, val: int, ns: str = "default", ttl_seconds: Optional[int] = None) -> None:
        now = _epoch_now()
        ttl = (now + int(ttl_seconds)) if ttl_seconds else None
        with self._begin() as c:
            c.execute(text("""
                INSERT INTO counters(ns,key,val,ttl_epoch,updated_epoch)
                VALUES(:ns,:k,:v,:ttl,:now)
                ON CONFLICT(ns,key) DO UPDATE SET val=:v, ttl_epoch=:ttl, updated_epoch=:now
            """), {"ns": ns, "k": key, "v": val, "ttl": ttl, "now": now})

    @_retry_sql()
    def incr_with_expiry(self, key: str, delta: int, ttl_seconds: int, ns: str = "default") -> int:
        now = _epoch_now()
        ttl = now + int(ttl_seconds)
        with self._begin() as c:
            c.execute(text("""
                INSERT INTO counters(ns,key,val,ttl_epoch,updated_epoch)
                VALUES(:ns,:k,:v,:ttl,:now)
                ON CONFLICT(ns,key) DO UPDATE SET
                    val = CASE
                          WHEN counters.ttl_epoch IS NOT NULL AND counters.ttl_epoch < :now
                          THEN :v
                          ELSE counters.val + :v
                          END,
                    ttl_epoch=:ttl,
                    updated_epoch=:now
            """), {"ns": ns, "k": key, "v": delta, "ttl": ttl, "now": now})
            cur = c.execute(text("SELECT val FROM counters WHERE ns=:ns AND key=:k"),
                            {"ns": ns, "k": key}).fetchone()
            return int(cur[0]) if cur else delta

    @_retry_sql()
    def get_counter(self, key: str, ns: str = "default") -> int:
        now = _epoch_now()
        with self._begin() as c:
            cur = c.execute(text("SELECT val, ttl_epoch FROM counters WHERE ns=:ns AND key=:k"),
                            {"ns": ns, "k": key}).fetchone()
        if not cur:
            return 0
        val, ttl = int(cur[0]), cur[1]
        if ttl is not None and ttl < now:
            return 0
        return val

    @_retry_sql()
    def get_counter_with_ttl(self, key: str, ns: str = "default") -> Tuple[int, Optional[int]]:
        now = _epoch_now()
        with self._begin() as c:
            cur = c.execute(text("SELECT val, ttl_epoch FROM counters WHERE ns=:ns AND key=:k"),
                            {"ns": ns, "k": key}).fetchone()
        if not cur:
            return (0, None)
        val, ttl = int(cur[0]), cur[1]
        if ttl is None:
            return (val, None)
        remain = max(0, ttl - now)
        return (0 if remain == 0 else val, remain)

    @_retry_sql()
    def reset_counter(self, key: str, ns: str = "default") -> None:
        with self._begin() as c:
            c.execute(text("DELETE FROM counters WHERE ns=:ns AND key=:k"), {"ns": ns, "k": key})

    # -------- NEW: account-local (Europe/Amsterdam) daily counters --------

    def set_counter_daily(self, key: str, val: int, ns: str = "default", tz_name: Optional[str] = None) -> None:
        """
        Set a counter and have it expire at the next local midnight (account tz by default).
        """
        ttl_seconds = _seconds_until_local_midnight(tz_name)
        self.set_counter(key, val, ns=ns, ttl_seconds=ttl_seconds)

    def incr_with_daily_cap(self, key: str, delta: int, cap: int, ns: str = "default", tz_name: Optional[str] = None) -> Tuple[int, int]:
        """
        Increment a counter that resets at *local midnight*, enforcing a per-day cap.
        Returns (new_value, applied_delta). If capped, applied_delta may be lower than delta.
        """
        # Ensure TTL is aligned to local midnight
        _, remain = self.get_counter_with_ttl(key, ns=ns)
        if remain is None or remain <= 0:
            self.set_counter_daily(key, 0, ns=ns, tz_name=tz_name)
        current = self.get_counter(key, ns=ns)
        allowed = max(0, min(delta, cap - current))
        if allowed == 0:
            return current, 0
        new_val = self.incr(key, allowed, ns=ns)
        return new_val, allowed

    def get_counter_daily(self, key: str, ns: str = "default", tz_name: Optional[str] = None) -> Tuple[int, int]:
        """
        Get (value, seconds_until_reset) for a daily counter keyed to local midnight.
        If no TTL exists yet, attaches one to midnight and returns (current, ttl).
        """
        val, remain = self.get_counter_with_ttl(key, ns=ns)
        if remain is None or remain <= 0:
            ttl_seconds = _seconds_until_local_midnight(tz_name)
            # Reapply TTL without changing value
            self.set_counter(key, val, ns=ns, ttl_seconds=ttl_seconds)
            remain = ttl_seconds
        return val, int(remain)

    # ------------------------ Flags ------------------------

    @_retry_sql()
    def set_flag(self, entity_type: str, entity_id: str, k: str, v: str, ttl_seconds: Optional[int] = None) -> None:
        now = _epoch_now()
        ttl = (now + int(ttl_seconds)) if ttl_seconds else None
        with self._begin() as c:
            c.execute(text("""
                INSERT INTO flags(entity_type,entity_id,k,v,ttl_epoch,updated_epoch)
                VALUES(:et,:eid,:k,:v,:ttl,:now)
                ON CONFLICT(entity_type,entity_id,k) DO UPDATE SET
                    v=:v, ttl_epoch=:ttl, updated_epoch=:now
            """), {"et": entity_type, "eid": entity_id, "k": k, "v": v, "ttl": ttl, "now": now})

    @_retry_sql()
    def get_flag(self, entity_type: str, entity_id: str, k: str) -> Optional[Dict[str, Any]]:
        now = _epoch_now()
        with self._begin() as c:
            cur = c.execute(text("""
                SELECT v, ttl_epoch, updated_epoch
                FROM flags WHERE entity_type=:et AND entity_id=:eid AND k=:k
            """), {"et": entity_type, "eid": entity_id, "k": k}).fetchone()
        if not cur:
            return None
        v, ttl, upd = cur[0], cur[1], cur[2]
        if ttl is not None and ttl < now:
            return None
        return {"v": v, "ttl_epoch": ttl, "updated_epoch": upd}

    @_retry_sql()
    def clear_flag(self, entity_type: str, entity_id: str, k: str) -> None:
        with self._begin() as c:
            c.execute(text("DELETE FROM flags WHERE entity_type=:et AND entity_id=:eid AND k=:k"),
                      {"et": entity_type, "eid": entity_id, "k": k})

    # ------------------------ Convenience logs ------------------------

    def log_kill(
        self, *, stage: str, entity_id: str, rule_type: str, reason: str,
        observed: Dict[str, Any], thresholds: Dict[str, Any], window: Optional[str] = None,
        attribution: Optional[str] = None, actor: Optional[str] = None, dedup_key: Optional[str] = None
    ) -> str:
        return self.log(
            entity_type="ad", entity_id=entity_id, action="PAUSE", level="warn", stage=stage,
            rule_type=rule_type, reason=reason, thresholds=thresholds, observed=observed,
            window=window, attribution=attribution, actor=actor, dedup_key=dedup_key
        )

    def log_promote(
        self, *, from_stage: str, to_stage: str, entity_id: str, reason: str,
        meta: Optional[Dict[str, Any]] = None, actor: Optional[str] = None, dedup_key: Optional[str] = None
    ) -> str:
        return self.log(
            entity_type="ad", entity_id=entity_id, action=f"{from_stage}_TO_{to_stage}",
            level="info", stage=to_stage, reason=reason, meta=meta, actor=actor, dedup_key=dedup_key
        )

    def log_scale(
        self, *, entity_id: str, pct: int, reason: str,
        meta: Optional[Dict[str, Any]] = None, actor: Optional[str] = None, dedup_key: Optional[str] = None
    ) -> str:
        m = {"increase_pct": int(pct)}
        if meta:
            m.update(meta)
        return self.log(
            entity_type="ad", entity_id=entity_id, action="SCALE_UP",
            level="info", stage="SCALE", reason=reason, meta=m, actor=actor, dedup_key=dedup_key
        )


__all__ = [
    "Store",
    "ActionRecord",
]
