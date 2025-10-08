from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests

# ---- Local time/currency config (AMS + EUR) ----
try:
    from zoneinfo import ZoneInfo  # py>=3.9
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

UTC = timezone.utc
ACCOUNT_TZ_NAME = os.getenv("ACCOUNT_TZ") or os.getenv("ACCOUNT_TIMEZONE") or "Europe/Amsterdam"
ACCOUNT_CURRENCY = os.getenv("ACCOUNT_CURRENCY", "EUR")
ACCOUNT_CURRENCY_SYMBOL = os.getenv("ACCOUNT_CURRENCY_SYMBOL", "€")

def _tz():
    try:
        return ZoneInfo(ACCOUNT_TZ_NAME) if ZoneInfo else None
    except Exception:
        return None

def _now_local() -> datetime:
    tz = _tz()
    return datetime.now(tz) if tz else datetime.now(UTC)

def _fmt_currency(amount: Optional[float]) -> str:
    if amount is None:
        return ""
    # Keep simple and deterministic formatting without locale
    return f"{ACCOUNT_CURRENCY_SYMBOL}{amount:,.0f}"

# ---- Env helpers ----
ENVBOOL = lambda v, d=False: (os.getenv(v, str(int(d))) or "").lower() in ("1", "true", "yes", "y")
ENVINT = lambda v, d: int(os.getenv(v, str(d)) or d)
ENVF = lambda v, d: float(os.getenv(v, str(d)) or d)

SLACK_TIMEOUT = ENVF("SLACK_TIMEOUT", 10.0)
SLACK_RETRY_MAX = ENVINT("SLACK_RETRY_MAX", 3)
SLACK_BACKOFF_BASE = ENVF("SLACK_BACKOFF_BASE", 0.4)
SLACK_BACKOFF_CAP = ENVF("SLACK_BACKOFF_CAP", 8.0)
SLACK_CB_FAILS = ENVINT("SLACK_CIRCUIT_THRESHOLD", 5)
SLACK_CB_RESET_SEC = ENVINT("SLACK_CIRCUIT_RESET_SEC", 120)
SLACK_BATCH_WINDOW = ENVINT("SLACK_BATCH_WINDOW_SEC", 15)
SLACK_DEDUP_WINDOW_S = ENVINT("SLACK_DEDUP_WINDOW_SEC", 900)
SLACK_TTL_DEFAULT_S = ENVINT("SLACK_TTL_SEC", 7200)
SLACK_SUPPRESS_MAX_H = ENVINT("SLACK_SUPPRESS_MAX_PER_H", 5)
TOKEN_BURST = ENVINT("SLACK_TB_BURST", 5)
TOKEN_RATE_QPS = ENVF("SLACK_TB_QPS", 1.0)
OUTBOX_PATH = os.getenv("SLACK_OUTBOX_DB", os.path.join(os.getcwd(), "data", "slack_outbox.sqlite"))
MAX_TEXT_LEN = max(1024, ENVINT("SLACK_MAX_TEXT_LEN", 38000))
MAX_BLOCKS = max(1, ENVINT("SLACK_MAX_BLOCKS", 45))
MAX_BLOCK_TEXT = max(256, ENVINT("SLACK_MAX_BLOCK_TEXT", 2900))
LOG_STDOUT = ENVBOOL("SLACK_LOG_STDOUT", True)

try:
    from prometheus_client import Counter, Histogram  # type: ignore

    METRICS = True
    M_SENT = Counter("slack_sent_total", "Messages sent", ["topic"])
    M_FAIL = Counter("slack_fail_total", "Messages failed", ["topic"])
    M_RETRY = Counter("slack_retry_total", "Retries", ["topic"])
    M_QUEUE = Counter("slack_enqueued_total", "Enqueued", ["topic"])
    H_LAT = Histogram("slack_send_latency_seconds", "Send latency", ["topic"])
except Exception:  # pragma: no cover
    METRICS = False

    class _N:
        def labels(self, *_, **__): return self
        def inc(self, *_): pass
        def observe(self, *_): pass
    M_SENT = M_FAIL = M_RETRY = M_QUEUE = H_LAT = _N()  # type: ignore


def _log(msg: str) -> None:
    if LOG_STDOUT:
        print(msg, flush=True)


def _now() -> datetime:
    return datetime.now(UTC)


def _truncate(s: str, limit: int) -> str:
    return s if len(s) <= limit else (s[: max(0, limit - 1)] + "…")


def _mk_section(text: str) -> Dict[str, Any]:
    return {"type": "section", "text": {"type": "mrkdwn", "text": _truncate(text, MAX_BLOCK_TEXT)}}


def _mk_context(text: str) -> Dict[str, Any]:
    return {"type": "context", "elements": [{"type": "mrkdwn", "text": _truncate(text, MAX_BLOCK_TEXT)}]}


def _mk_divider() -> Dict[str, Any]:
    return {"type": "divider"}


def _ts_footer() -> str:
    # Show local AMS time (falls back to UTC if zoneinfo missing)
    now_local = _now_local()
    tz_name = ACCOUNT_TZ_NAME if _tz() else "UTC"
    return f"Sent {now_local.strftime('%Y-%m-%d %H:%M:%S')} {tz_name}"


def _sanitize_for_dedup(s: str) -> str:
    s = re.sub(r"\s+", " ", s or "")
    s = re.sub(r"https?://\S+", "<url>", s)
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+", "<email>", s)
    return s.strip().lower()


def _hash_dedup(
    text: str,
    blocks: Optional[List[Dict[str, Any]]],
    attachments: Optional[List[Dict[str, Any]]],
    topic: str,
    severity: str,
) -> str:
    base = [
        topic,
        severity,
        _sanitize_for_dedup(text or ""),
        json.dumps(blocks or [], separators=(",", ":"), ensure_ascii=False),
        json.dumps(attachments or [], separators=(",", ":"), ensure_ascii=False),
    ]
    return hashlib.sha256(("|".join(base)).encode("utf-8")).hexdigest()


def _env_webhooks() -> Dict[str, str]:
    return {
        "default": os.getenv("SLACK_WEBHOOK_URL", "") or "",
        "alerts": os.getenv("SLACK_WEBHOOK_ALERTS", "") or "",
        "digest": os.getenv("SLACK_WEBHOOK_DIGEST", "") or "",
        "scale": os.getenv("SLACK_WEBHOOK_SCALE", "") or "",
    }


def _slack_enabled_now() -> bool:
    w = _env_webhooks()
    return any(w.values()) and ENVBOOL("SLACK_ENABLED", True)


@dataclass
class SlackMessage:
    text: str = ""
    blocks: Optional[List[Dict[str, Any]]] = None
    attachments: Optional[List[Dict[str, Any]]] = None
    topic: Union[Literal["default", "alerts", "digest", "scale"], str] = "default"
    severity: Literal["info", "warn", "error"] = "info"
    meta: Dict[str, Any] = field(default_factory=dict)
    thread_ts: Optional[str] = None
    ttl_seconds: Optional[int] = None
    dedup_key: Optional[str] = None

    def route_webhook(self) -> str:
        if isinstance(self.meta.get("webhook"), str) and self.meta["webhook"]:
            return str(self.meta["webhook"])
        w = _env_webhooks()
        if self.topic == "alerts" and w["alerts"]:
            return w["alerts"]
        if self.topic == "digest" and w["digest"]:
            return w["digest"]
        if self.topic == "scale" and w["scale"]:
            return w["scale"]
        return w["default"]

    def is_routable(self) -> bool:
        return bool(self.route_webhook())


EMOJI = {"info": "🟢", "warn": "🟠", "error": "🔴"}


def build_basic_blocks(title: str, lines: List[str], severity: str = "info", footer: Optional[str] = None) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = [_mk_section(f"*{EMOJI.get(severity,'ℹ️')} {title}*")]
    if lines:
        chunk, acc = [], ""
        for ln in lines:
            if len(acc) + len(ln) + 1 > MAX_BLOCK_TEXT:
                if acc:
                    chunk.append(acc)
                acc = ln
            else:
                acc += ("" if not acc else "\n") + ln
        if acc:
            chunk.append(acc)
        for ch in chunk[: (MAX_BLOCKS - 3)]:
            blocks.append(_mk_section(ch))
    if footer:
        blocks.append(_mk_divider())
        blocks.append(_mk_context(footer))
    return blocks[:MAX_BLOCKS]


def template_kill(stage: str, entity_name: str, reason: str, metrics: Dict[str, Any], link: Optional[str] = None) -> SlackMessage:
    title = f"{stage} • Ad Paused"
    body = [f"*Ad:* `{entity_name}`", f"*Reason:* {reason}", "*Metrics:* " + ", ".join(f"{k}={v}" for k, v in metrics.items())]
    if link:
        body.append(f"<{link}|Open in Ads Manager>")
    return SlackMessage(
        text=f"[{stage}] PAUSE {entity_name}: {reason}",
        blocks=build_basic_blocks(title, body, severity="warn", footer=_ts_footer()),
        severity="warn",
        topic="alerts",
    )


def template_promote(src: str, dst: str, entity_name: str, budget: Optional[float] = None, link: Optional[str] = None) -> SlackMessage:
    body = [f"*Creative:* `{entity_name}`"]
    if budget is not None:
        body.append(f"*Budget:* {_fmt_currency(budget)}/day")
    if link:
        body.append(f"<{link}|Open in Ads Manager>")
    return SlackMessage(
        text=f"[{src}->{dst}] {entity_name} promoted.",
        blocks=build_basic_blocks(f"{src} → {dst} • Promotion", body, footer=_ts_footer()),
        topic="alerts",
    )


def template_scale(entity_name: str, pct: int, new_budget: Optional[float] = None, link: Optional[str] = None) -> SlackMessage:
    body = [f"*Ad Set:* `{entity_name}`", f"*Increase:* +{pct}%"]
    if new_budget is not None:
        body.append(f"*New Budget:* {_fmt_currency(new_budget)}/day")
    if link:
        body.append(f"<{link}|Open in Ads Manager>")
    return SlackMessage(
        text=f"[SCALE] {entity_name} +{pct}%",
        blocks=build_basic_blocks("Scaling • Budget Increase", body, footer=_ts_footer()),
        topic="scale",
    )


def template_digest(date_label: str, stage_stats: Dict[str, Dict[str, Any]]) -> SlackMessage:
    lines = [f"*{stage}:* " + " | ".join(f"{k}={v}" for k, v in stats.items()) for stage, stats in stage_stats.items()]
    return SlackMessage(
        text=f"[DIGEST] {date_label}",
        blocks=build_basic_blocks(f"Daily Digest • {date_label}", lines, footer=_ts_footer()),
        topic="digest",
    )


class Outbox:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL;")
        self.db.execute("PRAGMA synchronous=NORMAL;")
        self._init()

    def _init(self) -> None:
        self.db.execute(
            """
          CREATE TABLE IF NOT EXISTS outbox(
            id TEXT PRIMARY KEY,
            ts_epoch INTEGER NOT NULL,
            webhook TEXT NOT NULL,
            payload TEXT NOT NULL,
            topic TEXT NOT NULL,
            severity TEXT NOT NULL,
            tries INTEGER NOT NULL DEFAULT 0,
            next_epoch INTEGER NOT NULL,
            dedup_key TEXT,
            dedup_until INTEGER,
            expires_epoch INTEGER,
            status TEXT NOT NULL DEFAULT 'pending'
          );
        """
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_outbox_next ON outbox(next_epoch, status);")
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_outbox_dedup ON outbox(dedup_key);")
        self.db.execute(
            """
          CREATE TABLE IF NOT EXISTS suppress(
            dedup_key TEXT PRIMARY KEY,
            window_start INTEGER NOT NULL,
            count INTEGER NOT NULL
          );
        """
        )

    def close(self) -> None:
        try:
            self.db.close()
        except Exception:
            pass

    def enqueue(self, webhook: str, payload: Dict[str, Any], topic: str, severity: str, dedup_key: str, ttl_s: Optional[int]) -> str:
        now = int(time.time())
        expires = now + (ttl_s or SLACK_TTL_DEFAULT_S)
        dedup_until = now + SLACK_DEDUP_WINDOW_S
        msg_id = str(uuid.uuid4())
        self.db.execute(
            """
            INSERT INTO outbox(id, ts_epoch, webhook, payload, topic, severity, tries, next_epoch,
                               dedup_key, dedup_until, expires_epoch, status)
            VALUES(?,?,?,?,?,?,0,?,?,?,?, 'pending')
        """,
            (msg_id, now, webhook, json.dumps(payload, separators=(",", ":"), ensure_ascii=False), topic, severity, now, dedup_key, dedup_until, expires),
        )
        return msg_id

    def exists_recent(self, dedup_key: str) -> bool:
        now = int(time.time())
        cur = self.db.execute(
            "SELECT 1 FROM outbox WHERE dedup_key=? AND dedup_until>=? AND status IN ('pending','sent') LIMIT 1", (dedup_key, now)
        ).fetchone()
        return bool(cur)

    def suppress_or_increment(self, dedup_key: str) -> bool:
        now = int(time.time())
        row = self.db.execute("SELECT window_start, count FROM suppress WHERE dedup_key=?", (dedup_key,)).fetchone()
        hour = 3600
        if not row:
            self.db.execute("INSERT INTO suppress(dedup_key, window_start, count) VALUES(?,?,?)", (dedup_key, now, 1))
            return False
        wstart, cnt = int(row[0]), int(row[1])
        if now - wstart > hour:
            self.db.execute("UPDATE suppress SET window_start=?, count=? WHERE dedup_key=?", (now, 1, dedup_key))
            return False
        if cnt + 1 >= SLACK_SUPPRESS_MAX_H:
            self.db.execute("UPDATE suppress SET count=count+1 WHERE dedup_key=?", (dedup_key,))
            return True
        self.db.execute("UPDATE suppress SET count=count+1 WHERE dedup_key=?", (dedup_key,))
        return False

    def dequeue_ready(self, limit: int = 20) -> List[Dict[str, Any]]:
        now = int(time.time())
        cur = self.db.execute(
            """
            SELECT id, webhook, payload, topic, severity, tries
            FROM outbox
            WHERE status='pending' AND next_epoch<=? AND expires_epoch>=?
            ORDER BY next_epoch ASC
            LIMIT ?
        """,
            (now, now, limit),
        )
        return [
            {"id": r[0], "webhook": r[1], "payload": json.loads(r[2]), "topic": r[3], "severity": r[4], "tries": int(r[5])}
            for r in cur.fetchall()
        ]

    def mark_sent(self, msg_id: str) -> None:
        self.db.execute("UPDATE outbox SET status='sent' WHERE id=?", (msg_id,))

    def mark_failed_retry(self, msg_id: str, attempt: int, backoff_sec: float) -> None:
        next_at = int(time.time() + backoff_sec)
        self.db.execute("UPDATE outbox SET tries=?, next_epoch=? WHERE id=?", (attempt, next_at, msg_id))

    def mark_dead(self, msg_id: str) -> None:
        self.db.execute("UPDATE outbox SET status='dead' WHERE id=?", (msg_id,))

    def queue_depth(self) -> int:
        cur = self.db.execute("SELECT COUNT(*) FROM outbox WHERE status='pending'")
        return int(cur.fetchone()[0])


class TokenBucket:
    def __init__(self, rate_qps: float, burst: int):
        self.rate = max(0.01, rate_qps)
        self.burst = max(1, burst)
        self.tokens = float(self.burst)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def allow(self) -> bool:
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.burst, self.tokens + (now - self.updated) * self.rate)
            self.updated = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

    def wait(self) -> None:
        while not self.allow():
            time.sleep(0.05)


class SlackClient:
    def __init__(self):
        self.enabled = _slack_enabled_now()
        self.dry_run = ENVBOOL("SLACK_DRY_RUN", False) or not self.enabled
        self.timeout = SLACK_TIMEOUT
        self.retry_max = max(0, SLACK_RETRY_MAX)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.cb_open_until: Optional[float] = None
        self.fail_count = 0
        self.outbox = Outbox(OUTBOX_PATH)
        self.tb_by_webhook: Dict[str, TokenBucket] = {}
        self.batch_buf: List[SlackMessage] = []
        self.batch_started: Optional[float] = None
        self._sender_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        if self.enabled and not self.dry_run:
            self._start_sender()

    def close(self) -> None:
        try:
            self._stop.set()
            if self._sender_thread and self._sender_thread.is_alive():
                self._sender_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            self.session.close()
        except Exception:
            pass
        try:
            self.outbox.close()
        except Exception:
            pass

    def __enter__(self) -> "SlackClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def notify_text(self, text: str, severity: Literal["info", "warn", "error"] = "info", topic: Union[str, Literal["default", "alerts", "digest", "scale"]] = "default", ttl_seconds: Optional[int] = None) -> None:
        self.notify(SlackMessage(text=text, severity=severity, topic=topic, ttl_seconds=ttl_seconds))

    def notify(self, msg: SlackMessage) -> None:
        webhook = msg.route_webhook()
        if not webhook:
            _log(f"[SLACK MOCK {msg.severity}/{msg.topic}] {msg.text} (no webhook)")
            return
        dedup_key = msg.dedup_key or _hash_dedup(msg.text, msg.blocks, msg.attachments, str(msg.topic), msg.severity)
        if self.outbox.exists_recent(dedup_key):
            if self.outbox.suppress_or_increment(dedup_key):
                return
        payloads = self._build_payloads(msg)
        sent_any = False
        for payload in payloads:
            if self._circuit_open():
                self._enqueue_payload(webhook, payload, str(msg.topic), msg.severity, dedup_key, msg.ttl_seconds)
                continue
            tb = self.tb_by_webhook.setdefault(webhook, TokenBucket(TOKEN_RATE_QPS, TOKEN_BURST))
            tb.wait()
            ok = self._post_with_retries(webhook, payload, topic=str(msg.topic))
            if ok:
                sent_any = True
                try:
                    M_SENT.labels(str(msg.topic)).inc()
                except Exception:
                    pass
                self._on_success()
            else:
                self._enqueue_payload(webhook, payload, str(msg.topic), msg.severity, dedup_key, msg.ttl_seconds)
                self._on_failure()
        if self.enabled and not self.dry_run:
            self._start_sender()
        _log(f"[SLACK {'SENT' if sent_any else 'QUEUED'} {msg.severity}/{msg.topic}] {msg.text}")

    def batch(self, msg: SlackMessage) -> None:
        now = time.time()
        if self.batch_started is None:
            self.batch_started = now
        self.batch_buf.append(msg)
        if (now - self.batch_started) >= SLACK_BATCH_WINDOW:
            title = "Aggregated Alerts"
            lines = [f"• {m.text}" for m in self.batch_buf]
            agg = SlackMessage(
                text=f"[BATCH] {len(self.batch_buf)} alerts",
                blocks=build_basic_blocks(title, lines, severity="warn", footer=_ts_footer()),
                topic="alerts",
                severity="warn",
            )
            self.notify(agg)
            self.batch_buf.clear()
            self.batch_started = None

    def flush(self, timeout: float = 10.0) -> None:
        deadline = time.time() + max(0.0, timeout)
        while self.outbox.queue_depth() > 0 and time.time() < deadline:
            time.sleep(0.1)

    def _enqueue_payload(self, webhook: str, payload: Dict[str, Any], topic: str, severity: str, dedup_key: str, ttl_seconds: Optional[int]) -> None:
        ttl = ttl_seconds or SLACK_TTL_DEFAULT_S
        self.outbox.enqueue(webhook, payload, topic, severity, dedup_key, ttl)
        try:
            M_QUEUE.labels(topic).inc()
        except Exception:
            pass

    def _build_payloads(self, msg: SlackMessage) -> List[Dict[str, Any]]:
        if msg.blocks:
            blocks = msg.blocks[:MAX_BLOCKS]
            payloads: List[Dict[str, Any]] = []
            chunk: List[Dict[str, Any]] = []
            size = 0
            for b in blocks:
                bs = len(json.dumps(b, separators=(",", ":"), ensure_ascii=False))
                if len(chunk) >= MAX_BLOCKS or size + bs > MAX_TEXT_LEN:
                    payloads.append({"blocks": chunk, "attachments": msg.attachments or []})
                    chunk, size = [], 0
                chunk.append(b)
                size += bs
            if chunk:
                payloads.append({"blocks": chunk, "attachments": msg.attachments or []})
            return payloads
        return [{"text": _truncate(msg.text or "", MAX_TEXT_LEN)}]

    def _start_sender(self) -> None:
        if self._sender_thread and self._sender_thread.is_alive():
            return
        self._stop.clear()
        t = threading.Thread(target=self._sender_loop, name="SlackSender", daemon=True)
        self._sender_thread = t
        t.start()

    def _sender_loop(self) -> None:
        while not self._stop.is_set():
            if self._circuit_open():
                time.sleep(0.5)
                continue
            batch = self.outbox.dequeue_ready(limit=20)
            if not batch:
                time.sleep(0.25)
                continue
            for item in batch:
                if self._stop.is_set():
                    break
                webhook = item["webhook"]
                tb = self.tb_by_webhook.setdefault(webhook, TokenBucket(TOKEN_RATE_QPS, TOKEN_BURST))
                tb.wait()
                ok = self._post_with_retries(webhook, item["payload"], topic=item["topic"])
                if ok:
                    self.outbox.mark_sent(item["id"])
                    try:
                        M_SENT.labels(item["topic"]).inc()
                    except Exception:
                        pass
                    self._on_success()
                else:
                    attempt = item["tries"] + 1
                    backoff = min(SLACK_BACKOFF_CAP, SLACK_BACKOFF_BASE * (2 ** (attempt - 1)))
                    self.outbox.mark_failed_retry(item["id"], attempt, backoff)
                    try:
                        M_RETRY.labels(item["topic"]).inc()
                    except Exception:
                        pass
                    self._on_failure()
                    if attempt > SLACK_RETRY_MAX + SLACK_CB_FAILS:
                        self.outbox.mark_dead(item["id"])
                        try:
                            M_FAIL.labels(item["topic"]).inc()
                        except Exception:
                            pass

    def _post_with_retries(self, webhook: str, payload: Dict[str, Any], topic: str) -> bool:
        attempts = self.retry_max + 1
        for n in range(attempts):
            t0 = time.perf_counter()
            try:
                resp = self.session.post(webhook, json=payload, timeout=self.timeout)
            except requests.RequestException:
                time.sleep(min(SLACK_BACKOFF_CAP, SLACK_BACKOFF_BASE * (2 ** n)))
                continue
            try:
                H_LAT.labels(topic).observe(time.perf_counter() - t0)
            except Exception:
                pass
            sc = resp.status_code
            if 200 <= sc < 300:
                return True
            if sc == 429:
                ra = 1.0
                try:
                    ra = float(resp.headers.get("Retry-After", "1"))
                except Exception:
                    pass
                time.sleep(max(0.5, ra))
                continue
            if 500 <= sc < 600:
                time.sleep(min(SLACK_BACKOFF_CAP, SLACK_BACKOFF_BASE * (2 ** n)))
                continue
            _log(f"[SLACK ERROR] HTTP {sc}: {resp.text[:200]}")
            return False
        return False

    def _circuit_open(self) -> bool:
        return self.cb_open_until is not None and time.time() < self.cb_open_until

    def _on_failure(self) -> None:
        self.fail_count += 1
        if self.fail_count >= SLACK_CB_FAILS:
            self.cb_open_until = time.time() + SLACK_CB_RESET_SEC

    def _on_success(self) -> None:
        self.fail_count = 0
        self.cb_open_until = None


_client: Optional[SlackClient] = None


def client() -> SlackClient:
    global _client
    if _client is None:
        _client = SlackClient()
    return _client


def notify(text: str, severity: Literal["info", "warn", "error"] = "info", topic: Union[str, Literal["default", "alerts", "digest", "scale"]] = "default", ttl_seconds: Optional[int] = None) -> None:
    client().notify_text(text, severity=severity, topic=topic, ttl_seconds=ttl_seconds)


def alert_kill(stage: str, entity_name: str, reason: str, metrics: Dict[str, Any], link: Optional[str] = None) -> None:
    client().notify(template_kill(stage, entity_name, reason, metrics, link))


def alert_promote(from_stage: str, to_stage: str, entity_name: str, budget: Optional[float] = None, link: Optional[str] = None) -> None:
    client().notify(template_promote(from_stage, to_stage, entity_name, budget, link))


def alert_scale(entity_name: str, pct: int, new_budget: Optional[float] = None, link: Optional[str] = None) -> None:
    client().notify(template_scale(entity_name, pct, new_budget, link))


def post_digest(date_label: str, stage_stats: Dict[str, Dict[str, Any]]) -> None:
    client().notify(template_digest(date_label, stage_stats))


__all__ = [
    "SlackClient",
    "SlackMessage",
    "Outbox",
    "TokenBucket",
    "notify",
    "alert_kill",
    "alert_promote",
    "alert_scale",
    "post_digest",
    "client",
]
