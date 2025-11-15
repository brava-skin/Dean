from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Union

import requests

try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

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

def _fmt_currency(amount: Optional[float | str]) -> str:
    if amount is None or amount == "":
        return "—"
    try:
        val = float(str(amount).replace(",", ""))
    except Exception:
        return str(amount)
    return f"{ACCOUNT_CURRENCY_SYMBOL}{val:,.2f}"


def _fmt_pct(v: Optional[float | str]) -> str:
    if v is None or v == "":
        return "—"
    s = str(v).strip()
    if s.endswith("%"):
        try:
            s = s.rstrip("%")
            val = float(s)
            return f"{val:.1f}%"
        except Exception:
            return s
    try:
        f = float(s)
    except Exception:
        return s
    if 0 <= f <= 1:
        f *= 100.0
    return f"{f:.1f}%"

def _fmt_int(v: Optional[int | float | str]) -> str:
    if v is None or v == "":
        return "0"
    try:
        return f"{int(float(str(v).replace(',', '')))}"
    except Exception:
        return str(v)

ENVBOOL = lambda v, d=False: (os.getenv(v, str(int(d))) or "").lower() in ("1", "true", "yes", "y")
ENVINT = lambda v, d: int(os.getenv(v, str(d)) or d)
ENVF = lambda v, d: float(os.getenv(v, str(d)) or d)

if not os.getenv("SLACK_ENABLED") and not os.getenv("CI"):
    os.environ["SLACK_ENABLED"] = "0"
    os.environ.setdefault("SLACK_DRY_RUN", "1")

SLACK_TEMP_DISABLED = ENVBOOL("SLACK_TEMP_DISABLE", True)


def _resolve_outbox_path() -> str:
    override = os.getenv("SLACK_OUTBOX_DB")
    if override:
        return override

    repo_candidate = os.path.join(os.getcwd(), "data", "slack_outbox.sqlite")
    try:
        os.makedirs(os.path.dirname(repo_candidate), exist_ok=True)
        test_path = os.path.join(os.path.dirname(repo_candidate), ".slack_outbox.writecheck")
        with open(test_path, "wb") as fh:
            fh.write(b"")
        os.remove(test_path)
        return repo_candidate
    except OSError:
        pass

    tmp_candidate = os.path.join(tempfile.gettempdir(), "dean_slack_outbox.sqlite")
    os.makedirs(os.path.dirname(tmp_candidate), exist_ok=True)
    return tmp_candidate

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
TOKEN_BURST = max(1, ENVINT("SLACK_TB_BURST", 1))
TOKEN_RATE_QPS = max(0.05, ENVF("SLACK_TB_QPS", 0.5))
OUTBOX_PATH = _resolve_outbox_path()
MAX_TEXT_LEN = max(1024, ENVINT("SLACK_MAX_TEXT_LEN", 38000))
MAX_BLOCKS = max(1, ENVINT("SLACK_MAX_BLOCKS", 45))
MAX_BLOCK_TEXT = max(256, ENVINT("SLACK_MAX_BLOCK_TEXT", 2900))
LOG_STDOUT = ENVBOOL("SLACK_LOG_STDOUT", True)
SLACK_INFO_QUEUE_DROP = ENVINT("SLACK_DROP_INFO_AT", 40)

try:
    from prometheus_client import Counter, Histogram

    METRICS = True
    M_SENT = Counter("slack_sent_total", "Messages sent", ["topic"])
    M_FAIL = Counter("slack_fail_total", "Messages failed", ["topic"])
    M_RETRY = Counter("slack_retry_total", "Retries", ["topic"])
    M_QUEUE = Counter("slack_enqueued_total", "Enqueued", ["topic"])
    H_LAT = Histogram("slack_send_latency_seconds", "Send latency", ["topic"])
except Exception:
    METRICS = False
    class _N:
        def labels(self, *_, **__): return self
        def inc(self, *_): pass
        def observe(self, *_): pass
    M_SENT = M_FAIL = M_RETRY = M_QUEUE = H_LAT = _N()


def _log(msg: str) -> None:
    if LOG_STDOUT:
        msg = msg.replace("🛑", "[STOP]").replace("⚠️", "[WARN]").replace("🚀", "[LAUNCH]").replace("⬆️", "[SCALE]").replace("🧬", "[DUP]")
        msg = msg.replace("🩺", "[HEALTH]").replace("⏳", "[WAIT]").replace("❌", "[ERROR]").replace("ℹ️", "[INFO]")
        msg = msg.replace("📥", "[LOAD]").replace("📦", "[SAVE]").replace("✅", "[OK]").replace("❗", "[ALERT]")
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode('ascii', 'replace').decode('ascii'), flush=True)


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
    now_local = _now_local()
    tz_name = ACCOUNT_TZ_NAME if _tz() else "UTC"
    return f"Sent {now_local.strftime('%Y-%m-%d %H:%M')} {tz_name}"

def _env_webhooks() -> Dict[str, str]:
    main_webhook = os.getenv("SLACK_WEBHOOK_URL", "") or ""
    
    return {
        "default": main_webhook,
        "alerts": os.getenv("SLACK_WEBHOOK_ALERTS", "") or main_webhook,
        "digest": os.getenv("SLACK_WEBHOOK_DIGEST", "") or main_webhook,
        "scale": os.getenv("SLACK_WEBHOOK_SCALE", "") or main_webhook,
    }

def _slack_enabled_now() -> bool:
    w = _env_webhooks()
    return any(w.values()) and ENVBOOL("SLACK_ENABLED", True)

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

    def sanitized_text(self) -> str:
        pattern = re.compile(r"[\x00-\x1f\x7f]")
        s = pattern.sub("", self.text or "")
        s = re.sub(r"[\[\](){}<>]", "", s)
        s = re.sub(r"\s+", " ", s)
        return s.strip()

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


EMOJI = {
    "info": "ℹ️",
    "warn": "⏸️",
    "error": "🛑",
    "ok": "🟢",
    "promote": "🚀",
    "scale": "📈",
    "digest": "🧾",
}

def _mk_title(sev: str, title: str) -> str:
    icon = EMOJI.get(sev, "ℹ️")
    return f"{icon} *{title}*"

def _sanitize_line(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"[\x00-\x1f\x7f]", " ", text)
    text = re.sub(r"[“”]", '"', text)
    text = re.sub(r"[‘’]", "'", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def build_basic_blocks(title: str, lines: List[str], severity: str = "info", footer: Optional[str] = None) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = [_mk_section(_mk_title(severity, _sanitize_line(title)))]
    if lines:
        chunk, acc = [], ""
        for ln in lines:
            ln = _sanitize_line(ln)
            if not ln:
                continue
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

def format_run_header(status: str, time_str: str, profile: str, spend: float, purch: int, cpa: Optional[float], be: Optional[float], impressions: int = 0, clicks: int = 0, ctr: Optional[float] = None, cpc: Optional[float] = None, cpm: Optional[float] = None, atc: int = 0, ic: int = 0, cost_per_atc: Optional[float] = None) -> str:
    spend_str = _fmt_currency(spend)
    cpa_str = fmt_eur(cpa)
    cost_per_atc_str = fmt_eur(cost_per_atc)

    def _fmt_int(value: int) -> str:
        return f"{value:,}"

    ctr_str = fmt_pct(ctr, 1)
    cpc_str = fmt_eur(cpc)
    cpm_str = fmt_eur(cpm)

    main_line = (
        f"{time_str} "
        f"Spend {spend_str} · ATC {atc} · IC {ic} · PUR {purch} | "
        f"CPA {cpa_str} · Cost/ATC {cost_per_atc_str} | "
        f"IMP {_fmt_int(impressions)} · Clicks {_fmt_int(clicks)} · "
        f"CTR {ctr_str} · CPC {cpc_str} · CPM {cpm_str}"
    )
    return main_line

def format_stage_line(stage: str, counts: Dict[str, any]) -> str:
    actions = []
    for key, value in counts.items():
        if key == "caps_enforced":
            if not bool(value):
                actions.append("Caps relaxed")
            continue

        if isinstance(value, dict):
            if 'count' in value:
                count_value = value['count']
            elif 'total' in value:
                count_value = value['total']
            elif len(value) > 0:
                count_value = list(value.values())[0]
            else:
                count_value = 0
        else:
            count_value = value
        
        try:
            count_value = float(count_value) if count_value is not None else 0
        except (ValueError, TypeError):
            count_value = 0
            
        if count_value > 0:
            if key == "kills":
                actions.append(f"Killed {int(count_value)}")
            elif key == "promotions":
                actions.append(f"Promoted {int(count_value)}")
            elif key == "launched" or key == "created":
                actions.append(f"Launched {int(count_value)}")
            elif key == "active":
                actions.append(f"Active {int(count_value)}")
            elif key == "hydrated":
                actions.append(f"Hydrated {int(count_value)}")
            elif key == "fatigue_flags":
                actions.append(f"Fatigue {count_value}")
            elif key == "data_quality_alerts":
                actions.append(f"Tracking issues {count_value}")
            elif key == "soft_passes":
                actions.append(f"Soft passed {count_value}")
            elif key == "scaled":
                actions.append(f"Scaled up {count_value}")
            elif key == "duped":
                actions.append(f"Duplicated {count_value}")
            elif key == "downscaled":
                actions.append(f"Scaled down {count_value}")
            elif key == "refreshed":
                actions.append(f"Refreshed {count_value}")
    
    if not actions:
        return f"{stage}: –"
    
    return f"{stage}: {', '.join(actions)}"

def prettify_ad_name(name: str) -> str:
    from infrastructure.utils import prettify_ad_name as utils_prettify_ad_name
    return utils_prettify_ad_name(name)

def fmt_eur(amount: Optional[float]) -> str:
    if amount is None:
        return "-"
    from infrastructure.utils import fmt_currency
    return fmt_currency(amount)

def fmt_pct(value: Optional[float], decimals: int = 1) -> str:
    if value is None:
        return "-"
    from infrastructure.utils import fmt_pct as utils_fmt_pct
    return utils_fmt_pct(value, decimals)

def fmt_roas(value: Optional[float]) -> str:
    if value is None:
        return "-"
    from infrastructure.utils import fmt_roas as utils_fmt_roas
    return utils_fmt_roas(value)

def fmt_int(value: Optional[int]) -> str:
    if value is None:
        return "-"
    from infrastructure.utils import fmt_int as utils_fmt_int
    return utils_fmt_int(value)

def post_run_header_and_get_thread_ts(
    status: str, 
    time_str: str, 
    profile: str, 
    spend: float, 
    purch: int, 
    cpa: Optional[float], 
    be: Optional[float],
    stage_summaries: List[Dict[str, Any]],
    impressions: int = 0,
    clicks: int = 0,
    ctr: Optional[float] = None,
    cpc: Optional[float] = None,
    cpm: Optional[float] = None,
    atc: int = 0,
    ic: int = 0,
    cost_per_atc: Optional[float] = None
) -> Optional[str]:
    header_text = format_run_header(status, time_str, profile, spend, purch, cpa, be, impressions, clicks, ctr, cpc, cpm, atc, ic, cost_per_atc)
    
    stage_lines = []
    for stage_data in stage_summaries:
        stage_name = stage_data.get("stage", "")
        counts = stage_data.get("counts", {})
        stage_line = format_stage_line(stage_name, counts)
        if stage_line and not stage_line.endswith(": –"):
            stage_lines.append(stage_line)
    
    if stage_lines:
        full_text = header_text + "\n" + "\n".join(stage_lines)
    else:
        full_text = header_text
    
    msg = SlackMessage(
        text=full_text,
        severity="info",
        topic="default"
    )
    
    client().notify(msg)
    return "mock_thread_ts"

def post_thread_ads_snapshot(thread_ts: str, ad_lines: List[str], alerts: List[str] = None) -> None:
    if not ad_lines and not alerts:
        return
    
    sections = []
    
    if ad_lines:
        sections.append("Active Ads\n" + "\n".join(ad_lines))
    
    if alerts:
        sections.append("🚨 Alerts\n" + "\n".join(alerts))
    
    full_text = "\n\n".join(sections)
    
    msg = SlackMessage(
        text=full_text,
        thread_ts=thread_ts,
        severity="info",
        topic="default"
    )
    
    client().notify(msg)

def template_kill(stage: str, entity_name: str, reason: str, metrics: Dict[str, Any], link: Optional[str] = None) -> SlackMessage:
    clean_name = prettify_ad_name(entity_name)
    
    ctr = metrics.get("CTR", metrics.get("ctr"))
    roas = metrics.get("ROAS", metrics.get("roas"))
    
    if isinstance(ctr, str):
        ctr_str = ctr
    else:
        ctr_str = fmt_pct(ctr) if ctr is not None else "–"
    
    if isinstance(roas, str):
        roas_str = roas
    else:
        roas_str = fmt_roas(roas) if roas is not None else "–"
    
    text = f"🚨 Hey! Had to kill {clean_name} in {stage}\n{reason}\nCTR was {ctr_str}, ROAS {roas_str} - not hitting our targets"
    
    return SlackMessage(
        text=text,
        severity="warn",
        topic="alerts",
    )

def template_digest(date_label: str, stage_stats: Dict[str, Dict[str, Any]]) -> SlackMessage:
    title = f"📊 Daily Report • {date_label}"
    lines = []
    for stage, stats in stage_stats.items():
        nice = " | ".join(f"{k}={v}" for k, v in stats.items())
        lines.append(f"*{stage}:* {nice}")
    
    intro = f"Hey! Here's what happened yesterday ({date_label}):"
    
    return SlackMessage(
        text=f"📊 Daily Report - {date_label}",
        blocks=build_basic_blocks(title, [intro] + lines, severity="info", footer=_ts_footer()),
        topic="digest",
    )

class Outbox:
    def __init__(self, path: str) -> None:
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
    def __init__(self, rate_qps: float, burst: int) -> None:
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
    def __init__(self) -> None:
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
        self._base_qps = TOKEN_RATE_QPS
        self.batch_buf: List[SlackMessage] = []
        self.batch_started: Optional[float] = None
        self._sender_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        if SLACK_TEMP_DISABLED:
            self.enabled = False
            self.dry_run = True
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
        depth = self.outbox.queue_depth()
        if msg.severity == "info" and depth >= SLACK_INFO_QUEUE_DROP:
            _log(f"[SLACK DROP] queue backlog {depth}, skipping info message: {msg.sanitized_text()}")
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
        _log(f"[SLACK {'SENT' if sent_any else 'QUEUED'} {msg.severity}/{msg.topic}] {msg.sanitized_text()}")

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
        depth = self.outbox.queue_depth()
        if depth and depth % 25 == 0:
            _log(f"[SLACK QUEUE] {depth} messages pending (topic={topic})")

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
                retry_after = resp.headers.get("Retry-After")
                try:
                    wait = float(retry_after)
                except (TypeError, ValueError):
                    wait = max(5.0, SLACK_BACKOFF_BASE * (2 ** n))
                wait = min(wait, SLACK_BACKOFF_CAP)
                self._on_failure()
                self._penalize_rate_limit(webhook, wait, topic)
                _log(f"[SLACK WARN] 429 rate limit, backing off {wait:.1f}s (webhook topic={topic})")
                time.sleep(wait)
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
        for tb in self.tb_by_webhook.values():
            with tb.lock:
                tb.rate = min(self._base_qps, tb.rate * 1.2)

    def _penalize_rate_limit(self, webhook: str, wait: float, topic: str) -> None:
        penalty = max(wait, 1.0)
        now = time.time()
        self.cb_open_until = max(self.cb_open_until or 0.0, now + penalty)
        tb = self.tb_by_webhook.get(webhook)
        if tb:
            with tb.lock:
                tb.tokens = 0.0
                tb.updated = time.monotonic()
                tb.rate = max(0.05, min(tb.rate, 1.0 / penalty))
        else:
            self.tb_by_webhook[webhook] = TokenBucket(max(0.05, 1.0 / penalty), TOKEN_BURST)


_client: Optional[SlackClient] = None

def client() -> SlackClient:
    global _client
    if _client is None:
        _client = SlackClient()
    return _client

def notify(text: str, severity: Literal["info", "warn", "error"] = "info", topic: Union[str, Literal["default", "alerts", "digest", "scale"]] = "default", ttl_seconds: Optional[int] = None) -> None:
    if SLACK_TEMP_DISABLED:
        sanitized = _sanitize_line(text)
        _log(f"[SLACK DISABLED {severity}/{topic}] {sanitized}")
        return

    msg = SlackMessage(text=_sanitize_line(text), severity=severity, topic=topic, ttl_seconds=ttl_seconds)
    msg.text = msg.sanitized_text()
    client().notify(msg)

def alert_kill(stage: str = "ASC+", entity_name: str = "", reason: str = "", metrics: Optional[Dict[str, Any]] = None, link: Optional[str] = None) -> None:
    client().notify(template_kill(stage, entity_name, reason, metrics, link))

def post_digest(date_label: str, stage_stats: Dict[str, Dict[str, Any]]) -> None:
    client().notify(template_digest(date_label, stage_stats))

def alert_error(error_msg: str) -> None:
    text = f"🚨 Something went wrong: {error_msg}\nNeed to check this ASAP"
    
    msg = SlackMessage(
        text=text,
        severity="error",
        topic="alerts",
    )
    client().notify(msg)

def build_ads_snapshot(rows_today: List[Dict[str, Any]], rows_lifetime: List[Dict[str, Any]], local_tz: str) -> List[str]:
    from datetime import datetime, timezone
    import re
    
    from infrastructure.utils import safe_f
    
    def _prettify_ad_name(name: str) -> str:
        name = re.sub(r'^\[(TEST|VALID|SCALE)\]\s*', '', name)
        name = re.sub(r'DynamicGreenScreenEffect', 'DGS Effect', name)
        name = re.sub(r'Green Screen Effect Template', 'Green Screen Template', name)
        if len(name) > 50:
            name = name[:47] + "..."
        return name
    
    def _fmt_eur(amount: Optional[float]) -> str:
        if amount is None:
            return "-"
        return f"€{amount:,.2f}"
    
    lifetime_by_id = {}
    for row in rows_lifetime or []:
        ad_id = row.get("ad_id")
        if ad_id:
            lifetime_by_id[str(ad_id)] = row
    
    ad_lines = []
    integrity_issues = 0
    
    for today_row in rows_today or []:
        ad_id = today_row.get("ad_id")
        ad_name = today_row.get("ad_name", "")
        spend_today = safe_f(today_row.get("spend"))
        
        lifetime_row = lifetime_by_id.get(str(ad_id), {})
        spend_life = safe_f(lifetime_row.get("spend")) if lifetime_row else None
        
        if spend_life is not None and spend_life < spend_today:
            integrity_issues += 1
        
        purch_today = 0
        for action in (today_row.get("actions") or []):
            if action.get("action_type") == "purchase":
                purch_today += int(action.get("value", 0))
        
        clean_name = _prettify_ad_name(ad_name)
        spend_life_display = _fmt_eur(spend_life) if spend_life is not None else "–"
        ad_lines.append(f"• {clean_name}: {_fmt_eur(spend_today)} today, {spend_life_display} all-time")
    
    def sort_key(item):
        try:
            spend_part = item.split(": €")[1].split(" today")[0] if ": €" in item else "0"
            spend_val = float(spend_part.replace(",", ""))
            return -spend_val
        except Exception:
            return 0
    
    ad_lines.sort(key=sort_key)
    
    if len(ad_lines) > 6:
        display_lines = ad_lines[:4]
        display_lines.append(f"+ {len(ad_lines) - 4} more in thread")
    else:
        display_lines = ad_lines[:4]
    
    if integrity_issues > 0:
        display_lines.append(f"note: {integrity_issues} ad shows lifetime < today (API window or join issue)")
    
    return display_lines

def alert_queue_empty() -> None:
    text = "🚨 URGENT: No more creatives in the queue!\nASC+ generates creatives dynamically - this alert should not appear"
    
    msg = SlackMessage(
        text=text,
        severity="error",
        topic="alerts",
    )
    client().notify(msg)

def alert_system_health(issue: str) -> None:
    text = f"🔧 System issue detected: {issue}\nThe automation might be affected"
    
    msg = SlackMessage(
        text=text,
        severity="warn",
        topic="alerts",
    )
    client().notify(msg)

def alert_ad_account_health_critical(account_id: str, issues: List[str]) -> None:
    issues_text = "\n".join([f"• {issue}" for issue in issues])
    text = f"🚨 CRITICAL: Ad Account Health Issues Detected\nAccount: {account_id}\n\nIssues:\n{issues_text}\n\n⚠️ Your ad account may be disabled or restricted. Check immediately!"
    
    msg = SlackMessage(
        text=text,
        severity="error",
        topic="alerts",
    )
    client().notify(msg)

def alert_ad_account_health_warning(account_id: str, warnings: List[str]) -> None:
    warnings_text = "\n".join([f"• {warning}" for warning in warnings])
    text = f"⚠️ Ad Account Health Warning\nAccount: {account_id}\n\nWarnings:\n{warnings_text}\n\nMonitor these issues to prevent account problems."
    
    msg = SlackMessage(
        text=text,
        severity="warn",
        topic="alerts",
    )
    client().notify(msg)

def alert_spend_cap_approaching(account_id: str, spent: float, cap: float, currency: str = "EUR") -> None:
    percentage = (spent / cap) * 100 if cap > 0 else 0
    text = f"📊 Spend Cap Warning\nAccount: {account_id}\nSpent: {spent:.2f} {currency} ({percentage:.1f}% of cap)\nCap: {cap:.2f} {currency}\n\n⚠️ Approaching spend limit. Consider increasing cap or monitoring spend."
    
    msg = SlackMessage(
        text=text,
        severity="warn",
        topic="alerts",
    )
    client().notify(msg)

def alert_budget_alert(entity_name: str, current_budget: float, target_budget: float) -> None:
    clean_name = prettify_ad_name(entity_name)
    change_pct = ((target_budget - current_budget) / current_budget) * 100
    direction = "up" if change_pct > 0 else "down"
    
    text = f"💰 Budget change for {clean_name}\n{change_pct:+.1f}% {direction} to {fmt_eur(target_budget)}/day"
    
    msg = SlackMessage(
        text=text,
        severity="info",
        topic="alerts",
    )
    client().notify(msg)

__all__ = [
    "SlackClient",
    "SlackMessage",
    "Outbox",
    "TokenBucket",
    "notify",
    "alert_kill",
    "alert_error",
    "alert_queue_empty",
    "alert_system_health",
    "alert_budget_alert",
    "build_ads_snapshot",
    "post_digest",
    "format_run_header",
    "format_stage_line",
    "prettify_ad_name",
    "fmt_eur",
    "fmt_pct",
    "fmt_roas",
    "fmt_int",
    "post_run_header_and_get_thread_ts",
    "post_thread_ads_snapshot",
    "client",
]


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Slack outbox utilities")
    parser.add_argument(
        "--flush",
        action="store_true",
        help="Drain the pending Slack outbox messages respecting rate limits.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Maximum seconds to wait while draining the outbox.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print current queue depth and exit.",
    )
    args = parser.parse_args()

    slack_client = client()
    depth = slack_client.outbox.queue_depth()
    _log(f"[SLACK CLI] current queue depth: {depth}")

    if args.flush:
        deadline = time.time() + max(0.0, args.timeout)
        _log(f"[SLACK CLI] draining queue (timeout {args.timeout}s)")
        slack_client.flush(timeout=args.timeout)
        depth_after = slack_client.outbox.queue_depth()
        if depth_after > 0 and time.time() >= deadline:
            _log(f"[SLACK CLI] drain timed out with {depth_after} messages still pending")
        else:
            _log(f"[SLACK CLI] drain complete, {depth_after} messages remaining")
    elif not args.status:
        parser.print_help()
