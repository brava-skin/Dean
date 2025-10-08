from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import pytz


# -----------------------
# Clock primitives
# -----------------------
class Clock:
    def now_utc(self) -> datetime:
        raise NotImplementedError


class RealClock(Clock):
    def now_utc(self) -> datetime:
        return datetime.now(timezone.utc)


class FixedClock(Clock):
    def __init__(self, dt_utc: datetime):
        if dt_utc.tzinfo is None:
            dt_utc = dt_utc.replace(tzinfo=timezone.utc)
        self._dt = dt_utc.astimezone(timezone.utc)

    @staticmethod
    def from_env(var: str = "NOW_UTC") -> Optional["FixedClock"]:
        val = os.getenv(var)
        if not val:
            return None
        dt = _parse_any_datetime(val)
        return FixedClock(dt)

    def now_utc(self) -> datetime:
        return self._dt


def _parse_any_datetime(s: str) -> datetime:
    s = s.strip()
    if re.fullmatch(r"\d{10}", s):
        return datetime.fromtimestamp(int(s), tz=timezone.utc)
    if re.fullmatch(r"\d{13}", s):
        return datetime.fromtimestamp(int(s) / 1000.0, tz=timezone.utc)
    s = s.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError as e:
        raise ValueError(f"Invalid datetime: {s!r}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _require_tz(name: str) -> pytz.BaseTzInfo:
    try:
        return pytz.timezone(name)
    except Exception as e:
        raise ValueError(f"Invalid IANA timezone: {name!r}") from e


# -----------------------
# Config (Time & Currency)
# -----------------------
@dataclass(frozen=True)
class TZConfig:
    account_tz: str
    user_tz: str
    # Optional audience tz (for mapping windows like Chicago → Amsterdam)
    audience_tz: str = "America/Chicago"

    @staticmethod
    def from_env() -> "TZConfig":
        # NEW DEFAULT: Europe/Amsterdam for the ad account
        acc = os.getenv("ACCOUNT_TIMEZONE") or os.getenv("TIMEZONE") or "Europe/Amsterdam"
        usr = os.getenv("USER_TIMEZONE") or "Europe/Amsterdam"
        aud = os.getenv("AUDIENCE_TIMEZONE") or "America/Chicago"
        _require_tz(acc)
        _require_tz(usr)
        _require_tz(aud)
        return TZConfig(acc, usr, aud)


@dataclass(frozen=True)
class CurrencyConfig:
    # NEW: currency helpers (defaults to EUR account, product USD)
    account_currency: str = "EUR"
    product_currency: str = "USD"
    # Prefer live rate via env; fallback constant if not set
    usd_eur_rate: float = float(os.getenv("USD_EUR_RATE") or os.getenv("EXCHANGE_RATE_USD_EUR") or 0.92)

    @staticmethod
    def from_env() -> "CurrencyConfig":
        acc = (os.getenv("ACCOUNT_CURRENCY") or "EUR").upper()
        prod = (os.getenv("PRODUCT_CURRENCY") or "USD").upper()
        rate_env = os.getenv("USD_EUR_RATE") or os.getenv("EXCHANGE_RATE_USD_EUR")
        try:
            rate = float(rate_env) if rate_env else 0.92
        except Exception:
            rate = 0.92
        return CurrencyConfig(acc, prod, rate)


# -----------------------
# Timekit core
# -----------------------
class Timekit:
    def __init__(self, tzcfg: Optional[TZConfig] = None, clock: Optional[Clock] = None,
                 curr: Optional[CurrencyConfig] = None):
        self.tzcfg = tzcfg or TZConfig.from_env()
        self.clock = clock or FixedClock.from_env() or RealClock()
        self.curr = curr or CurrencyConfig.from_env()

        self._acc = _require_tz(self.tzcfg.account_tz)
        self._usr = _require_tz(self.tzcfg.user_tz)
        self._aud = _require_tz(self.tzcfg.audience_tz)

    # --- now() helpers
    def now_utc(self) -> datetime:
        return self.clock.now_utc()

    def now_account(self) -> datetime:
        return self.now_utc().astimezone(self._acc)

    def now_user(self) -> datetime:
        return self.now_utc().astimezone(self._usr)

    def now_audience(self) -> datetime:
        return self.now_utc().astimezone(self._aud)

    # --- day boundaries & ymd
    @staticmethod
    def start_of_day(dt: datetime) -> datetime:
        if dt.tzinfo is None:
            raise ValueError("Timezone-aware datetime required")
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def end_of_day(dt: datetime) -> datetime:
        return Timekit.start_of_day(dt) + timedelta(days=1)

    @staticmethod
    def ymd(dt: datetime) -> str:
        if dt.tzinfo is None:
            raise ValueError("Timezone-aware datetime required")
        return dt.date().isoformat()

    def today_ymd_account(self) -> str:
        return self.ymd(self.now_account())

    def yesterday_ymd_account(self) -> str:
        return self.ymd(self.now_account() - timedelta(days=1))

    def today_ymd_user(self) -> str:
        return self.ymd(self.now_user())

    def yesterday_ymd_user(self) -> str:
        return self.ymd(self.now_user() - timedelta(days=1))

    # --- windows
    def calendar_day_window_account(self, days_ago: int = 0) -> Tuple[datetime, datetime]:
        base = self.now_account() - timedelta(days=days_ago)
        s = self.start_of_day(base)
        e = self.end_of_day(base)
        return s, e

    def rolling_hours_window_account(self, hours: int) -> Tuple[datetime, datetime]:
        if hours <= 0:
            raise ValueError("hours must be >= 1")
        end = self.now_account()
        return end - timedelta(hours=hours), end

    def week_to_date_account(self, week_start: int = 0) -> Tuple[datetime, datetime]:
        if not 0 <= week_start <= 6:
            raise ValueError("week_start must be 0..6 (Mon=0)")
        now = self.now_account()
        delta = (now.weekday() - week_start) % 7
        s = self.start_of_day(now - timedelta(days=delta))
        return s, self.end_of_day(now)

    def month_to_date_account(self) -> Tuple[datetime, datetime]:
        now = self.now_account()
        s = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return s, self.end_of_day(now)

    # --- Meta-style ranges (account tz)
    def meta_range_today(self) -> Dict[str, str]:
        d = self.today_ymd_account()
        return {"since": d, "until": d}

    def meta_range_yesterday(self) -> Dict[str, str]:
        d = self.yesterday_ymd_account()
        return {"since": d, "until": d}

    def meta_range_last_n_full_days(self, n: int) -> Dict[str, str]:
        if n <= 0:
            raise ValueError("n must be >= 1")
        today_acc = self.now_account().date()
        until = (today_acc - timedelta(days=1)).isoformat()
        since = (today_acc - timedelta(days=n)).isoformat()
        return {"since": since, "until": until}

    def meta_range_wtd(self, week_start: int = 0) -> Dict[str, str]:
        s, e = self.week_to_date_account(week_start)
        return {"since": self.ymd(s), "until": self.ymd(e)}

    def meta_range_mtd(self) -> Dict[str, str]:
        s, e = self.month_to_date_account()
        return {"since": self.ymd(s), "until": self.ymd(e)}

    # --- DST / tick
    def dst_state_account(self) -> Dict[str, Union[bool, int]]:
        now = self.now_account()
        off = now.utcoffset() or timedelta(0)
        return {"is_dst": bool(now.dst()), "utc_offset_minutes": int(off.total_seconds() // 60)}

    def tick_key(self, period_minutes: int = 30, for_user: bool = False) -> str:
        if period_minutes <= 0:
            raise ValueError("period_minutes must be >= 1")
        now = self.now_user() if for_user else self.now_account()
        bucket_min = (now.minute // period_minutes) * period_minutes
        t0 = now.replace(minute=bucket_min, second=0, microsecond=0)
        label = "USR" if for_user else "ACC"
        return f"{label}@{t0.isoformat()}/{period_minutes}m"

    def describe_tick(self, period_minutes: int = 30) -> str:
        a = self.tick_key(period_minutes, for_user=False)
        u = self.tick_key(period_minutes, for_user=True)
        ds = self.dst_state_account()
        return f"{a} | {u} | DST={int(ds['is_dst'])} off={ds['utc_offset_minutes']}m"

    def is_dst_fold_or_gap(self, dt_local: datetime) -> Dict[str, bool]:
        if dt_local.tzinfo is None:
            raise ValueError("Timezone-aware datetime required")
        tz = dt_local.tzinfo
        before = (dt_local - timedelta(minutes=30)).astimezone(tz)
        after = (dt_local + timedelta(minutes=30)).astimezone(tz)
        return {"possible_gap": after.utcoffset() != dt_local.utcoffset(),
                "possible_fold": before.utcoffset() != dt_local.utcoffset()}

    # --- window parsing / expansion (24h clock strings -> hours)
    @staticmethod
    def parse_window_str(s: str) -> Tuple[int, int]:
        m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*-\s*(\d{1,2}):(\d{2})\s*", s)
        if not m:
            raise ValueError(f"Invalid window: {s!r}")
        h1, m1, h2, m2 = (int(m.group(i)) for i in (1, 2, 3, 4))
        if not (0 <= h1 <= 23 and 0 <= h2 <= 23 and 0 <= m1 <= 59 and 0 <= m2 <= 59):
            raise ValueError("Hours must be 0..23 and minutes 0..59")
        return h1, h2

    @staticmethod
    def expand_windows_to_hours(windows: Sequence[str]) -> Tuple[int, ...]:
        hours: List[int] = []
        for w in windows:
            a, b = Timekit.parse_window_str(w)
            if a <= b:
                hours.extend(range(a, b + 1))
            else:
                hours.extend(list(range(a, 24)) + list(range(0, b + 1)))
        return tuple(sorted(set(hours)))

    # --- Should pause (true if current hour IS in pause list)
    def should_pause_now(
        self,
        stage_hours: Union[Tuple[int, ...], Dict[int, Tuple[int, ...]]],
        holidays_ymd: Optional[Iterable[str]] = None,
    ) -> bool:
        now = self.now_account()
        if holidays_ymd and self.ymd(now) in set(holidays_ymd):
            return False
        hour = now.hour
        if isinstance(stage_hours, dict):
            hours = stage_hours.get(now.weekday(), tuple())
        else:
            hours = stage_hours
        return hour in hours

    # --- ISO with no micros
    @staticmethod
    def iso_no_micro(dt: datetime) -> str:
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(dt.tzinfo).replace(microsecond=0).isoformat()

    # --- Banner
    def banner(self) -> str:
        n = self.now_utc()
        a = self.now_account()
        u = self.now_user()
        ds = self.dst_state_account()
        src = "FIXED" if isinstance(self.clock, FixedClock) else "REAL"
        return (
            f"UTC={self.iso_no_micro(n)} [{src}]  "
            f"ACC={self.iso_no_micro(a)}({self.tzcfg.account_tz})  "
            f"USR={self.iso_no_micro(u)}({self.tzcfg.user_tz})  "
            f"DST={int(ds['is_dst'])} off={ds['utc_offset_minutes']}m  "
            f"CURR[acct={self.curr.account_currency}, prod={self.curr.product_currency}, usd→eur≈{self.curr.usd_eur_rate:.4f}]"
        )

    # ---------------------------------------------------------------------
    # NEW: Hour mapping helpers (e.g., Chicago audience hours -> Amsterdam)
    # ---------------------------------------------------------------------
    @staticmethod
    def _map_hours_between_tz(
        hours: Sequence[int],
        src_tz: pytz.BaseTzInfo,
        dst_tz: pytz.BaseTzInfo,
        on_date: Optional[date] = None,
    ) -> Tuple[int, ...]:
        """
        Map a set of whole hours defined in src_tz to the corresponding hours in dst_tz
        preserving the *absolute* instants. Handles DST properly.

        on_date: date in dst_tz on which to interpret the mapping (defaults to today in dst).
        """
        if not hours:
            return tuple()

        if on_date is None:
            # Use today's date in dst_tz
            now_dst = datetime.now(timezone.utc).astimezone(dst_tz)
            on_date = now_dst.date()

        mapped: List[int] = []
        # Use the *same* civil date in src_tz that corresponds to that day in dst_tz midnight
        dst_midnight = dst_tz.localize(datetime(on_date.year, on_date.month, on_date.day, 0, 0, 0))
        # Convert midnight in dst to src to figure which src civil date overlaps
        src_ref = dst_midnight.astimezone(src_tz)
        src_date = src_ref.date()

        for h in sorted(set(int(h) % 24 for h in hours)):
            src_local = src_tz.localize(datetime(src_date.year, src_date.month, src_date.day, h, 0, 0))
            # Convert that instant to dst tz and collect the hour
            dst_local = src_local.astimezone(dst_tz)
            mapped.append(dst_local.hour)

        return tuple(sorted(set(mapped)))

    def audience_hours_to_account(
        self, audience_hours: Sequence[int], on_date: Optional[date] = None
    ) -> Tuple[int, ...]:
        """Convenience: map America/Chicago hours → Account (Europe/Amsterdam)."""
        return self._map_hours_between_tz(audience_hours, self._aud, self._acc, on_date)

    # ---------------------------------------------------------------------
    # NEW: Currency helpers (USD → Account / generic)
    # ---------------------------------------------------------------------
    def convert(self, amount: float, from_ccy: str, to_ccy: str) -> float:
        from_ccy = from_ccy.upper()
        to_ccy = to_ccy.upper()
        if from_ccy == to_ccy:
            return float(amount)

        # Currently we only handle USD<->EUR simply; extend as needed
        if from_ccy == "USD" and to_ccy == "EUR":
            return float(amount) * float(self.curr.usd_eur_rate)
        if from_ccy == "EUR" and to_ccy == "USD":
            rate = 1.0 / float(self.curr.usd_eur_rate) if self.curr.usd_eur_rate else 0.0
            return float(amount) * rate

        # If more currencies are needed, plug in your FX source here or raise.
        raise ValueError(f"Unsupported conversion {from_ccy}->{to_ccy}")

    def usd_to_account(self, amount_usd: float) -> float:
        """Convert USD amount to account currency using config."""
        if self.curr.account_currency == "EUR":
            return self.convert(amount_usd, "USD", "EUR")
        if self.curr.account_currency == "USD":
            return float(amount_usd)
        # Extend if you ever switch account currency to something else
        raise ValueError(f"Unsupported account currency {self.curr.account_currency}")

    def tripwire_threshold_account(self, product_price_usd: float, multiple: float = 2.0) -> float:
        """
        Example: 'Instant tripwire: spend ≥ 2× product price (USD→EUR) & 0 purchases'
        Returns the spend threshold in *account currency*.
        """
        return self.usd_to_account(product_price_usd * multiple)


# -----------------------
# Singleton accessors
# -----------------------
@lru_cache(maxsize=1)
def _kit() -> Timekit:
    return Timekit()

def now_utc() -> datetime: return _kit().now_utc()
def now_account() -> datetime: return _kit().now_account()
def now_user() -> datetime: return _kit().now_user()
def today_ymd_account() -> str: return _kit().today_ymd_account()
def yesterday_ymd_account() -> str: return _kit().yesterday_ymd_account()
def today_ymd_user() -> str: return _kit().today_ymd_user()
def yesterday_ymd_user() -> str: return _kit().yesterday_ymd_user()
def meta_time_range_today_account() -> Dict[str, str]: return _kit().meta_range_today()
def meta_time_range_yesterday_account() -> Dict[str, str]: return _kit().meta_range_yesterday()
def meta_time_range_last_n_full_days_account(n: int) -> Dict[str, str]: return _kit().meta_range_last_n_full_days(n)
def meta_time_range_wtd_account(week_start: int = 0) -> Dict[str, str]: return _kit().meta_range_wtd(week_start)
def meta_time_range_mtd_account() -> Dict[str, str]: return _kit().meta_range_mtd()
def tick_key(period_minutes: int = 30, for_user_display: bool = False) -> str: return _kit().tick_key(period_minutes, for_user_display)
def describe_tick(period_minutes: int = 30) -> str: return _kit().describe_tick(period_minutes)
def dst_state_account() -> Dict[str, Union[bool, int]]: return _kit().dst_state_account()
def banner() -> str: return _kit().banner()

# NEW: public currency helpers
def usd_to_account(amount_usd: float) -> float: return _kit().usd_to_account(amount_usd)
def convert(amount: float, from_ccy: str, to_ccy: str) -> float: return _kit().convert(amount, from_ccy, to_ccy)
def tripwire_threshold_account(product_price_usd: float, multiple: float = 2.0) -> float:
    return _kit().tripwire_threshold_account(product_price_usd, multiple)

# NEW: public hour mapping (e.g., define Chicago prime hours and map them to AMS)
def audience_hours_to_account(hours: Sequence[int], on_date: Optional[date] = None) -> Tuple[int, ...]:
    return _kit().audience_hours_to_account(hours, on_date)


# -----------------------
# Misc legacy helpers
# -----------------------
def now_local(tz_name: str) -> datetime:
    """
    Return the current datetime in the provided IANA timezone (aware datetime).
    Used by main.py for human-friendly logging.
    """
    tz = _require_tz(tz_name)
    return now_utc().astimezone(tz)

def today_ymd(tz: timezone | None = None) -> str:
    """
    Legacy helper used by meta_client.py.
    Defaults to account timezone to match delivery/reporting windows.
    """
    if tz is None:
        return today_ymd_account()
    return now_utc().astimezone(tz).date().isoformat()

def yesterday_ymd(tz: timezone | None = None) -> str:
    """
    Legacy helper used by meta_client.py.
    Defaults to account timezone to match delivery/reporting windows.
    """
    if tz is None:
        return yesterday_ymd_account()
    return (now_utc().astimezone(tz) - timedelta(days=1)).date().isoformat()


# -----------------------
# Drift checker (unchanged)
# -----------------------
class DriftChecker:
    def __init__(self):
        self._t0_wall = time.time()
        self._t0_mono = time.monotonic()

    def drift_seconds(self) -> float:
        return (time.time() - self._t0_wall) - (time.monotonic() - self._t0_mono)

    def warn_if_drift(self, seconds: float = 1.0) -> Optional[float]:
        d = abs(self.drift_seconds())
        return d if d > seconds else None
