"""Public OKTE day-ahead prices for a historical load window."""
from __future__ import annotations

import json
from datetime import date, timedelta
from urllib import parse as urlparse
from urllib import request as urlrequest

import pandas as pd

OKTE_DAM_URL = "https://isot.okte.sk/api/v1/dam/results"
SITE_TZ = "Europe/Bratislava"
_CHUNK_DAYS = 90


def fetch_okte_dam_prices(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    *,
    timeout: int = 60,
) -> pd.DataFrame:
    """Return ``datetime, price_eur_per_kwh`` covering ``[start, end]`` in site time.

    The ISOT DAM API is keyless. Periods were hourly until 2025-10-01 and
    15-minute afterwards; both are stamped at the *start* of the delivery
    period so they line up with a typical load CSV.
    """
    start_d = pd.Timestamp(start).normalize().date()
    end_d = pd.Timestamp(end).normalize().date()
    if end_d < start_d:
        raise ValueError(f"OKTE window is empty: {start_d} .. {end_d}")

    frames: list[pd.DataFrame] = []
    cursor = start_d
    while cursor <= end_d:
        chunk_end = min(cursor + timedelta(days=_CHUNK_DAYS - 1), end_d)
        frames.append(_fetch_chunk(cursor, chunk_end, timeout=timeout))
        cursor = chunk_end + timedelta(days=1)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["datetime"], keep="last").sort_values("datetime")
    return out.reset_index(drop=True)


def align_okte_to_load(load_df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Attach OKTE prices to every load timestamp (forward-fill hourly slots)."""
    if load_df.empty:
        raise ValueError("Load CSV has no rows to attach OKTE prices to")
    if prices.empty:
        raise ValueError("OKTE returned no prices for the load horizon")
    left = load_df.sort_values("datetime").reset_index(drop=True)
    right = prices[["datetime", "price_eur_per_kwh"]].sort_values("datetime")
    merged = pd.merge_asof(left, right, on="datetime", direction="backward")
    if merged["price_eur_per_kwh"].isna().any():
        merged["price_eur_per_kwh"] = merged["price_eur_per_kwh"].bfill()
    missing = int(merged["price_eur_per_kwh"].isna().sum())
    if missing:
        raise ValueError(
            f"OKTE prices do not cover {missing} load timestamps "
            f"({left['datetime'].min()} .. {left['datetime'].max()})"
        )
    return merged


def _fetch_chunk(day_from: date, day_to: date, *, timeout: int) -> pd.DataFrame:
    query = urlparse.urlencode(
        {"deliveryDayFrom": day_from.isoformat(), "deliveryDayTo": day_to.isoformat()}
    )
    req = urlrequest.Request(
        f"{OKTE_DAM_URL}?{query}",
        headers={"User-Agent": "uc32-cost-optimizer/1.0"},
    )
    with urlrequest.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not rows:
        raise ValueError(f"OKTE returned no DAM results for {day_from} .. {day_to}")
    raw = pd.DataFrame(rows)
    if "price" not in raw.columns or "deliveryStart" not in raw.columns:
        raise ValueError("OKTE response is missing price/deliveryStart")
    start = (
        pd.to_datetime(raw["deliveryStart"], utc=True, errors="coerce")
        .dt.tz_convert(SITE_TZ)
        .dt.tz_localize(None)
    )
    price_mwh = pd.to_numeric(raw["price"], errors="coerce")
    out = pd.DataFrame(
        {
            "datetime": start,
            "price_eur_per_kwh": (price_mwh / 1000.0).round(6),
        }
    ).dropna(subset=["datetime", "price_eur_per_kwh"])
    if out.empty:
        raise ValueError(f"OKTE returned no usable prices for {day_from} .. {day_to}")
    return out
