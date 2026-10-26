"""Client for the SHMU open data server (Slovak Hydrometeorological Institute).

https://opendata.shmu.sk — free, no registration, CC-BY 4.0.

Why this exists: Open-Meteo is a reanalysis/model product and its irradiance is
biased against ground measurement, by a location-dependent amount. Measured
against SHMU pyranometers over five days at six Slovak sites the bias ranged
from -12.9% (Bratislava) to +3.3% (Kosice), mean -6.1%. For an investment case
that error propagates straight into yield, payback and sizing, so the workflow
needs a measured reference to correct against.

What the server offers that matters here: the automatic weather station feed
``meteorology/climate/now`` carries ``zglo``, global solar radiation in W/m2, as
a one-minute average, for 44 stations across Slovakia, published every five
minutes. The catch is that this directory is a rolling window of roughly 30 days
— there is no deep sub-hourly archive — so it is a calibration source, not a
substitute for the historical weather series.
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from urllib.parse import quote

# The server presents an incomplete TLS chain, so plain HTTP is used and the
# payload is treated as untrusted public data (it is only numeric measurements).
SHMU_BASE_URL = "http://opendata.shmu.sk/meteorology/climate/now/data"
SHMU_METADATA_URL = "http://opendata.shmu.sk/meteorology/climate/now/metadata"

GLOBAL_RADIATION_COLUMN = "zglo"

# Snapshots are published every 5 minutes and each one carries roughly 14
# minutes of history, so sampling every 10 minutes still covers the timeline.
SNAPSHOT_MINUTES = (0, 10, 20, 30, 40, 50)


@dataclass(frozen=True)
class ShmuStation:
    ind_kli: int
    name: str
    latitude: float
    longitude: float
    altitude_m: float

    def distance_km(self, lat: float, lon: float) -> float:
        """Great-circle distance to a site, for nearest-station selection."""
        r = 6371.0
        p1, p2 = math.radians(self.latitude), math.radians(lat)
        dp = math.radians(lat - self.latitude)
        dl = math.radians(lon - self.longitude)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# SHMU synoptic stations (www.shmu.sk, "Zoznam synoptickych stanic"). Only those
# that were observed reporting zglo in the open data feed are listed, so the
# nearest-station search cannot pick a site with no pyranometer.
SHMU_STATIONS: tuple[ShmuStation, ...] = (
    ShmuStation(11812, "Maly Javornik", 48.2558, 17.1538, 586),
    ShmuStation(11813, "Bratislava - Koliba", 48.1686, 17.1106, 286),
    ShmuStation(11816, "Bratislava - letisko", 48.1717, 17.2000, 131),
    ShmuStation(11819, "Jaslovske Bohunice", 48.4867, 17.6708, 176),
    ShmuStation(11826, "Piestany", 48.6131, 17.8328, 163),
    ShmuStation(11841, "Zilina - Dolny Hricov", 49.2319, 18.6178, 309),
    ShmuStation(11855, "Nitra - Velke Janikovce", 48.2806, 18.1356, 135),
    ShmuStation(11856, "Mochovce", 48.2894, 18.4561, 261),
    ShmuStation(11858, "Hurbanovo", 47.8733, 18.1944, 115),
    ShmuStation(11867, "Prievidza", 48.7697, 18.5939, 260),
    ShmuStation(11880, "Dudince", 48.1692, 18.8761, 139),
    ShmuStation(11900, "Ziar nad Hronom", 48.5861, 18.8522, 275),
    ShmuStation(11918, "Liesek", 49.3694, 19.6794, 692),
    ShmuStation(11927, "Bolkovce", 48.3389, 19.7364, 214),
    ShmuStation(11930, "Lomnicky stit", 49.1953, 20.2150, 2635),
    ShmuStation(11933, "Strbske Pleso", 49.1217, 20.0603, 1322),
    ShmuStation(11934, "Poprad", 49.0689, 20.2456, 694),
    ShmuStation(11938, "Telgart", 48.8486, 20.1892, 901),
    ShmuStation(11952, "Ganovce", 49.0333, 20.3167, 703),
    ShmuStation(11958, "Kojsovska hola", 48.7833, 20.9833, 1244),
    ShmuStation(11968, "Kosice - letisko", 48.6722, 21.2225, 230),
    ShmuStation(11976, "Stropkov", 49.2156, 21.6500, 216),
    ShmuStation(11978, "Milhostov", 48.6631, 21.7239, 105),
    ShmuStation(11993, "Kamenica nad Cirochou", 48.9389, 22.0061, 176),
)

# Rough bounding box of Slovakia, used to decide whether SHMU is relevant at all.
SLOVAKIA_BBOX = (47.7, 16.8, 49.7, 22.6)  # lat_min, lon_min, lat_max, lon_max


def is_in_slovakia(lat: float, lon: float) -> bool:
    lat_min, lon_min, lat_max, lon_max = SLOVAKIA_BBOX
    return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max


def nearest_station(lat: float, lon: float, *, max_distance_km: float = 120.0) -> ShmuStation | None:
    """Closest radiation-reporting station, or None when the site is too far."""
    if not SHMU_STATIONS:
        return None
    best = min(SHMU_STATIONS, key=lambda s: s.distance_km(lat, lon))
    return best if best.distance_km(lat, lon) <= max_distance_km else None


def _snapshot_url(day: date, hour: int, minute: int) -> str:
    name = f"aws1min - {day:%Y-%m-%d} {hour:02d}-{minute:02d}-00.json"
    return f"{SHMU_BASE_URL}/{day:%Y%m%d}/{quote(name)}"


def _fetch_snapshot(session, url: str, station_id: int, timeout: int) -> list[tuple[str, float]]:
    try:
        resp = session.get(url, timeout=timeout)
        if resp.status_code != 200:
            return []
        rows = resp.json().get("data") or []
    except Exception:
        # Individual snapshots go missing routinely; the caller works with
        # whatever coverage it gets and reports it.
        return []

    out = []
    for row in rows:
        if row.get("ind_kli") != station_id:
            continue
        value = row.get(GLOBAL_RADIATION_COLUMN)
        stamp = row.get("minuta")
        if value is None or stamp is None:
            continue
        try:
            out.append((str(stamp), float(value)))
        except (TypeError, ValueError):
            continue
    return out


def fetch_global_radiation(
    station: ShmuStation,
    start: date,
    end: date,
    *,
    max_workers: int = 16,
    timeout: int = 25,
):
    """Measured global radiation for one station over ``[start, end]``.

    Returns a ``pandas.Series`` of W/m2 indexed by local timestamp at one-minute
    resolution, deduplicated and sorted. Missing snapshots are skipped silently;
    inspect the length of the result to judge coverage.
    """
    import pandas as pd
    import requests

    days = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)

    urls = [
        _snapshot_url(day, hour, minute)
        for day in days
        for hour in range(24)
        for minute in SNAPSHOT_MINUTES
    ]

    session = requests.Session()
    records: list[tuple[str, float]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for chunk in pool.map(
            lambda u: _fetch_snapshot(session, u, station.ind_kli, timeout), urls
        ):
            records.extend(chunk)

    if not records:
        return pd.Series(dtype=float, name="shmu_ghi_w_m2")

    frame = pd.DataFrame(records, columns=["timestamp", "value"]).drop_duplicates("timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()
    series = frame["value"].astype(float)
    series.name = "shmu_ghi_w_m2"
    return series


def resample_to(series, freq: str = "15min"):
    """Average one-minute measurements onto the workflow time grid."""
    if series is None or len(series) == 0:
        return series
    out = series.resample(freq).mean().dropna()
    out.name = series.name
    return out


def available_window(*, probe_days: int = 45, timeout: int = 20) -> tuple[date, date] | None:
    """Discover the rolling window the server currently exposes.

    The feed keeps roughly the last 30 days; the exact span moves daily, so it is
    probed rather than assumed.
    """
    import requests

    session = requests.Session()
    today = datetime.utcnow().date()

    def exists(offset: int) -> bool:
        day = today - timedelta(days=offset)
        try:
            return session.get(f"{SHMU_BASE_URL}/{day:%Y%m%d}/", timeout=timeout).status_code == 200
        except Exception:
            return False

    # The window is contiguous, so probing every day costs 45 sequential round
    # trips to learn two dates. Find the newest published day, then binary search
    # for the far edge: about eight requests instead of forty-five.
    newest = next((offset for offset in range(4) if exists(offset)), None)
    if newest is None:
        return None

    low, high = newest, newest
    step = 1
    while high + step < probe_days and exists(high + step):
        high += step
        step *= 2
    lo, hi = high, min(high + step, probe_days)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if exists(mid):
            lo = mid
        else:
            hi = mid

    return today - timedelta(days=lo), today - timedelta(days=newest)


def bias_report(modelled, measured) -> dict:
    """Compare a modelled irradiance series against SHMU measurement.

    Both inputs must share a time index. Only daylight steps are scored, since
    night-time zeros would flatter every metric.
    """
    import numpy as np
    import pandas as pd

    joined = pd.DataFrame({"model": modelled, "measured": measured}).dropna()
    day = joined[joined["measured"] > 20.0]
    if len(day) < 24:
        return {
            "status": "insufficient_overlap",
            "overlapping_steps": int(len(joined)),
            "daylight_steps": int(len(day)),
        }

    err = day["model"] - day["measured"]
    mean_measured = float(day["measured"].mean())
    denom = ((day["measured"] - mean_measured) ** 2).sum()

    return {
        "status": "ok",
        "daylight_steps": int(len(day)),
        "mean_measured_w_m2": round(mean_measured, 2),
        "mean_modelled_w_m2": round(float(day["model"].mean()), 2),
        "bias_w_m2": round(float(err.mean()), 2),
        "bias_pct": round(100.0 * float(err.mean()) / mean_measured, 2),
        "mae_w_m2": round(float(err.abs().mean()), 2),
        "mae_pct": round(100.0 * float(err.abs().mean()) / mean_measured, 2),
        "rmse_w_m2": round(float(np.sqrt((err**2).mean())), 2),
        "correlation": round(float(day["model"].corr(day["measured"])), 4),
        "r2": round(float(1.0 - (err**2).sum() / denom), 4) if denom > 0 else None,
        # Multiply modelled irradiance by this to remove the mean bias.
        "recommended_scale_factor": round(mean_measured / float(day["model"].mean()), 4)
        if float(day["model"].mean()) > 1e-6
        else None,
    }
