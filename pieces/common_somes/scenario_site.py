"""Load site / PV plant parameters from SoMES scenario.yaml."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_scenario_yaml(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    p = Path(str(path))
    if not p.is_file():
        return {}
    import yaml

    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return cfg if isinstance(cfg, dict) else {}


def site_location(cfg: dict[str, Any]) -> tuple[float | None, float | None]:
    site = cfg.get("site") if isinstance(cfg.get("site"), dict) else {}
    lat = site.get("latitude")
    lon = site.get("longitude")
    try:
        return (
            float(lat) if lat is not None else None,
            float(lon) if lon is not None else None,
        )
    except (TypeError, ValueError):
        return None, None


def pv_plant_params(cfg: dict[str, Any]) -> tuple[float | None, float | None]:
    """Return (installed_kwp, tilt_deg) from scenario pv / solar blocks."""
    pv = cfg.get("pv") if isinstance(cfg.get("pv"), dict) else {}
    solar = cfg.get("solar") if isinstance(cfg.get("solar"), dict) else {}
    kwp = pv.get("installed_kwp")
    if kwp is None:
        kwp = solar.get("capacity_kWp") or solar.get("capacity_kwp")
    tilt = pv.get("tilt_deg")
    if tilt is None and isinstance(pv.get("arrays"), list) and pv["arrays"]:
        first = pv["arrays"][0] if isinstance(pv["arrays"][0], dict) else {}
        tilt = first.get("tilt_deg")
    try:
        kwp_f = float(kwp) if kwp is not None else None
    except (TypeError, ValueError):
        kwp_f = None
    try:
        tilt_f = float(tilt) if tilt is not None else None
    except (TypeError, ValueError):
        tilt_f = None
    return kwp_f, tilt_f


def apply_scenario_site_to_mapping(
    values: dict[str, Any],
    cfg: dict[str, Any],
    *,
    lat_key: str = "latitude",
    lon_key: str = "longitude",
    kwp_key: str | None = "pvout_peak_kw",
    tilt_key: str | None = "panel_tilt",
) -> dict[str, Any]:
    """Override mapping in-place with scenario site/PV when present."""
    out = dict(values)
    lat, lon = site_location(cfg)
    if lat is not None:
        out[lat_key] = lat
    if lon is not None:
        out[lon_key] = lon
    kwp, tilt = pv_plant_params(cfg)
    if kwp_key and kwp is not None:
        out[kwp_key] = kwp
    if tilt_key and tilt is not None:
        out[tilt_key] = tilt
    return out
