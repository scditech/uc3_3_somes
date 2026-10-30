"""Physical PV production model used to build the PVOUT baseline.

The UC3.4 chain learns a mapping from weather features to a PVOUT target. That
only carries information if PVOUT is not itself a closed-form function of one of
those features. The original Open-Meteo generator used

    PVOUT = kWp * GHI / 1000 * 0.75

which makes PVOUT an exact multiple of GHI, so a model trained on it recovers a
constant and learns nothing about the plant.

This module instead runs the standard irradiance-to-AC chain:

    solar position -> POA transposition (Hay-Davies) -> incidence-angle losses
    -> cell temperature (Faiman) -> DC power -> system losses -> inverter clipping

PVOUT then depends jointly on GHI, DNI, DIF, air temperature, wind speed, array
geometry and inverter sizing, which is what the downstream model is meant to
reason about.

Pure Python and math only, so the piece keeps its small dependency footprint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

STC_IRRADIANCE_W_M2 = 1000.0
STC_CELL_TEMP_C = 25.0
NOCT_IRRADIANCE_W_M2 = 800.0
NOCT_AMBIENT_C = 20.0

# Faiman convective coefficients for an open-rack / roof-mounted array.
FAIMAN_U1 = 6.84

# ASHRAE incidence-angle modifier coefficient.
IAM_B0 = 0.05


@dataclass
class ArraySpec:
    """Array geometry, module behaviour and the system loss chain."""

    installed_kwp: float = 5.2
    tilt_deg: float = 30.0
    azimuth_deg: float = 180.0  # 180 = due south on the northern hemisphere
    albedo: float = 0.2

    temp_coeff_pmax_pct_per_c: float = -0.34
    noct_c: float = 45.0

    # Loss chain follows the usual PVsyst breakdown so each line can be argued
    # with the EPC contractor instead of hiding behind a single performance ratio.
    soiling_loss_pct: float = 2.0
    shading_loss_pct: float = 1.0
    mismatch_loss_pct: float = 2.0
    dc_wiring_loss_pct: float = 1.5
    ac_wiring_loss_pct: float = 0.5
    lid_loss_pct: float = 1.5
    module_quality_loss_pct: float = 0.5
    inverter_efficiency_pct: float = 98.0
    availability_pct: float = 99.0

    # AC clipping limit = installed_kwp / dc_ac_ratio
    dc_ac_ratio: float = 1.2

    @property
    def ac_limit_kw(self) -> float:
        if self.installed_kwp <= 0:
            return 0.0
        ratio = self.dc_ac_ratio if self.dc_ac_ratio > 0 else 1.0
        return self.installed_kwp / ratio

    @property
    def static_loss_factor(self) -> float:
        """Product of the losses that vary with neither irradiance nor temperature."""
        out = 1.0
        for pct in (
            self.soiling_loss_pct,
            self.shading_loss_pct,
            self.mismatch_loss_pct,
            self.dc_wiring_loss_pct,
            self.ac_wiring_loss_pct,
            self.lid_loss_pct,
            self.module_quality_loss_pct,
        ):
            out *= max(0.0, 1.0 - float(pct) / 100.0)
        out *= max(0.0, min(1.0, self.inverter_efficiency_pct / 100.0))
        out *= max(0.0, min(1.0, self.availability_pct / 100.0))
        return out

    def describe(self) -> dict:
        return {
            "installed_kwp": self.installed_kwp,
            "tilt_deg": self.tilt_deg,
            "azimuth_deg": self.azimuth_deg,
            "albedo": self.albedo,
            "temp_coeff_pmax_pct_per_c": self.temp_coeff_pmax_pct_per_c,
            "noct_c": self.noct_c,
            "dc_ac_ratio": self.dc_ac_ratio,
            "ac_limit_kw": round(self.ac_limit_kw, 3),
            "static_loss_factor": round(self.static_loss_factor, 5),
        }


def solar_position(dt_utc: datetime, lat: float, lon: float) -> tuple[float, float]:
    """Return ``(elevation_deg, azimuth_deg)``; azimuth 0=N, 90=E, 180=S, 270=W.

    NOAA low-precision equations (Spencer declination and equation of time).
    ``dt_utc`` must be UTC. Callers holding local wall-clock timestamps have to
    subtract the UTC offset first, otherwise daylight saving shifts solar noon by
    an hour and the plane-of-array transposition is wrong all summer.
    """
    dt = dt_utc
    doy = dt.timetuple().tm_yday
    frac_hour = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
    gamma = 2.0 * math.pi / 365.0 * (doy - 1 + (frac_hour - 12.0) / 24.0)

    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.001480 * math.sin(3 * gamma)
    )
    eot_min = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )

    # True solar time from UTC: longitude offset plus the equation of time.
    solar_time = frac_hour + lon / 15.0 + eot_min / 60.0
    ha = math.radians(15.0 * (solar_time - 12.0))

    lat_r = math.radians(lat)
    sin_elev = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(decl) * math.cos(ha)
    sin_elev = max(-1.0, min(1.0, sin_elev))
    elevation = math.degrees(math.asin(sin_elev))

    # Azimuth clockwise from north: 0 at solar noon would mean north, so the
    # atan2 pair below is arranged to give 180 deg (south) at ha = 0.
    sin_az = -math.sin(ha) * math.cos(decl)
    cos_az = math.sin(decl) * math.cos(lat_r) - math.cos(decl) * math.sin(lat_r) * math.cos(ha)
    azimuth = math.degrees(math.atan2(sin_az, cos_az)) % 360.0

    return elevation, azimuth


def extraterrestrial_irradiance(dt: datetime) -> float:
    doy = dt.timetuple().tm_yday
    return 1367.0 * (1.0 + 0.033 * math.cos(2.0 * math.pi * doy / 365.0))


def erbs_diffuse_fraction(ghi: float, cos_zenith: float, dni_extra: float) -> float:
    """Erbs correlation, used only when the provider gives no diffuse component."""
    ghi_extra = dni_extra * max(cos_zenith, 0.0)
    if ghi_extra <= 1e-6 or ghi <= 0.0:
        return 1.0
    kt = min(max(ghi / ghi_extra, 0.0), 1.0)
    if kt <= 0.22:
        return 1.0 - 0.09 * kt
    if kt <= 0.80:
        return 0.9511 - 0.1604 * kt + 4.388 * kt**2 - 16.638 * kt**3 + 12.336 * kt**4
    return 0.165


def poa_irradiance(
    *,
    ghi: float,
    dni: float,
    dif: float,
    elevation_deg: float,
    azimuth_deg: float,
    dni_extra: float,
    tilt_deg: float,
    surface_azimuth_deg: float,
    albedo: float,
) -> dict[str, float]:
    """Hay-Davies transposition of horizontal irradiance onto the array plane."""
    if elevation_deg <= 0.0:
        return {"poa_global": 0.0, "beam": 0.0, "sky_diffuse": 0.0, "ground": 0.0, "cos_aoi": 0.0}

    tilt_r = math.radians(tilt_deg)
    surf_az_r = math.radians(surface_azimuth_deg)
    zen_r = math.radians(90.0 - elevation_deg)
    sun_az_r = math.radians(azimuth_deg)

    cos_aoi = math.cos(zen_r) * math.cos(tilt_r) + math.sin(zen_r) * math.sin(tilt_r) * math.cos(
        sun_az_r - surf_az_r
    )
    cos_aoi = max(0.0, min(1.0, cos_aoi))

    beam = dni * cos_aoi

    # Anisotropy index splits sky diffuse into circumsolar and isotropic parts.
    ai = min(max(dni / dni_extra, 0.0), 1.0) if dni_extra > 1e-6 else 0.0
    cos_zen = max(math.cos(zen_r), 0.01)
    rb = cos_aoi / cos_zen
    sky_diffuse = dif * (ai * rb + (1.0 - ai) * (1.0 + math.cos(tilt_r)) / 2.0)

    ground = ghi * albedo * (1.0 - math.cos(tilt_r)) / 2.0

    return {
        "poa_global": max(0.0, beam + sky_diffuse + ground),
        "beam": beam,
        "sky_diffuse": sky_diffuse,
        "ground": ground,
        "cos_aoi": cos_aoi,
    }


def cell_temperature(poa: float, ambient_c: float, wind_ms: float, noct_c: float) -> float:
    """Faiman module temperature, with u0 implied by the datasheet NOCT."""
    noct_rise = max(noct_c - NOCT_AMBIENT_C, 1.0)
    u0 = NOCT_IRRADIANCE_W_M2 / noct_rise - FAIMAN_U1 * 1.0
    u0 = max(5.0, min(60.0, u0))
    return ambient_c + poa / (u0 + FAIMAN_U1 * max(wind_ms, 0.0))


def incidence_angle_modifier(cos_aoi: float) -> float:
    if cos_aoi <= 0.0:
        return 0.0
    iam = 1.0 - IAM_B0 * (1.0 / max(cos_aoi, 0.05) - 1.0)
    return max(0.0, min(1.0, iam))


def ac_power_kw(
    *,
    dt: datetime,
    ghi: float,
    dni: float | None,
    dif: float | None,
    temp_c: float,
    wind_ms: float,
    lat: float,
    lon: float,
    spec: ArraySpec,
    utc_offset_seconds: int = 0,
) -> dict[str, float]:
    """Run the full chain for one timestamp and return the intermediate values.

    ``dt`` is the local wall-clock timestamp of the weather record and
    ``utc_offset_seconds`` the offset the provider reported for it, so daylight
    saving is handled without guessing.
    """
    dt_utc = dt - timedelta(seconds=int(utc_offset_seconds))
    elevation, azimuth = solar_position(dt_utc, lat, lon)
    dni_extra = extraterrestrial_irradiance(dt_utc)
    cos_zen = math.sin(math.radians(elevation)) if elevation > 0 else 0.0

    ghi = max(0.0, float(ghi or 0.0))
    if dif is None:
        dif = ghi * erbs_diffuse_fraction(ghi, cos_zen, dni_extra)
    dif = max(0.0, min(float(dif), ghi)) if ghi > 0 else 0.0
    if dni is None:
        dni = (ghi - dif) / max(cos_zen, 0.01) if cos_zen > 0 else 0.0
    dni = max(0.0, min(float(dni), 1200.0))

    poa = poa_irradiance(
        ghi=ghi,
        dni=dni,
        dif=dif,
        elevation_deg=elevation,
        azimuth_deg=azimuth,
        dni_extra=dni_extra,
        tilt_deg=spec.tilt_deg,
        surface_azimuth_deg=spec.azimuth_deg,
        albedo=spec.albedo,
    )

    poa_eff = poa["poa_global"] * incidence_angle_modifier(poa["cos_aoi"]) if poa["poa_global"] > 0 else 0.0

    if spec.installed_kwp <= 0 or poa_eff <= 0.0:
        return {
            "elevation_deg": elevation,
            "azimuth_deg": azimuth,
            "gti": poa["poa_global"],
            "poa_effective": poa_eff,
            "cell_temp_c": temp_c,
            "dc_kw": 0.0,
            "pvout_kw": 0.0,
            "clipped_kw": 0.0,
            "dni": dni,
            "dif": dif,
        }

    tcell = cell_temperature(poa_eff, temp_c, wind_ms, spec.noct_c)
    temp_factor = 1.0 + (spec.temp_coeff_pmax_pct_per_c / 100.0) * (tcell - STC_CELL_TEMP_C)
    temp_factor = max(0.5, min(1.2, temp_factor))

    # Efficiency roll-off in weak light, flat above ~200 W/m2.
    if poa_eff >= 200.0:
        low_light = 1.0
    else:
        low_light = max(0.75, 0.92 + 0.08 * math.log10(max(poa_eff, 1.0) / 200.0))

    dc_kw = spec.installed_kwp * (poa_eff / STC_IRRADIANCE_W_M2) * temp_factor * low_light
    dc_kw = max(0.0, dc_kw)

    ac_uncapped = dc_kw * spec.static_loss_factor
    limit = spec.ac_limit_kw if spec.ac_limit_kw > 0 else float("inf")
    pv_kw = min(ac_uncapped, limit)

    return {
        "elevation_deg": elevation,
        "azimuth_deg": azimuth,
        "gti": poa["poa_global"],
        "poa_effective": poa_eff,
        "cell_temp_c": tcell,
        "dc_kw": dc_kw,
        "pvout_kw": pv_kw,
        "clipped_kw": max(0.0, ac_uncapped - pv_kw),
        "dni": dni,
        "dif": dif,
    }


def array_spec_from_scenario(cfg: dict, *, installed_kwp: float, tilt_deg: float) -> ArraySpec:
    """Build an :class:`ArraySpec` from scenario.yaml, keeping documented defaults."""
    pv = cfg.get("pv") if isinstance(cfg.get("pv"), dict) else {}
    losses = pv.get("losses") if isinstance(pv.get("losses"), dict) else {}

    def _f(source: dict, key: str, default: float) -> float:
        try:
            val = source.get(key)
            return default if val is None or val == "" else float(val)
        except (TypeError, ValueError):
            return default

    spec = ArraySpec(installed_kwp=installed_kwp, tilt_deg=tilt_deg)
    spec.azimuth_deg = _f(pv, "azimuth_deg", spec.azimuth_deg)
    spec.albedo = _f(pv, "albedo", spec.albedo)
    spec.temp_coeff_pmax_pct_per_c = _f(pv, "temp_coeff_pmax_pct_per_c", spec.temp_coeff_pmax_pct_per_c)
    spec.noct_c = _f(pv, "noct_c", spec.noct_c)
    spec.dc_ac_ratio = _f(pv, "dc_ac_ratio", spec.dc_ac_ratio)

    spec.soiling_loss_pct = _f(losses, "soiling_loss_pct", spec.soiling_loss_pct)
    spec.shading_loss_pct = _f(losses, "shading_loss_pct", spec.shading_loss_pct)
    spec.mismatch_loss_pct = _f(losses, "mismatch_loss_pct", spec.mismatch_loss_pct)
    spec.dc_wiring_loss_pct = _f(losses, "dc_wiring_loss_pct", spec.dc_wiring_loss_pct)
    spec.ac_wiring_loss_pct = _f(losses, "ac_wiring_loss_pct", spec.ac_wiring_loss_pct)
    spec.lid_loss_pct = _f(losses, "lid_loss_pct", spec.lid_loss_pct)
    spec.module_quality_loss_pct = _f(losses, "module_quality_loss_pct", spec.module_quality_loss_pct)
    spec.inverter_efficiency_pct = _f(losses, "inverter_efficiency_pct", spec.inverter_efficiency_pct)
    spec.availability_pct = _f(losses, "availability_pct", spec.availability_pct)

    # equipment.constraints.installation.shading is the CFO-facing knob and only
    # applies when pv.losses.shading_loss_pct was not given explicitly.
    if losses.get("shading_loss_pct") is None:
        inst = ((cfg.get("equipment") or {}).get("constraints") or {}).get("installation") or {}
        shading = str(inst.get("shading", "")).strip().lower()
        spec.shading_loss_pct = {"none": 0.0, "low": 1.0, "medium": 4.0, "high": 9.0}.get(
            shading, spec.shading_loss_pct
        )

    return spec
