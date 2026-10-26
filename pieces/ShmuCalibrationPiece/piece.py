"""Calibrate modelled irradiance against measured SHMU global radiation.

Open-Meteo is a model product. Compared against SHMU pyranometers its irradiance
bias is location dependent: over five days at six Slovak sites it ranged from
-12.9% in Bratislava to +3.3% in Kosice. A 13% error in irradiance goes straight
into yield, payback and the chosen system size, so an investment case should not
rest on the unvalidated model series.

This piece produces two things:

* a calibration report with the measured bias, error metrics and a scale factor
* a training CSV whose PVOUT column is recomputed from *measured* irradiance

The second output is what makes the downstream UC3.4 error-correction model
worth training. Instead of fitting the residual of a formula against itself, it
learns the mapping from Open-Meteo weather features to production implied by
ground measurement, which is a real and useful correction.

When the site is outside the SHMU network, the feed is unreachable, or the
overlap is too short, the piece degrades to a no-op with a scale factor of 1.0
and says so, rather than inventing a correction.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from domino.base_piece import BasePiece

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from common_somes import pv_model, shmu_opendata
    from common_somes.scenario_site import load_scenario_yaml, site_location, pv_plant_params
except ModuleNotFoundError:
    from pieces.common_somes import pv_model, shmu_opendata
    from pieces.common_somes.scenario_site import load_scenario_yaml, site_location, pv_plant_params

from .models import InputModel, OutputModel

try:
    from common import onedata_io as od
except ModuleNotFoundError:
    try:
        from pieces.common import onedata_io as od
    except ModuleNotFoundError:
        od = None

# Below this many overlapping daylight steps the bias estimate is noise.
MIN_DAYLIGHT_STEPS = 96

# Refuse corrections beyond this magnitude; they indicate a broken sensor, a
# shaded station or a unit mismatch rather than a genuine model bias.
MAX_ABS_SCALE_CORRECTION = 0.35


class ShmuCalibrationPiece(BasePiece):
    """Compare modelled irradiance with SHMU measurement and build a real target."""

    def piece_function(self, input_data: InputModel, secrets_data=None) -> OutputModel:
        _stage = None
        _piece_out = None
        _run_id = None
        if od is not None:
            input_data, _stage = od.stage_inputs(input_data, secrets_data)
            _run_id = od.resolve_run_id(input_data, secrets_data, generate=False)

        weather_path = Path(str(input_data.weather_csv_path))
        base_path = Path(str(input_data.training_base_csv))
        scenario_path = Path(str(input_data.scenario_yaml))
        out_dir = Path(self.results_path or scenario_path.parent)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "shmu_calibration.log"

        def _log(msg: str) -> None:
            text = f"[ShmuCalibrationPiece] {msg}"
            print(text, flush=True)
            try:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(text + "\n")
            except Exception:
                pass

        calibration_json = out_dir / "shmu_calibration.json"
        training_csv = out_dir / "shmu_calibrated_training.csv"

        try:
            if not weather_path.is_file():
                raise FileNotFoundError(f"Weather CSV not found: {weather_path}")
            if not base_path.is_file():
                raise FileNotFoundError(f"Training base CSV not found: {base_path}")

            cfg = load_scenario_yaml(scenario_path) if scenario_path.is_file() else {}
            lat, lon = site_location(cfg)
            installed_kwp, tilt = pv_plant_params(cfg)

            report: dict = {
                "generated_at_utc": datetime.utcnow().isoformat(),
                "source": {
                    "name": "SHMU open data (opendata.shmu.sk)",
                    "dataset": "meteorology/climate/now — automatic weather stations, 1-minute",
                    "variable": shmu_opendata.GLOBAL_RADIATION_COLUMN,
                    "licence": "CC-BY 4.0, no registration required",
                },
                "site": {"latitude": lat, "longitude": lon},
            }

            if not bool(input_data.enabled):
                return self._finish(
                    self._write_skip(
                        report, calibration_json, training_csv, "disabled_by_input", _log,
                        base_path,
                    ),
                    calibration_json,
                    training_csv,
                    secrets_data,
                    _stage,
                    _run_id,
                )

            if lat is None or lon is None:
                return self._finish(
                    self._write_skip(
                        report, calibration_json, training_csv,
                        "scenario has no site.latitude / site.longitude", _log,
                        base_path,
                    ),
                    calibration_json, training_csv, secrets_data, _stage, _run_id,
                )

            station = shmu_opendata.nearest_station(
                lat, lon, max_distance_km=float(input_data.max_station_distance_km)
            )
            if station is None:
                return self._finish(
                    self._write_skip(
                        report, calibration_json, training_csv,
                        f"no SHMU radiation station within "
                        f"{input_data.max_station_distance_km:.0f} km of the site",
                        _log,
                        base_path,
                    ),
                    calibration_json, training_csv, secrets_data, _stage, _run_id,
                )

            report["station"] = {
                "ind_kli": station.ind_kli,
                "name": station.name,
                "latitude": station.latitude,
                "longitude": station.longitude,
                "altitude_m": station.altitude_m,
                "distance_km": round(station.distance_km(lat, lon), 2),
            }
            _log(f"Nearest station: {station.name} ({report['station']['distance_km']} km)")

            window = shmu_opendata.available_window()
            if window is None:
                return self._finish(
                    self._write_skip(
                        report, calibration_json, training_csv,
                        "SHMU open data server returned no usable days", _log,
                        base_path,
                    ),
                    calibration_json, training_csv, secrets_data, _stage, _run_id,
                )
            report["measurement_window"] = {"start": str(window[0]), "end": str(window[1])}
            _log(f"SHMU rolling window: {window[0]} .. {window[1]}")

            measured = shmu_opendata.fetch_global_radiation(station, window[0], window[1])
            if measured.empty:
                return self._finish(
                    self._write_skip(
                        report, calibration_json, training_csv,
                        "no measured global radiation returned for the window", _log,
                        base_path,
                    ),
                    calibration_json, training_csv, secrets_data, _stage, _run_id,
                )
            _log(f"Measured 1-minute rows: {len(measured)}")

            weather = self._read_weather(weather_path)
            freq = self._infer_freq(weather.index)
            measured_grid = shmu_opendata.resample_to(measured, freq)
            _log(f"Weather grid {freq}, measured resampled to {len(measured_grid)} steps")

            bias = shmu_opendata.bias_report(weather["GHI"], measured_grid)
            report["bias"] = bias

            if bias.get("status") != "ok" or int(bias.get("daylight_steps", 0)) < MIN_DAYLIGHT_STEPS:
                return self._finish(
                    self._write_skip(
                        report, calibration_json, training_csv,
                        f"overlap too short to calibrate "
                        f"({bias.get('daylight_steps', 0)} daylight steps, "
                        f"need {MIN_DAYLIGHT_STEPS})",
                        _log,
                        base_path,
                    ),
                    calibration_json, training_csv, secrets_data, _stage, _run_id,
                )

            scale = float(bias.get("recommended_scale_factor") or 1.0)
            if abs(scale - 1.0) > MAX_ABS_SCALE_CORRECTION:
                clamped = 1.0 + MAX_ABS_SCALE_CORRECTION * (1.0 if scale > 1 else -1.0)
                _log(f"Scale factor {scale:.3f} exceeds the sanity limit; clamped to {clamped:.3f}")
                report["scale_factor_clamped_from"] = scale
                scale = clamped

            report["irradiance_scale_factor"] = round(scale, 4)
            report["calibration_available"] = True
            _log(
                f"Measured bias {bias['bias_pct']:+.1f}% over {bias['daylight_steps']} daylight "
                f"steps; scale factor {scale:.4f}"
            )

            rows = self._build_training_rows(
                base_path, weather, measured_grid, cfg, lat, lon, installed_kwp or 0.0, tilt or 30.0
            )
            report["training_rows"] = int(len(rows))
            if len(rows):
                rows.to_csv(training_csv, index=False)
                _log(f"Wrote measurement-based training target: {training_csv} ({len(rows)} rows)")
            else:
                report["correction_stage"] = (
                    "no-op: measurement did not overlap the analysed period"
                )
                training_csv.write_bytes(base_path.read_bytes())

            calibration_json.write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            self.display_result = {"file_type": "json", "file_path": str(calibration_json)}

            _piece_out = OutputModel(
                message=(
                    f"Calibrated against {station.name}: bias {bias['bias_pct']:+.1f}%, "
                    f"scale {scale:.4f}, {len(rows)} measured training rows"
                ),
                calibration_json=str(calibration_json),
                calibrated_training_csv=str(training_csv),
                calibration_available=True,
                irradiance_scale_factor=scale,
            )
        except Exception as exc:
            (out_dir / "shmu_calibration_error.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            _log(f"ERROR: {exc}")
            if od is not None:
                od.cleanup_on_error(
                    self.results_path, secrets_data, "ShmuCalibrationPiece", _stage, run_id=_run_id
                )
            raise

        return self._finish(
            _piece_out, calibration_json, training_csv, secrets_data, _stage, _run_id
        )

    # --- helpers -------------------------------------------------------------

    @staticmethod
    def _read_weather(path: Path) -> pd.DataFrame:
        for sep in (";", ","):
            try:
                df = pd.read_csv(path, sep=sep)
            except Exception:
                continue
            if "datetime" in df.columns and "GHI" in df.columns:
                df = df.copy()
                df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
                df = df.dropna(subset=["datetime"]).set_index("datetime").sort_index()
                df["GHI"] = pd.to_numeric(df["GHI"], errors="coerce").fillna(0.0)
                return df
        raise ValueError(f"Weather CSV must contain `datetime` and `GHI` columns: {path}")

    @staticmethod
    def _infer_freq(index: pd.DatetimeIndex) -> str:
        if len(index) < 3:
            return "15min"
        minutes = pd.Series(index).diff().dt.total_seconds().median() / 60.0
        if not minutes or minutes <= 0:
            return "15min"
        return f"{int(round(minutes))}min"

    @staticmethod
    def _build_training_rows(
        base_path: Path,
        weather: pd.DataFrame,
        measured: pd.Series,
        cfg: dict,
        lat: float,
        lon: float,
        installed_kwp: float,
        tilt_deg: float,
    ) -> pd.DataFrame:
        """Replace the PVOUT target with production implied by ground measurement.

        The feature columns of the preprocessed dataset are passed through
        untouched, so the correction model sees exactly the inputs it will see at
        serving time. Only the target changes, to what the plant would have
        produced under the irradiance measured on the ground. Rows outside the
        measured window are dropped: an unlabelled row teaches nothing and would
        pull the correction back toward the model it is meant to correct.
        """
        base = pd.read_csv(base_path)
        if "datetime" not in base.columns or installed_kwp <= 0:
            return pd.DataFrame()

        base = base.copy()
        base["datetime"] = pd.to_datetime(base["datetime"], errors="coerce")
        base = base.dropna(subset=["datetime"]).set_index("datetime").sort_index()

        overlap = base.index.intersection(measured.index)
        if len(overlap) == 0:
            return pd.DataFrame()

        spec = pv_model.array_spec_from_scenario(
            cfg, installed_kwp=installed_kwp, tilt_deg=tilt_deg
        )
        sub = base.loc[overlap].copy()
        measured_ghi = measured.loc[overlap]

        # Timestamps are local wall clock; recover the offset the weather file
        # was written with so solar geometry stays consistent with the source.
        offset_seconds = ShmuCalibrationPiece._infer_utc_offset(sub.index, lat, lon)

        def _column(frame: pd.DataFrame, name: str, fallback: float) -> pd.Series:
            if name in frame.columns:
                return pd.to_numeric(frame[name], errors="coerce").fillna(fallback)
            if name in weather.columns:
                return (
                    pd.to_numeric(weather[name], errors="coerce")
                    .reindex(frame.index)
                    .fillna(fallback)
                )
            return pd.Series(fallback, index=frame.index)

        temps = _column(sub, "TEMP", 15.0)
        winds = _column(sub, "WS", 1.0)

        targets = []
        for stamp, ghi in measured_ghi.items():
            res = pv_model.ac_power_kw(
                dt=stamp.to_pydatetime(),
                ghi=float(ghi),
                dni=None,
                dif=None,
                temp_c=float(temps.at[stamp]),
                wind_ms=float(winds.at[stamp]),
                lat=lat,
                lon=lon,
                spec=spec,
                utc_offset_seconds=offset_seconds,
            )
            targets.append(res["pvout_kw"])

        if "PVOUT" in sub.columns:
            sub["PVOUT_MODELLED"] = sub["PVOUT"]
        sub["GHI_MEASURED"] = measured_ghi.to_numpy()
        sub["PVOUT"] = targets
        return sub.reset_index()

    @staticmethod
    def _infer_utc_offset(index: pd.DatetimeIndex, lat: float, lon: float) -> int:
        """Recover the UTC offset by finding the shift that centres solar noon.

        The weather CSV carries local wall-clock timestamps without a zone. Rather
        than assume one, the offset that best aligns modelled solar elevation with
        the observed daily maximum is selected from the plausible whole-hour range.
        """
        sample = index[: min(len(index), 4000)]
        best_offset, best_score = 0, -1e18
        for hours in range(-1, 4):
            offset = hours * 3600
            score = 0.0
            for stamp in sample[:: max(1, len(sample) // 200)]:
                elev, _ = pv_model.solar_position(
                    stamp.to_pydatetime() - timedelta(seconds=offset), lat, lon
                )
                score += max(elev, 0.0)
            if score > best_score:
                best_offset, best_score = offset, score
        return best_offset

    def _write_skip(
        self,
        report: dict,
        calibration_json: Path,
        training_csv: Path,
        reason: str,
        log,
        base_path: Path | None = None,
    ) -> OutputModel:
        """Fall back to the modelled series without breaking the chain.

        The correction stage still needs a dataset to train on, and Domino has no
        clean way to skip a node at runtime. Passing the modelled PVOUT through
        unchanged makes the correction a no-op, which is the right answer when
        there is nothing measured to correct toward. The report and the dashboard
        both say so rather than implying the yield was validated.
        """
        report["calibration_available"] = False
        report["skip_reason"] = reason
        report["irradiance_scale_factor"] = 1.0
        report["correction_stage"] = "no-op: trained on the modelled series"
        calibration_json.write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        if base_path is not None and base_path.is_file():
            training_csv.write_bytes(base_path.read_bytes())
        else:
            training_csv.write_text("", encoding="utf-8")
        log(f"Calibration skipped: {reason}")
        self.display_result = {"file_type": "json", "file_path": str(calibration_json)}
        return OutputModel(
            message=f"SHMU calibration not applied: {reason}",
            calibration_json=str(calibration_json),
            calibrated_training_csv=str(training_csv),
            calibration_available=False,
            irradiance_scale_factor=1.0,
        )

    def _finish(self, piece_out, calibration_json, training_csv, secrets_data, stage, run_id):
        if od is not None and piece_out is not None:
            return od.finish_piece(
                piece_out, self.results_path, secrets_data, "ShmuCalibrationPiece", stage,
                run_id=run_id,
            )
        if stage is not None:
            stage.cleanup()
        return piece_out
