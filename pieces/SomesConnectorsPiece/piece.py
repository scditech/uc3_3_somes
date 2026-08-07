"""SoMES Data Connectors — ingest weather, PV, BESS, grid, flexible loads."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd

try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece

from .models import InputModel, OutputModel


class SomesConnectorsPiece(BasePiece):
    """Module 1 connectors for SoMES-specific operational data categories.

    Resolution order per dataset: explicit file, CSV endpoint, live provider API,
    cached copy, synthetic fixture. The provenance of every dataset is recorded in
    the manifest and every dataset is profiled by the quality checks.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from pieces.common_somes.connectors import (
            CONNECTOR_FILES,
            demo_generated_keys,
            fetch_prices_okte,
            fetch_weather_open_meteo,
            generate_demo_connectors,
            resolve_connector,
        )
        from pieces.common_somes.quality import (
            build_quality_bundle,
            missing_data_frame,
            quality_indicator_frame,
        )

        out_dir = Path(self.results_path or input_data.connectors_dir or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        src = Path(input_data.connectors_dir) if input_data.connectors_dir else out_dir
        src.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "somes_connectors.log"

        def _log(msg: str) -> None:
            text = f"[SomesConnectorsPiece] {msg}"
            print(text, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

        try:
            weather_fetch = None
            if input_data.weather_api_enabled:
                def weather_fetch():  # noqa: E306 - local closure keeps provider args together
                    return fetch_weather_open_meteo(
                        latitude=float(input_data.latitude),
                        longitude=float(input_data.longitude),
                        forecast_days=int(input_data.forecast_days),
                    )

            prices_fetch = None
            if input_data.prices_api_enabled:
                def prices_fetch():  # noqa: E306
                    return fetch_prices_okte(lookback_days=int(input_data.prices_lookback_days))

            provenance: dict[str, dict[str, str]] = {}
            live_used = 0
            synthetic = demo_generated_keys(src)
            for key, url, fetcher in (
                ("weather_forecast", input_data.weather_url, weather_fetch),
                ("prices_history", input_data.prices_url, prices_fetch),
                ("pv_production", input_data.pv_production_url, None),
                ("bess_telemetry", input_data.bess_telemetry_url, None),
            ):
                existing = src / CONNECTOR_FILES[key]
                info = resolve_connector(
                    key,
                    out_dir=src,
                    url=url,
                    local_path=existing if (existing.is_file() and key not in synthetic) else None,
                    live_fetch=fetcher,
                    allow_demo=bool(input_data.allow_demo_fallback),
                )
                provenance[key] = {"source": info["source"], "detail": info["detail"]}
                if info["source"] in ("url", "live_api"):
                    live_used += 1
                if info["attempts"]:
                    for attempt in info["attempts"]:
                        _log(f"{key}: {attempt}")
                _log(f"{key} resolved from {info['source']} ({info['detail']})")

            files = generate_demo_connectors(src, overwrite_existing=False)
            for key in provenance:
                provenance[key].setdefault("source", "demo")

            # Publish resolved connector files into the piece results directory.
            for key, name in CONNECTOR_FILES.items():
                src_path = src / name
                if not src_path.is_file():
                    continue
                target = out_dir / name
                if src_path.resolve() != target.resolve():
                    target.write_bytes(src_path.read_bytes())
                files[key] = target

            prices_out = out_dir / CONNECTOR_FILES["prices_history"]
            if prices_out.is_file() and "prices_history" not in provenance:
                provenance["prices_history"] = {"source": "file", "detail": str(prices_out)}

            datasets: dict[str, pd.DataFrame] = {}
            for key in ("weather_forecast", "pv_production", "bess_telemetry", "prices_history"):
                path = out_dir / CONNECTOR_FILES[key]
                if path.is_file():
                    datasets[key] = pd.read_csv(path)
            bundle = build_quality_bundle(datasets)
            bundle["provenance"] = provenance
            quality_path = out_dir / "connectors_data_quality.json"
            quality_path.write_text(json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")
            indicators_path = out_dir / "data_quality_indicators.csv"
            quality_indicator_frame(bundle).to_csv(indicators_path, index=False)
            missing_path = out_dir / "missing_data_report.csv"
            missing_data_frame(bundle).to_csv(missing_path, index=False)

            manifest = {
                "format": "somes_connectors_manifest_v2",
                "workflow_type": "SoMES",
                "files": {k: str((out_dir / Path(v).name).as_posix()) for k, v in files.items()},
                "provenance": provenance,
                "live_sources_used": live_used,
                "data_quality_severity": bundle["overall_severity"],
            }
            manifest_path = out_dir / "connectors_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            _log(
                f"Ready connectors: {list(files)}; live sources={live_used}; "
                f"quality={bundle['overall_severity']}"
            )
            return OutputModel(
                message=f"SoMES connectors ready (live={live_used}, quality={bundle['overall_severity']})",
                connectors_manifest_json=str(manifest_path),
                weather_forecast_csv=str(out_dir / "weather_forecast.csv"),
                pv_production_csv=str(out_dir / "pv_production_measurements.csv"),
                bess_telemetry_csv=str(out_dir / "bess_telemetry.csv"),
                prices_csv=str(prices_out) if prices_out.is_file() else "",
                grid_constraints_json=str(out_dir / "grid_constraints.json"),
                flexible_loads_json=str(out_dir / "flexible_loads.json"),
                data_quality_report_json=str(quality_path),
                data_quality_indicators_csv=str(indicators_path),
                missing_data_report_csv=str(missing_path),
                live_sources_used=live_used,
            )
        except Exception as exc:
            (out_dir / "somes_connectors_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"ERROR: {exc}")
            raise
