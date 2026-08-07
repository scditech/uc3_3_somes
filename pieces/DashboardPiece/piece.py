from __future__ import annotations

try:
    from common import onedata_io as od
except ModuleNotFoundError:
    try:
        from pieces.common import onedata_io as od
    except ModuleNotFoundError:
        od = None

import json
from datetime import datetime, timezone
from pathlib import Path
import traceback

import pandas as pd
try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece

try:
    from .models import METRIC_HELP, InputModel, OutputModel
except ImportError:
    from .models import METRIC_HELP, InputModel, OutputModel


def render_kpi_metric(column, label: str, value: str, help_key: str, *, widget_key: str) -> None:
    """Metrika s viditeľným tlačidlom ? (popover) — help= pri st.metric býva málo viditeľné."""
    import streamlit as st

    text = METRIC_HELP.get(help_key, "")
    column.metric(label, value, help=text or None)
    if text:
        with column.popover("?", help="Vysvetlenie ukazovateľa", key=f"kpi_pop_{widget_key}"):
            st.markdown(text)


class DashboardPiece(BasePiece):
    """Build SoMES operational dashboard (investment KPIs only as SEED appendix)."""

    def piece_function(self, input_data: InputModel, secrets_data=None) -> OutputModel:
        _stage = None
        _run_id = None
        if od is not None:
            input_data, _stage = od.stage_inputs(input_data, secrets_data)
            _run_id = od.resolve_run_id(
                input_data, secrets_data, generate=False, results_path=getattr(self, "results_path", None)
            )
            if hasattr(input_data, "run_id") and _run_id and not getattr(input_data, "run_id", ""):
                try:
                    input_data.run_id = _run_id
                except Exception:
                    pass
        rep_path = Path(input_data.report_json) if input_data.report_json else None
        kpi_path = Path(input_data.kpi_results_csv) if input_data.kpi_results_csv else None
        inv_path = Path(input_data.investment_evaluation_csv) if input_data.investment_evaluation_csv else None
        alerts_path = Path(input_data.anomaly_alerts_csv) if input_data.anomaly_alerts_csv else None
        drift_path = Path(input_data.drift_report_json) if input_data.drift_report_json else None
        dispatch_path = Path(input_data.next_day_dispatch_plan_csv) if input_data.next_day_dispatch_plan_csv else None
        out_dir = Path(
            self.results_path
            or (dispatch_path.parent if dispatch_path else (rep_path.parent if rep_path else Path(".")))
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "dashboard.log"

        def _log(msg: str) -> None:
            text = f"[DashboardPiece] {msg}"
            print(text, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

        _log(f"Input report_json={rep_path}")
        _log(f"Input kpi_results_csv={kpi_path}")
        _log(f"Input investment_evaluation_csv={inv_path}")
        _log(f"Input anomaly_alerts_csv={alerts_path}")
        _log(f"Input drift_report_json={drift_path}")
        _log(f"Input next_day_dispatch_plan_csv={dispatch_path}")

        # Prefer operational SoMES dashboard when dispatch plan is available.
        if dispatch_path and dispatch_path.is_file():
            import sys

            repo_root = Path(__file__).resolve().parents[2]
            if str(repo_root) not in sys.path:
                sys.path.insert(0, str(repo_root))
            from pieces.common_somes.architecture import build_ops_dashboard, write_json

            plan = pd.read_csv(dispatch_path, parse_dates=["datetime"])
            kpis = {}
            if input_data.operational_kpis_json and Path(input_data.operational_kpis_json).is_file():
                kpis = json.loads(Path(input_data.operational_kpis_json).read_text(encoding="utf-8"))
            validation = {}
            if input_data.technical_validation_json and Path(input_data.technical_validation_json).is_file():
                validation = json.loads(Path(input_data.technical_validation_json).read_text(encoding="utf-8"))
            alerts_block = {"summary": {"total": 0, "critical": 0, "warning": 0, "info": 0}, "latest": [], "drift": {}}
            if alerts_path and alerts_path.is_file():
                alerts_df = pd.read_csv(alerts_path)
                if not alerts_df.empty:
                    sev = alerts_df.get("severity", pd.Series([], dtype=str)).astype(str).str.lower()
                    alerts_block["summary"] = {
                        "total": int(len(alerts_df)),
                        "critical": int((sev == "critical").sum()),
                        "warning": int((sev == "warning").sum()),
                        "info": int((sev == "info").sum()),
                    }
            if drift_path and drift_path.is_file():
                alerts_block["drift"] = json.loads(drift_path.read_text(encoding="utf-8"))
            seed_appendix = {
                "note": "Investment CAPEX/NPV/payback belongs to UC3.2 SEED.",
                "delegate_to": "UC3.2_SEED",
            }
            if inv_path and inv_path.is_file():
                inv_df = pd.read_csv(inv_path)
                seed_appendix["investment_evaluation_preview"] = (inv_df.to_dict(orient="records") or [{}])[0]
            feedback = {}
            if input_data.forecast_vs_actual_json and Path(input_data.forecast_vs_actual_json).is_file():
                feedback = json.loads(Path(input_data.forecast_vs_actual_json).read_text(encoding="utf-8"))
            data_quality = {}
            if input_data.data_quality_report_json and Path(input_data.data_quality_report_json).is_file():
                dq = json.loads(Path(input_data.data_quality_report_json).read_text(encoding="utf-8"))
                data_quality = {
                    "overall_severity": dq.get("overall_severity"),
                    "validated": dq.get("validated"),
                    "datasets": list((dq.get("datasets") or {}).keys()),
                    "recommendations": (dq.get("correction_recommendations") or [])[:10],
                }
            payload = build_ops_dashboard(
                next_day_plan=plan,
                operational_kpis=kpis,
                technical_validation=validation,
                alerts_summary=alerts_block,
                forecast_vs_actual=feedback,
                seed_appendix=seed_appendix,
                data_quality=data_quality,
            )
            out_json = write_json(out_dir / "dashboard_data.json", payload)
            _log(f"Wrote operational dashboard {out_json}")
            _piece_out = OutputModel(dashboard_data_json=str(out_json))
            if od is not None:
                if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):
                    try:
                        _piece_out.run_id = _run_id
                    except Exception:
                        pass
                return od.finish_piece(
                    _piece_out, self.results_path, secrets_data, "DashboardPiece", _stage, run_id=_run_id
                )
            if _stage is not None:
                _stage.cleanup()
            return _piece_out

        if not rep_path or not rep_path.is_file():
            raise FileNotFoundError(f"Report JSON not found: {rep_path}")
        if not kpi_path or not kpi_path.is_file():
            raise FileNotFoundError(f"KPI CSV not found: {kpi_path}")
        if not inv_path or not inv_path.is_file():
            raise FileNotFoundError(f"Investment CSV not found: {inv_path}")

        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            kpi_df = pd.read_csv(kpi_path)
            inv_df = pd.read_csv(inv_path)

            exec_ = rep.get("executive_summary") or {}
            mrk = rep.get("mrk_and_rv") or {}
            unc = rep.get("uncertainty_assessment") or {}
            inv = (inv_df.to_dict(orient="records") or [{}])[0]
            art = rep.get("artifacts") or {}
            profile_path = Path(art.get("baseline_vs_optimized_profile_csv") or "")
            chart = {"title": "Priebeh spotreby energie: baseline vs FVE+batéria", "x": [], "series": []}
            if profile_path.is_file():
                prof = pd.read_csv(profile_path)
                chart = {
                    "title": "Priebeh spotreby energie: baseline vs FVE+batéria",
                    "x": prof["datetime"].astype(str).tolist(),
                    "series": [
                        {
                            "name": "Bez FVE a batérie",
                            "unit": "kWh/interval",
                            "values": pd.to_numeric(
                                prof["baseline_energy_kwh_interval"], errors="coerce"
                            ).fillna(0.0).round(4).tolist(),
                        },
                        {
                            "name": "S FVE a batériou",
                            "unit": "kWh/interval",
                            "values": pd.to_numeric(
                                prof["optimized_energy_kwh_interval"], errors="coerce"
                            ).fillna(0.0).round(4).tolist(),
                        },
                    ],
                }

            alerts_block = {
                "summary": {"total": 0, "critical": 0, "warning": 0, "info": 0},
                "latest": [],
                "drift": {},
            }
            if alerts_path and alerts_path.is_file():
                alerts_df = pd.read_csv(alerts_path)
                if not alerts_df.empty:
                    sev = alerts_df.get("severity", pd.Series([], dtype=str)).astype(str).str.lower()
                    alerts_block["summary"] = {
                        "total": int(len(alerts_df)),
                        "critical": int((sev == "critical").sum()),
                        "warning": int((sev == "warning").sum()),
                        "info": int((sev == "info").sum()),
                    }
                    cols = [c for c in ["datetime", "department_id", "severity", "reason", "actual_kw", "expected_kw", "delta_kw", "robust_z"] if c in alerts_df.columns]
                    latest = alerts_df.sort_values("datetime", ascending=False, na_position="last").head(15)
                    alerts_block["latest"] = latest[cols].to_dict(orient="records")
            if drift_path and drift_path.is_file():
                alerts_block["drift"] = json.loads(drift_path.read_text(encoding="utf-8"))

            payload = {
                "format": "cfo_finance_dashboard_v1",
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "decision_kpis": {
                    "operating_cost_baseline_eur": exec_.get("operating_cost_baseline_eur"),
                    "operating_cost_with_pv_battery_eur": exec_.get("operating_cost_pv_battery_eur"),
                    "operating_savings_period_eur": exec_.get("operating_savings_eur_period"),
                    "operating_savings_annual_estimate_eur": exec_.get("operating_savings_eur_per_year_estimate"),
                    "total_capex_eur": inv.get("total_capex_eur"),
                    "simple_payback_years": inv.get("simple_payback_years"),
                    "discounted_payback_years": inv.get("discounted_payback_years"),
                    "npv_operating_eur": inv.get("npv_operating_eur"),
                    "p50_annual_savings_eur": unc.get("p50_annual_savings_eur"),
                    "p90_annual_savings_eur": unc.get("p90_annual_savings_eur"),
                    "rv_downsizing_potential_kw": mrk.get("rv_downsizing_potential_kw"),
                    "rv_fixed_fee_savings_period_eur": mrk.get("estimated_fixed_rv_fee_savings_if_resized_eur_for_period"),
                    "trading_only_annual_margin_eur_estimate": (
                        (rep.get("trading_only_analysis") or {}).get("annual_margin_eur_estimate")
                    ),
                    "battery_annual_equivalent_cycles_est": (
                        (rep.get("battery_lifetime_assessment") or {}).get("annual_equivalent_cycles_est")
                    ),
                    "battery_estimated_life_years_effective": (
                        (rep.get("battery_lifetime_assessment") or {}).get("estimated_life_years_effective")
                    ),
                    "finance_annual_net_cashflow_after_finance_eur": (
                        (rep.get("finance_layer") or {}).get("annual_net_cashflow_after_finance_eur")
                    ),
                    "finance_npv_after_finance_eur": (
                        (rep.get("finance_layer") or {}).get("npv_after_finance_eur")
                    ),
                },
                "single_chart": chart,
                "battery_lifetime_assessment": (rep.get("battery_lifetime_assessment") or {}),
                "c_rate_sweep": (rep.get("c_rate_sweep") or []),
                "trading_only_analysis": (rep.get("trading_only_analysis") or {}),
                "finance_layer": (rep.get("finance_layer") or {}),
                "quality_flags": {
                    "report_schema_version": ((rep.get("meta") or {}).get("schema_version")),
                    "catalog_url_outage_detected": (((rep.get("equipment") or {}).get("catalog_sync_status") or {}).get("url_outage_detected")),
                    "historical_prices_in_csv": ((rep.get("input_quality") or {}).get("historical_prices_in_csv")),
                },
                "alerts_and_drift": alerts_block,
            }

            out_json = out_dir / "dashboard_data.json"
            out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            _log(f"Wrote dashboard JSON: {out_json}; kpi_rows={len(kpi_df)}")
            _piece_out = OutputModel(dashboard_data_json=str(out_json))
            if od is not None:
                if hasattr(_piece_out, 'run_id') and _run_id and not getattr(_piece_out, 'run_id', ''):
                    try:
                        _piece_out.run_id = _run_id
                    except Exception:
                        pass
                return od.finish_piece(
                    _piece_out, self.results_path, secrets_data, "DashboardPiece", _stage, run_id=_run_id
                )
            if _stage is not None:
                _stage.cleanup()
            return _piece_out
        except Exception as exc:
            (out_dir / "dashboard_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"ERROR during dashboard assembly: {exc}")
            raise


# --- sizing grid UI ---

"""Mriežka variantov FVE (kWp) × batéria (kWh) pre dashboard."""


import pandas as pd
import plotly.express as px
import streamlit as st



def _grid_to_frame(grid: list[dict]) -> pd.DataFrame:
    if not grid:
        return pd.DataFrame()
    df = pd.DataFrame(grid)
    if "fve_kwp" not in df.columns and "kwp" in df.columns:
        df["fve_kwp"] = df["kwp"]
    if "bateria_kwh" not in df.columns and "kwh" in df.columns:
        df["bateria_kwh"] = df["kwh"]
    if "payback_years" not in df.columns and "simple_payback_years" in df.columns:
        df["payback_years"] = df["simple_payback_years"]
    return df


def render_sizing_grid_section(
    grid: list[dict],
    *,
    recommended_kwp: float | None = None,
    recommended_kwh: float | None = None,
) -> None:
    df = _grid_to_frame(grid)
    if df.empty:
        st.info("Mriežka variantov zatiaľ nie je k dispozícii (spusti auto návrh vo workflow).")
        return

    st.subheader("Mriežka variantov (FVE × batéria)")
    st.caption(
        "Stĺpce: **FVE (kWp)** = fotovoltaika, **Batéria (kWh)** = kapacita úložiska. "
        "Každý riadok je jedna kombinácia veľkostí z auto optimalizácie."
    )

    n_kwp = df["fve_kwp"].nunique()
    n_kwh = df["bateria_kwh"].nunique()
    st.markdown(f"**Rozsah v mriežke:** {n_kwp} veľkostí FVE × {n_kwh} veľkostí batérie = **{len(df)}** variantov")

    labels = []
    for _, row in df.iterrows():
        kwp = float(row["fve_kwp"])
        kwh = float(row["bateria_kwh"])
        pb = row.get("payback_years")
        pb_s = f"{float(pb):.2f} r." if pb is not None and pd.notna(pb) else "—"
        labels.append(f"FVE {kwp:.0f} kWp × batéria {kwh:.0f} kWh (návratnosť {pb_s})")

    pick = st.selectbox(
        "Vybrať variant z mriežky",
        options=list(range(len(df))),
        format_func=lambda i: labels[int(i)],
        help=METRIC_HELP.get("grid_pick"),
    )
    sel = df.iloc[int(pick)]
    c1, c2, c3, c4 = st.columns(4)
    render_kpi_metric(c1, "FVE (kWp)", f"{float(sel['fve_kwp']):,.0f}", "grid_fve_kwp", widget_key="grid_kwp")
    render_kpi_metric(c2, "Batéria (kWh)", f"{float(sel['bateria_kwh']):,.0f}", "grid_battery_kwh", widget_key="grid_kwh")
    if pd.notna(sel.get("payback_years")):
        render_kpi_metric(c3, "Návratnosť (r.)", f"{float(sel['payback_years']):.2f}", "grid_payback", widget_key="grid_pb")
    if pd.notna(sel.get("npv_eur")):
        render_kpi_metric(c4, "NPV (€)", f"{float(sel['npv_eur']):,.0f}", "grid_npv", widget_key="grid_npv")

    if recommended_kwp is not None and recommended_kwh is not None:
        if abs(float(sel["fve_kwp"]) - float(recommended_kwp)) < 1e-6 and abs(
            float(sel["bateria_kwh"]) - float(recommended_kwh)
        ) < 1e-6:
            st.success("Tento variant zodpovedá **odporúčanému návrhu** z workflow.")
        else:
            st.caption(
                f"Odporúčaný návrh workflow: **FVE {recommended_kwp:.0f} kWp** + "
                f"**batéria {recommended_kwh:.0f} kWh**."
            )

    show_cols = [
        c
        for c in (
            "fve_kwp",
            "bateria_kwh",
            "payback_years",
            "annual_operating_savings_eur",
            "npv_eur",
            "total_capex_eur",
            "score",
        )
        if c in df.columns
    ]
    display = df[show_cols].copy()
    display = display.rename(
        columns={
            "fve_kwp": "FVE (kWp)",
            "bateria_kwh": "Batéria (kWh)",
            "payback_years": "Návratnosť (r.)",
            "annual_operating_savings_eur": "Ročná úspora (€)",
            "npv_eur": "NPV (€)",
            "total_capex_eur": "CAPEX (€)",
            "score": "Skóre",
        }
    )
    st.dataframe(display, use_container_width=True, height=min(420, 80 + 35 * len(display)))

    if n_kwp > 1 and n_kwh > 1 and "payback_years" in df.columns:
        st.markdown("**Mapa návratnosti (FVE × batéria)**")
        pivot = df.pivot_table(
            index="bateria_kwh",
            columns="fve_kwp",
            values="payback_years",
            aggfunc="mean",
        )
        pivot.index.name = "Batéria (kWh)"
        pivot.columns.name = "FVE (kWp)"
        st.dataframe(pivot.style.format("{:.2f}"), use_container_width=True)
        long = df[["fve_kwp", "bateria_kwh", "payback_years"]].dropna()
        if not long.empty:
            fig = px.imshow(
                pivot,
                labels=dict(x="FVE (kWp)", y="Batéria (kWh)", color="Návratnosť (r.)"),
                color_continuous_scale="RdYlGn_r",
                aspect="auto",
                title="Návratnosť podľa veľkosti FVE a batérie",
            )
            st.plotly_chart(fig, use_container_width=True)



# --- timeseries dashboard ---

"""
Time-series / investment section based on the original Domino DashboardPiece (charts, metrics, tables).
Invoked with `payload` from `dashboard_data.json` (timeseries branch).
"""


import pandas as pd
import plotly.express as px
import streamlit as st



def _records_to_df(records: object) -> pd.DataFrame:
    if not isinstance(records, list) or not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    return df


def _as_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _pick_existing(columns: list[str], options: list[str]) -> str | None:
    lower_map = {column.lower(): column for column in columns}
    for option in options:
        if option.lower() in lower_map:
            return lower_map[option.lower()]
    return None


def _filter_by_scenario(df: pd.DataFrame, selected_scenario: str) -> pd.DataFrame:
    if df.empty:
        return df
    for col in ["scenario", "scenario_name", "case", "variant"]:
        if col in df.columns:
            filtered = df[df[col].astype(str) == selected_scenario]
            if not filtered.empty:
                return filtered
    return df


def _first_row(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def _kpi_value(mapping: dict, keys: list[str], default: float = 0.0) -> float:
    for key in keys:
        if key in mapping:
            return _as_float(mapping[key], default)
    return default


def _format_eur(value: float) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    return f"{value:,.0f} €"


def _render_dataset_table(title: str, df: pd.DataFrame, missing_message: str) -> None:
    st.markdown(f"**{title}**")
    if df.empty:
        st.info(missing_message)
        return
    st.caption(f"Rows: {len(df)} | Columns: {len(df.columns)}")
    st.dataframe(df, use_container_width=True)


def _find_soc_series(*frames: pd.DataFrame) -> tuple[pd.DataFrame, str | None, str | None]:
    for frame in frames:
        if frame.empty:
            continue
        datetime_col = _pick_existing(frame.columns.tolist(), ["datetime", "timestamp", "time"])
        soc_col = _pick_existing(frame.columns.tolist(), ["soc_pct", "battery_soc", "state_of_charge", "soc"])
        if datetime_col and soc_col:
            return frame, datetime_col, soc_col
    return pd.DataFrame(), None, None


def _time_range_str(df: pd.DataFrame, dt_col: str) -> str:
    if df.empty or not dt_col or dt_col not in df.columns:
        return ""
    s = pd.to_datetime(df[dt_col], errors="coerce").dropna()
    if s.empty:
        return ""
    return f"{s.min().strftime('%d.%m.%Y')} – {s.max().strftime('%d.%m.%Y')}"


def _simulation_period_days(df: pd.DataFrame, dt_col: str) -> tuple[float | None, str]:
    if df.empty or not dt_col or dt_col not in df.columns:
        return None, ""
    s = pd.to_datetime(df[dt_col], errors="coerce").dropna()
    if s.empty or len(s) < 2:
        return None, ""
    t_min, t_max = s.min(), s.max()
    days = (t_max - t_min).total_seconds() / 86400.0
    range_str = f"{t_min.strftime('%d.%m.%Y')} – {t_max.strftime('%d.%m.%Y')}"
    return days, range_str


METRIC_LABELS = {
    "total_capex_eur": "Total investment (€)",
    "solar_capex_eur": "Solar CAPEX (€)",
    "battery_capex_eur": "Battery CAPEX (€)",
    "annual_savings_eur": "Annual savings (€)",
    "simple_payback_years": "Payback period (years)",
    "npv_eur": "Net present value (€)",
    "solar_lcoe_eur_per_mwh": "Levelized cost of energy (€/MWh)",
    "annual_co2_saved_ton": "CO₂ saved (t/year)",
    "battery_cycles_est": "Battery equivalent full cycles (over period)",
}


def render_timeseries_dashboard(payload: dict) -> None:
    """Renders the original financial/time-series dashboard from DashboardPiece JSON."""
    if not payload:
        st.warning("Timeseries simulation data is missing.")
        return

    datasets = payload.get("datasets", {})
    status = payload.get("inputs", {})
    alerts_and_drift = payload.get("alerts_and_drift") or {}

    if alerts_and_drift:
        st.subheader("Anomaly alerts & drift")
        summary = alerts_and_drift.get("summary") or {}
        c1, c2, c3, c4 = st.columns(4)
        render_kpi_metric(c1, "Upozornenia celkom", f"{int(summary.get('total', 0))}", "alerts_total", widget_key="alerts_total")
        render_kpi_metric(c2, "Kritické", f"{int(summary.get('critical', 0))}", "alerts_critical", widget_key="alerts_critical")
        render_kpi_metric(c3, "Varovania", f"{int(summary.get('warning', 0))}", "alerts_warning", widget_key="alerts_warning")
        render_kpi_metric(c4, "Info", f"{int(summary.get('info', 0))}", "alerts_info", widget_key="alerts_info")

        drift = alerts_and_drift.get("drift") or {}
        dep = drift.get("departments") or []
        if dep:
            st.markdown("**Drift state by department**")
            st.dataframe(pd.DataFrame(dep), use_container_width=True, hide_index=True)
        latest = alerts_and_drift.get("latest") or []
        if latest:
            st.markdown("**Latest alerts**")
            st.dataframe(pd.DataFrame(latest), use_container_width=True, hide_index=True)
        st.divider()

    if not datasets and payload.get("single_chart"):
        st.subheader("Finančné ukazovatele (KPI)")
        st.caption("Pod každou hodnotou je tlačidlo **?** — kliknutím zobrazíte vysvetlenie v slovenčine.")
        k = payload.get("decision_kpis") or {}
        c1, c2, c3, c4 = st.columns(4)
        render_kpi_metric(
            c1,
            "Úspora (obdobie)",
            _format_eur(_as_float(k.get("operating_savings_period_eur"), 0.0)),
            "savings_period",
            widget_key="fin_savings",
        )
        render_kpi_metric(
            c2,
            "CAPEX (investícia)",
            _format_eur(_as_float(k.get("total_capex_eur"), 0.0)),
            "capex",
            widget_key="fin_capex",
        )
        pb = k.get("simple_payback_years")
        render_kpi_metric(
            c3,
            "Návratnosť",
            f"{_as_float(pb):.2f} r." if pb is not None else "—",
            "payback",
            widget_key="fin_payback",
        )
        render_kpi_metric(
            c4,
            "NPV",
            _format_eur(_as_float(k.get("npv_operating_eur"), 0.0)),
            "npv",
            widget_key="fin_npv",
        )

        st.subheader(payload["single_chart"].get("title", "Consumption chart"))
        x = payload["single_chart"].get("x") or []
        series = payload["single_chart"].get("series") or []
        if x and series:
            chart_df = pd.DataFrame({"datetime": x})
            for s in series:
                name = s.get("name", "series")
                chart_df[name] = s.get("values") or []
            fig = px.line(chart_df, x="datetime", y=[c for c in chart_df.columns if c != "datetime"])
            st.plotly_chart(fig, use_container_width=True)
        return

    preprocess_df = _records_to_df(datasets.get("preprocess_predict", []))
    predict_df = _records_to_df(datasets.get("predict_predictions", []))
    simulate_df = _records_to_df(datasets.get("simulate_results", []))
    simulate_summary_df = _records_to_df(datasets.get("simulate_summary", []))
    kpi_df = _records_to_df(datasets.get("kpi_results", []))
    investment_df = _records_to_df(datasets.get("investment_evaluation", []))
    virtual_battery_soc_df = _records_to_df(datasets.get("virtual_battery_soc", []))

    scenario_options = payload.get("scenarios") or ["Default"]
    default_scenario = payload.get("default_scenario", scenario_options[0])
    selected_scenario = st.selectbox(
        "Scenár",
        scenario_options,
        index=scenario_options.index(default_scenario) if default_scenario in scenario_options else 0,
        help=METRIC_HELP.get("scenario_select"),
    )

    st.subheader("Solar PV & battery (scenario)")
    scenario_info = payload.get("scenario_info") or {}
    solar_kwp = scenario_info.get("solar_kwp")
    battery_kwh = scenario_info.get("battery_kwh")
    scenario_desc = (scenario_info.get("description") or "").strip()
    if solar_kwp is not None or battery_kwh is not None:
        cap1, cap2 = st.columns(2)
        render_kpi_metric(
            cap1,
            "Výkon FVE",
            f"{solar_kwp:,.0f} kWp" if solar_kwp is not None else "—",
            "solar_pv_capacity",
            widget_key="cap_kwp",
        )
        render_kpi_metric(
            cap2,
            "Kapacita batérie",
            f"{battery_kwh:,.0f} kWh" if battery_kwh is not None else "—",
            "battery_capacity",
            widget_key="cap_kwh",
        )
        if scenario_desc:
            st.caption(scenario_desc)
    else:
        st.info(
            "Solar PV and battery capacity are not in the report. "
            "Re-run the workflow so that **scenario.yml** is passed to DashboardPiece."
        )

    sim_period_days: float | None = None
    sim_period_str: str = ""
    for _df, _name in [(simulate_df, "datetime"), (virtual_battery_soc_df, "datetime"), (predict_df, "datetime")]:
        _col = _pick_existing(_df.columns.tolist(), ["datetime", "timestamp", "time"])
        if _col:
            sim_period_days, sim_period_str = _simulation_period_days(_df, _col)
            if sim_period_days is not None and sim_period_days > 0:
                break

    st.subheader("Executive summary")
    kpi_data = {}
    kpi_data.update(_first_row(simulate_summary_df))
    kpi_data.update(_first_row(kpi_df))
    kpi_data.update(_first_row(investment_df))

    total_capex = _kpi_value(kpi_data, ["total_capex_eur", "total_capex", "capex_eur"])
    payback = _kpi_value(kpi_data, ["simple_payback_years", "payback_years", "payback_period", "payback"])
    npv_val = _kpi_value(kpi_data, ["npv_eur", "npv", "net_present_value_eur"])
    saving = _kpi_value(
        kpi_data,
        ["annual_savings_eur", "annual_savings_€", "estimated_yearly_savings_eur", "savings_eur"],
    )

    col1, col2, col3, col4 = st.columns(4)
    render_kpi_metric(col1, "Investícia (CAPEX)", _format_eur(total_capex), "total_investment", widget_key="exec_capex")
    render_kpi_metric(
        col2,
        "Návratnosť",
        f"{payback:.1f} r." if payback and payback < 999 else "—",
        "payback_period",
        widget_key="exec_payback",
    )
    render_kpi_metric(col3, "NPV", _format_eur(npv_val), "npv_full", widget_key="exec_npv")
    render_kpi_metric(col4, "Ročná úspora", _format_eur(saving), "annual_savings", widget_key="exec_savings")

    if npv_val is not None and npv_val > 0:
        if payback and payback < 999:
            st.success(
                f"**Recommendation:** Positive NPV indicates the project is financially favourable. "
                f"The investment is expected to pay back in about **{payback:.1f}** years."
            )
        else:
            st.success("**Recommendation:** Positive NPV indicates the project is financially favourable under the current assumptions.")
    elif npv_val is not None and npv_val <= 0 and total_capex and total_capex > 0:
        st.info("NPV is not positive in this scenario. Consider reviewing assumptions (tariffs, CAPEX, discount rate) or timeline.")
    st.divider()

    st.subheader("Investment summary")
    invest_display = investment_df.copy()
    invest_display = _filter_by_scenario(invest_display, selected_scenario)
    if not invest_display.empty:
        exclude = {"datetime", "timestamp", "date"}
        numeric_cols = [c for c in invest_display.columns if c.lower() not in exclude]
        if numeric_cols:
            row = invest_display[numeric_cols].apply(pd.to_numeric, errors="coerce").iloc[0]
            summary_data = []
            for key, val in row.dropna().items():
                label = METRIC_LABELS.get(key, key.replace("_", " ").title())
                if "eur" in key.lower() or "€" in label:
                    summary_data.append({"Metric": label, "Value": _format_eur(val)})
                elif "year" in key.lower() or "payback" in key.lower():
                    summary_data.append({"Metric": label, "Value": f"{val:.1f} years"})
                else:
                    summary_data.append({"Metric": label, "Value": f"{val:,.2f}"})
            if summary_data:
                st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        if sim_period_str:
            st.caption(
                f"**Simulated period:** {sim_period_str}"
                + (f" ({sim_period_days:.0f} days)" if sim_period_days and sim_period_days > 0 else "")
            )
        battery_cycles_val = _kpi_value(_first_row(invest_display), ["battery_cycles_est", "cycles_equivalent"])
        if battery_cycles_val is not None and sim_period_days and sim_period_days > 0:
            cycles_per_year = battery_cycles_val * (365.0 / sim_period_days)
            st.caption(
                f"Battery equivalent full cycles above ({battery_cycles_val:.2f}) are for this period. "
                f"**Extrapolated to one year: {cycles_per_year:.1f} equivalent full cycles/year.**"
            )
        st.caption(
            "Positive NPV means the project is financially favourable over the analysis period. "
            "Payback is the number of years until cumulative savings cover the initial investment. "
            "Battery equivalent full cycles: total charge/discharge (SoC change) over the simulation period expressed as full 0↔100% cycles."
        )
    else:
        st.info("Investment evaluation data was not provided.")
    st.divider()

    load_df = _filter_by_scenario(simulate_df.copy(), selected_scenario)
    sim_summary_row = _first_row(_filter_by_scenario(simulate_summary_df.copy(), selected_scenario)) or _first_row(
        simulate_summary_df
    )

    st.caption(
        "**For the financial director:** Use **annual (extrapolated)** figures for planning and budgets. "
        "They are scaled from the simulated period when it is shorter than a year."
    )

    total_kwh_baseline = total_kwh_simulated = None
    if not load_df.empty:
        original_col = _pick_existing(load_df.columns.tolist(), ["baseline_load_kw", "original_load_kw", "load_kw", "original_load"])
        net_col = _pick_existing(load_df.columns.tolist(), ["simulated_load_kw", "net_load_kw", "net_load", "grid_import_kw"])
        if original_col and net_col:
            total_kwh_baseline = (load_df[original_col].astype(float) * 0.25).sum()
            total_kwh_simulated = (load_df[net_col].astype(float) * 0.25).sum()

    cost_baseline = _as_float(sim_summary_row.get("baseline_cost_eur")) if sim_summary_row else None
    cost_scenario = _as_float(sim_summary_row.get("scenario_cost_eur")) if sim_summary_row else None
    cost_savings = _as_float(sim_summary_row.get("savings_eur")) if sim_summary_row else None

    col_period, col_year = st.columns(2)
    with col_period:
        st.subheader("Over simulated period")
        if sim_period_str:
            st.caption(sim_period_str + (f" ({sim_period_days:.0f} days)" if sim_period_days and sim_period_days > 0 else ""))
        if total_kwh_baseline is not None and total_kwh_simulated is not None:
            render_kpi_metric(
                col_period,
                "Spotreba bez FVE a batérie",
                f"{total_kwh_baseline:,.0f} kWh",
                "consumption_baseline",
                widget_key="p_cons_base",
            )
            render_kpi_metric(
                col_period,
                "Spotreba so FVE a batériou",
                f"{total_kwh_simulated:,.0f} kWh",
                "consumption_with_pv_bess",
                widget_key="p_cons_opt",
            )
        if cost_baseline is not None and cost_scenario is not None:
            render_kpi_metric(col_period, "Náklady bez FVE a batérie", _format_eur(cost_baseline), "cost_baseline", widget_key="p_cost_base")
            render_kpi_metric(col_period, "Náklady so FVE a batériou", _format_eur(cost_scenario), "cost_with_pv_bess", widget_key="p_cost_opt")
            render_kpi_metric(col_period, "Úspora", _format_eur(cost_savings), "cost_savings", widget_key="p_cost_save")
        if total_kwh_baseline is None and cost_baseline is None:
            st.info("No consumption or cost data for this period.")

    with col_year:
        st.subheader("Extrapolated to 1 year")
        st.caption("Annual equivalent (for planning). Based on simulated period.")
        if sim_period_days and sim_period_days > 0:
            factor = 365.0 / sim_period_days
            if total_kwh_baseline is not None and total_kwh_simulated is not None:
                render_kpi_metric(
                    col_year,
                    "Spotreba bez FVE/batérie (rok)",
                    f"{total_kwh_baseline * factor:,.0f} kWh/rok",
                    "consumption_baseline_year",
                    widget_key="y_cons_base",
                )
                render_kpi_metric(
                    col_year,
                    "Spotreba s FVE/batériou (rok)",
                    f"{total_kwh_simulated * factor:,.0f} kWh/rok",
                    "consumption_with_year",
                    widget_key="y_cons_opt",
                )
            if cost_baseline is not None and cost_scenario is not None and cost_savings is not None:
                render_kpi_metric(
                    col_year,
                    "Náklady bez FVE/batérie (rok)",
                    _format_eur(cost_baseline * factor),
                    "cost_baseline_year",
                    widget_key="y_cost_base",
                )
                render_kpi_metric(
                    col_year,
                    "Náklady s FVE/batériou (rok)",
                    _format_eur(cost_scenario * factor),
                    "cost_with_year",
                    widget_key="y_cost_opt",
                )
                render_kpi_metric(
                    col_year,
                    "Úspora (rok)",
                    _format_eur(cost_savings * factor),
                    "savings_year",
                    widget_key="y_cost_save",
                )
        else:
            st.info("Cannot extrapolate: simulated period unknown or zero.")
            if total_kwh_baseline is not None and total_kwh_simulated is not None:
                render_kpi_metric(
                    col_year,
                    "Spotreba bez FVE a batérie",
                    f"{total_kwh_baseline:,.0f} kWh",
                    "consumption_baseline",
                    widget_key="y2_cons_base",
                )
                render_kpi_metric(
                    col_year,
                    "Spotreba so FVE a batériou",
                    f"{total_kwh_simulated:,.0f} kWh",
                    "consumption_with_pv_bess",
                    widget_key="y2_cons_opt",
                )
            if cost_baseline is not None and cost_scenario is not None:
                render_kpi_metric(col_year, "Náklady bez FVE a batérie", _format_eur(cost_baseline), "cost_baseline", widget_key="y2_cost_base")
                render_kpi_metric(col_year, "Náklady so FVE a batériou", _format_eur(cost_scenario), "cost_with_pv_bess", widget_key="y2_cost_opt")
                render_kpi_metric(col_year, "Úspora", _format_eur(cost_savings), "cost_savings", widget_key="y2_cost_save")

    if not sim_summary_row or "baseline_cost_eur" not in (sim_summary_row or {}):
        st.info("Cost summary (baseline / scenario) is not available – SimulatePiece output (summary.csv) required.")
    st.divider()

    st.subheader("Predicted consumption over time (without vs with solar PV & battery)")
    if load_df.empty:
        st.info("Simulated load data was not provided.")
    else:
        datetime_col = _pick_existing(load_df.columns.tolist(), ["datetime", "timestamp", "time"])
        period_str = _time_range_str(load_df, datetime_col or "")
        if period_str:
            st.caption(f"Period: {period_str}")
        original_col = _pick_existing(load_df.columns.tolist(), ["baseline_load_kw", "original_load_kw", "load_kw", "original_load"])
        net_col = _pick_existing(load_df.columns.tolist(), ["simulated_load_kw", "net_load_kw", "net_load", "grid_import_kw"])
        if datetime_col and original_col and net_col:
            chart_df = load_df[[datetime_col, original_col, net_col]].copy()
            chart_df = chart_df.sort_values(by=datetime_col)
            fig_load = px.line(
                chart_df,
                x=datetime_col,
                y=[original_col, net_col],
                title="Predicted load (kW): without vs with solar PV & battery",
            )
            fig_load.update_layout(legend_title_text="")
            st.plotly_chart(fig_load, use_container_width=True)
        else:
            st.info("Required columns for load curve are missing.")

    if not load_df.empty:
        cost_baseline_col = _pick_existing(load_df.columns.tolist(), ["baseline_cost_eur"])
        cost_scenario_col = _pick_existing(load_df.columns.tolist(), ["scenario_cost_eur"])
        dt_col_cost = _pick_existing(load_df.columns.tolist(), ["datetime", "timestamp", "time"])
        if dt_col_cost and cost_baseline_col and cost_scenario_col:
            cost_chart = load_df[[dt_col_cost, cost_baseline_col, cost_scenario_col]].copy()
            cost_chart = cost_chart.sort_values(by=dt_col_cost)
            cost_chart["Cost without solar PV & battery (€)"] = cost_chart[cost_baseline_col].astype(float)
            cost_chart["Cost with solar PV & battery (€)"] = cost_chart[cost_scenario_col].astype(float)
            fig_cost = px.line(
                cost_chart,
                x=dt_col_cost,
                y=["Cost without solar PV & battery (€)", "Cost with solar PV & battery (€)"],
                title="Predicted electricity cost over time (€ per interval)",
            )
            st.plotly_chart(fig_cost, use_container_width=True)

    st.subheader("Forecast vs actual load")
    prediction_df = _filter_by_scenario(predict_df.copy(), selected_scenario)
    if prediction_df.empty:
        st.info("Forecast data was not provided.")
    else:
        dt_col = _pick_existing(prediction_df.columns.tolist(), ["datetime", "timestamp", "time"])
        if _time_range_str(prediction_df, dt_col or ""):
            st.caption(f"Period: {_time_range_str(prediction_df, dt_col or '')}")
        actual_col = _pick_existing(prediction_df.columns.tolist(), ["load_kw", "actual_load_kw", "load"])
        pred_col = _pick_existing(prediction_df.columns.tolist(), ["prediction_load_kw", "prediction_load_mw", "predicted_load_kw"])
        if dt_col and actual_col and pred_col:
            fig_pred = px.line(
                prediction_df.sort_values(by=dt_col),
                x=dt_col,
                y=[actual_col, pred_col],
                title="Actual load vs forecast",
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        else:
            st.info("Required columns for forecast chart are missing.")

    st.subheader("Battery state of charge (SoC)")
    soc_source_df, datetime_col, soc_col = _find_soc_series(
        _filter_by_scenario(virtual_battery_soc_df.copy(), selected_scenario),
        _filter_by_scenario(preprocess_df.copy(), selected_scenario),
        _filter_by_scenario(predict_df.copy(), selected_scenario),
        _filter_by_scenario(simulate_df.copy(), selected_scenario),
    )
    if soc_source_df.empty:
        st.info(
            "Battery SoC data was not provided. "
            "Ensure **virtual_battery_soc.csv** (BatterySimPiece output) is in the workflow."
        )
    else:
        if datetime_col:
            st.caption(f"Period: {_time_range_str(soc_source_df, datetime_col)}")
        soc_df = soc_source_df[[datetime_col, soc_col]].copy().sort_values(by=datetime_col)
        fig_soc = px.line(soc_df, x=datetime_col, y=soc_col, title="Battery SoC (%)")
        st.plotly_chart(fig_soc, use_container_width=True)

    st.subheader("Investment metrics (chart)")
    invest_df = _filter_by_scenario(investment_df.copy(), selected_scenario)
    if invest_df.empty:
        st.info("Investment evaluation data was not provided.")
    else:
        exclude = {"datetime", "timestamp", "date"}
        numeric_cols = [c for c in invest_df.columns if c.lower() not in exclude]
        if not numeric_cols:
            st.info("No numeric columns in investment_evaluation data – cannot draw chart.")
        else:
            row = invest_df[numeric_cols].apply(pd.to_numeric, errors="coerce").iloc[0]
            metrics_df = row.dropna().reset_index()
            metrics_df.columns = ["metric", "value"]
            metrics_df["label"] = metrics_df["metric"].map(lambda x: METRIC_LABELS.get(x, x.replace("_", " ").title()))
            if metrics_df.empty:
                st.info("No numeric values in investment_evaluation data – cannot draw chart.")
            else:
                fig_inv = px.bar(metrics_df, x="label", y="value", title="Investment evaluation")
                st.plotly_chart(fig_inv, use_container_width=True)

    with st.expander("Technical data – source files and status"):
        st.subheader("Source file data")
        _render_dataset_table(
            "PreprocessEnergyDataPiece: train_dataset.parquet",
            preprocess_df,
            "File not provided or empty.",
        )
        _render_dataset_table("PredictPiece: predictions_15min.csv", predict_df, "File not provided or empty.")
        _render_dataset_table("SimulatePiece: simulated_results.csv", simulate_df, "File not provided or empty.")
        _render_dataset_table("SimulatePiece: summary.csv", simulate_summary_df, "File not provided or empty.")
        _render_dataset_table("BatterySimPiece: virtual_battery_soc.csv", virtual_battery_soc_df, "File not provided or empty.")
        _render_dataset_table("KPIPiece: kpi_results.csv", kpi_df, "File not provided or empty.")
        _render_dataset_table("InvestmentEvalPiece: investment_evaluation.csv", investment_df, "File not provided or empty.")
        st.subheader("Input files status")
        status_rows = []
        for input_name, details in status.items():
            status_rows.append({
                "input": input_name,
                "provided": details.get("provided", False),
                "rows": details.get("rows", 0),
                "error": details.get("error"),
            })
        if status_rows:
            st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No input status metadata available.")



# --- unified dashboard view ---

"""Zdieľané vykreslenie unified dashboardu (bez spúšťania Streamlit page config)."""


import json
import sys
from pathlib import Path

import streamlit as st

_PKG = Path(__file__).resolve().parent
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
try:
    from .models import METRIC_HELP
    from .piece import render_kpi_metric
    from .sizing_grid_ui import render_sizing_grid_section
    from .timeseries_dashboard import render_timeseries_dashboard
except ImportError:
    from .models import METRIC_HELP
    # render_kpi_metric defined above
    # render_sizing_grid_section defined above
    # render_timeseries_dashboard defined above

ROOT = Path(__file__).resolve().parent.parent.parent
UNIFIED = ROOT / "tests" / "dashboard_data.json"
INV_FALLBACK = ROOT / "tests" / "FeasibilityReportPiece_Outputs" / "dashboard_data.json"
TS_FALLBACK = ROOT / "tests" / "DashboardPiece_Outputs" / "dashboard_data.json"


def load_unified_payload() -> dict | None:
    for p in (UNIFIED, INV_FALLBACK):
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("format") == "alternate_unified_v1":
                return data
            if "feasibility" in data and "investment" not in data:
                return {"investment": data, "timeseries": None}
            return data
    fb = ROOT / "tests" / "FeasibilityReportPiece_Outputs" / "feasibility_report.json"
    if fb.is_file():
        return {
            "investment": {
                "feasibility": json.loads(fb.read_text(encoding="utf-8")),
                "meta": {},
                "sizing_grid": [],
            },
            "timeseries": None,
        }
    return None


def get_timeseries_payload(raw: dict, *, allow_fallback: bool) -> dict | None:
    ts = raw.get("timeseries")
    if isinstance(ts, dict) and (
        ts.get("datasets") is not None
        or ts.get("meta")
        or ts.get("alerts_and_drift") is not None
        or ts.get("single_chart") is not None
        or ts.get("decision_kpis") is not None
    ):
        return ts
    if allow_fallback and TS_FALLBACK.is_file():
        return json.loads(TS_FALLBACK.read_text(encoding="utf-8"))
    return None


def render_investment(payload: dict, *, sizing_grid: list | None = None) -> None:
    meta = payload.get("meta") or {}
    feas = payload.get("feasibility") or {}
    cfo = feas.get("cfo_notes") or payload.get("cfo_notes") or {}
    grid = payload.get("sizing_grid") or []
    basis = feas.get("model_basis") or meta.get("model_basis") or "parametric_grid_only"

    if meta.get("site_name"):
        st.caption(f"Prevádzka: **{meta['site_name']}**")
    if meta.get("generated_at_utc"):
        st.caption(f"Vygenerované: {meta['generated_at_utc']}")

    if basis == "timeseries_simulation":
        st.success("Hlavné metriky z **časovej simulácie** (KPI → InvestmentEval).")
    else:
        st.warning("Len parametrická mriežka — spusti plný workflow pre simuláciu.")

    feasible = feas.get("feasible", False)
    target_pb = feas.get("target_payback_years")
    achieved = feas.get("achieved_payback_years")
    min_pb = feas.get("minimum_payback_in_search_space_years")

    st.subheader("Súhrn rozhodnutia")
    c1, c2, c3, c4 = st.columns(4)
    render_kpi_metric(c1, "Cieľová návratnosť (r.)", f"{target_pb:.1f}" if target_pb is not None else "—", "target_payback", widget_key="inv_tgt_pb")
    render_kpi_metric(c2, "Dosiahnutá návratnosť (r.)", f"{achieved:.2f}" if achieved is not None else "—", "achieved_payback", widget_key="inv_ach_pb")
    render_kpi_metric(c3, "Odporúčaná FVE", f"{feas.get('recommended_kwp', 0):,.0f} kWp", "recommended_kwp", widget_key="inv_kwp")
    render_kpi_metric(c4, "Odporúčaná batéria", f"{feas.get('recommended_kwh', 0):,.0f} kWh", "recommended_kwh", widget_key="inv_kwh")

    if feasible:
        st.success("Cieľová návratnosť je **splniteľná**.")
    else:
        st.error("Cieľová návratnosť **nie je splniteľná** pri týchto vstupoch.")
        if min_pb is not None:
            st.info(f"Minimum v mriežke: **{min_pb:.2f} r.**")

    st.subheader("Ekonomika")
    e1, e2, e3 = st.columns(3)
    render_kpi_metric(e1, "CAPEX (FVE + BESS)", f"{feas.get('capex_eur', 0):,.0f} €", "capex_fve_bess", widget_key="inv_capex")
    render_kpi_metric(e2, "Ročná úspora", f"{feas.get('annual_savings_eur', 0):,.0f} €", "annual_savings_inv", widget_key="inv_sav")
    npv = cfo.get("npv_eur_at_best")
    render_kpi_metric(e3, "NPV", f"{npv:,.0f} €" if npv is not None else "—", "npv_inv", widget_key="inv_npv")

    grid_use = sizing_grid if sizing_grid is not None else grid
    render_sizing_grid_section(
        grid_use,
        recommended_kwp=float(feas.get("recommended_kwp", 0) or 0) or None,
        recommended_kwh=float(feas.get("recommended_kwh", 0) or 0) or None,
    )


def render_unified_dashboard() -> bool:
    """Vykreslí unified dashboard. Vráti True ak boli dáta, inak False."""
    raw = load_unified_payload()
    if not raw:
        st.warning("Chýba `tests/dashboard_data.json`. Najprv dokonči workflow.")
        return False

    gen = raw.get("generated_at_utc")
    if gen:
        st.caption(f"Posledná aktualizácia (UTC): **{gen}**")
    st.caption("Pod každou metrikou je tlačidlo **?** — kliknutím zobrazíte slovenské vysvetlenie.")
    with st.expander("Slovník ukazovateľov", expanded=False):
        _glossary = (
            ("Úspora (obdobie)", "savings_period"),
            ("CAPEX", "capex"),
            ("Návratnosť", "payback"),
            ("NPV", "npv"),
            ("Ročná úspora", "annual_savings"),
        )
        for title, key in _glossary:
            st.markdown(f"**{title}**")
            st.caption(METRIC_HELP.get(key, ""))

    if raw.get("format") == "alternate_unified_v1":
        inv = raw.get("investment")
        ts_payload = get_timeseries_payload(raw, allow_fallback=True)
        if inv:
            tab_i, tab_t = st.tabs(["Investičný návrh", "Časová simulácia"])
            with tab_i:
                render_investment(inv, sizing_grid=inv.get("sizing_grid") if isinstance(inv, dict) else None)
            with tab_t:
                if ts_payload:
                    render_timeseries_dashboard(ts_payload)
                else:
                    st.info("Časová simulácia nie je k dispozícii.")
        elif ts_payload:
            render_timeseries_dashboard(ts_payload)
        else:
            st.error("Prázdny dashboard payload.")
    else:
        ts_payload = get_timeseries_payload(raw, allow_fallback=True)
        if ts_payload and not raw.get("feasibility"):
            render_timeseries_dashboard(ts_payload)
        else:
            render_investment(raw)
    return True

