"""Render SoMES operational dashboard payload to a self-contained HTML file."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


REASON_SK = {
    "spike_up": "skok nahor",
    "drop_down": "prepad nadol",
    "spike": "skok",
    "drop": "prepad",
}


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, *, digits: int = 1, suffix: str = "") -> str:
    number = _num(value)
    if number is None:
        return "—" if value is None else str(value)
    sign = "−" if number < 0 else ""
    abs_n = abs(number)
    raw = f"{abs_n:.{digits}f}"
    whole, _, frac = raw.partition(".")
    groups: list[str] = []
    while whole:
        groups.append(whole[-3:])
        whole = whole[:-3]
    grouped = " ".join(reversed(groups))
    body = f"{grouped},{frac}" if digits else grouped
    return f"{sign}{body}{suffix}"


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def _badge(ok: bool | None, *, yes: str = "áno", no: str = "nie") -> str:
    if ok is True:
        return f'<span class="badge ok">{_esc(yes)}</span>'
    if ok is False:
        return f'<span class="badge bad">{_esc(no)}</span>'
    return '<span class="badge">—</span>'


def _severity_badge(severity: Any) -> str:
    sev = str(severity or "").strip().lower()
    klass = {"critical": "bad", "warning": "warn", "info": "ok"}.get(sev, "")
    label = {"critical": "kritické", "warning": "varovanie", "info": "info"}.get(sev, sev or "—")
    return f'<span class="badge {klass}">{_esc(label)}</span>'


def _series(chart: dict[str, Any], *needles: str) -> list[float]:
    for item in chart.get("series") or []:
        name = str(item.get("name") or "").lower()
        if any(needle in name for needle in needles):
            return [_num(v) or 0.0 for v in (item.get("values") or [])]
    return []


def _argmax(values: list[float]) -> int | None:
    if not values:
        return None
    return max(range(len(values)), key=lambda i: values[i])


def _hhmm(label: Any) -> str:
    text = str(label or "")
    if " " in text:
        text = text.split(" ", 1)[1]
    return text[:5] if len(text) >= 5 else text


def _tip(text: str) -> str:
    return f'<span class="tip" tabindex="0">?<span class="tip-text">{_esc(text)}</span></span>'


def _card(label: str, value: str, *, note: str = "", klass: str = "", tip: str = "") -> str:
    extra = f' {klass}' if klass else ""
    note_html = f'<div class="note">{_esc(note)}</div>' if note else ""
    tip_html = _tip(tip) if tip else ""
    return (
        f'<div class="card"><div class="label"><span>{_esc(label)}</span>{tip_html}</div>'
        f'<div class="value{extra}">{value}</div>{note_html}</div>'
    )


def _alert_rows(alerts: list[dict[str, Any]]) -> str:
    if not alerts:
        return '<p class="note-block">Žiadne anomálie v tomto behu.</p>'
    rows: list[str] = []
    for item in alerts:
        reason = str(item.get("reason") or "")
        reason_sk = REASON_SK.get(reason, reason or "—")
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('datetime') or '—')}</td>"
            f"<td>{_severity_badge(item.get('severity'))}</td>"
            f"<td>{_esc(reason_sk)}</td>"
            f"<td class='num'>{_fmt(item.get('actual_kw'), digits=1, suffix=' kW')}</td>"
            f"<td class='num'>{_fmt(item.get('expected_kw'), digits=1, suffix=' kW')}</td>"
            f"<td class='num'>{_fmt(item.get('delta_kw'), digits=1, suffix=' kW')}</td>"
            f"<td class='num'>{_fmt(item.get('robust_z'), digits=2)}</td>"
            "</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Čas</th><th>Závažnosť</th><th>Dôvod</th>"
        "<th class='num'>Namerané</th><th class='num'>Model</th>"
        "<th class='num'>Δ</th><th class='num'>robust z</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _drift_rows(departments: list[dict[str, Any]]) -> str:
    if not departments:
        return ""
    rows = []
    for item in departments:
        rows.append(
            "<tr>"
            f"<td>{_esc(item.get('department_id') or '—')}</td>"
            f"<td>{_severity_badge(item.get('state'))}</td>"
            f"<td class='num'>{_fmt(item.get('psi_load'), digits=3)}</td>"
            f"<td class='num'>{_fmt(item.get('psi_residual'), digits=3)}</td>"
            f"<td class='num'>{_fmt(item.get('residual_mean_shift_kw'), digits=2, suffix=' kW')}</td>"
            "</tr>"
        )
    return (
        "<h3>Drift modelu</h3>"
        "<div class='table-wrap'><table><thead><tr>"
        "<th>Oddelenie</th><th>Stav</th><th class='num'>PSI load</th>"
        "<th class='num'>PSI residual</th><th class='num'>Posun residual</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _reading(payload: dict[str, Any]) -> str:
    chart = payload.get("single_chart") or {}
    kpis = payload.get("decision_kpis") or {}
    labels = chart.get("x") or []
    battery = _series(chart, "battery")
    imp = _series(chart, "import")
    exp = _series(chart, "export")
    soc = _series(chart, "soc")
    bits: list[str] = []
    if labels:
        bits.append(
            f"Graf je <strong>schválený plán na { _esc(str(labels[0])[:10]) }</strong> "
            "(96 × 15 min), nie nameraný D+1. Alerty nižšie sú z merania po dni."
        )
    if exp and max(abs(v) for v in exp) < 1e-6:
        bits.append(
            "Export je celý deň 0 kW — to sedí so zmluvou <strong>export 0 kW</strong> "
            "z konektorov, preto je aj vlastná spotreba 1,00."
        )
    if soc:
        bits.append(
            f"SOC ide z {_fmt(soc[0], digits=0, suffix=' %')} na minimum "
            f"{_fmt(min(soc), digits=0, suffix=' %')} a maximum "
            f"{_fmt(max(soc), digits=0, suffix=' %')} (pásmo 5–95 %)."
        )
    if battery:
        charge = [i for i, v in enumerate(battery) if v < -5]
        discharge = [i for i, v in enumerate(battery) if v > 5]
        if charge:
            bits.append(
                f"Nabíjanie (záporné kW) približne {_hhmm(labels[charge[0]])}–"
                f"{_hhmm(labels[charge[-1]])} — FVE prebytok ide do batérie, nie do siete."
            )
        if discharge:
            bits.append(
                f"Vybíjanie (kladné kW) najmä {_hhmm(labels[discharge[0]])} a popoludní pred večernou špičkou."
            )
    peak_i = _argmax(imp) if imp else None
    if peak_i is not None and labels:
        bits.append(
            f"Špička importu {_fmt(imp[peak_i], digits=1, suffix=' kW')} o {_hhmm(labels[peak_i])} "
            f"— po vybití batérie na dno ide večerný odber zo siete. Limit pripojenia je 500 kW, "
            f"plánovaná špička {_fmt(kpis.get('peak_import_kw'), digits=1, suffix=' kW')} je v limite."
        )
    cost = _num(kpis.get("operating_cost_eur_period"))
    if cost is not None:
        bits.append(
            f"Prevádzkový náklad plánu {_fmt(cost, digits=2, suffix=' €')}; "
            "úspora oproti nákupu celej spotreby zo siete je v páse vyššie. "
            "Toto nie je návratnosť CAPEX — tá ostáva v UC3.2 SEED."
        )
    return "<ul>" + "".join(f"<li>{b}</li>" for b in bits) + "</ul>"


def _comparison_panel(feedback: dict[str, Any]) -> str:
    load_s = feedback.get("load_series") or {}
    pv_s = feedback.get("pv_series") or {}
    if not (load_s.get("x") or pv_s.get("x")):
        return ""
    return """
    <h2>Predikcia vs meranie (D+1)</h2>
    <div class="panel">
      <p class="note-block">Nasimulované meranie po schválenom pláne: neplánovaný odber 8–10 h, výpadok okolo poludnia, slabšia FVE.</p>
      <div class="chart-wrap"><canvas id="compareChart" height="360"></canvas></div>
    </div>
    """


CSS = """
:root {
  --bg: #f5f6f8;
  --panel: #ffffff;
  --ink: #14181f;
  --muted: #5c6675;
  --line: #dfe3e9;
  --accent: #1a6a8c;
  --good: #2f8f5b;
  --warn: #b4761e;
  --bad: #b23a3a;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font: 15px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
}
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }
header { border-bottom: 2px solid var(--ink); padding-bottom: 16px; margin-bottom: 14px; }
.kicker {
  font-size: 11px; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--accent); font-weight: 650; margin: 0 0 8px;
}
h1 { font-size: 26px; margin: 0 0 6px; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 14px; }
h2 {
  font-size: 15px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); margin: 36px 0 14px; font-weight: 600;
}
h3 { font-size: 14px; margin: 18px 0 8px; }
.meta-strip {
  display: flex; align-items: flex-start; gap: 10px;
  margin: 0 0 22px; padding: 7px 12px;
  background: #e8eef2; border: 1px solid #d3dce4; border-radius: 8px;
  font-size: 13px; line-height: 1.45; color: var(--muted);
}
.meta-strip .meta-k {
  flex: 0 0 auto; font-size: 11px; font-weight: 650;
  letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent);
  padding-top: 1px; white-space: nowrap;
}
.meta-strip p { margin: 0; }
.tip {
  display: inline-flex; align-items: center; justify-content: center;
  width: 15px; height: 15px; margin-left: 6px; border-radius: 50%;
  border: 1px solid currentColor; font-size: 10px; font-weight: 700;
  line-height: 1; cursor: help; position: relative; vertical-align: 1px;
  flex: 0 0 auto; opacity: .72;
}
.tip:hover, .tip:focus { opacity: 1; outline: none; z-index: 5; }
.tip .tip-text {
  visibility: hidden; opacity: 0; pointer-events: none;
  position: absolute; z-index: 80; top: calc(100% + 8px); left: 0;
  width: max-content; max-width: min(360px, 70vw);
  padding: 10px 12px; border-radius: 6px; background: #1b212b; color: #f4f6f8;
  font-size: 12px; font-weight: 400; line-height: 1.45; text-transform: none;
  letter-spacing: 0; text-align: left; box-shadow: 0 8px 22px rgba(20,24,31,.22);
}
.card .label .tip .tip-text { left: auto; right: 0; }
.tip:hover .tip-text, .tip:focus .tip-text { visibility: visible; opacity: 1; }
.panel {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 20px 22px; margin-bottom: 18px; overflow: visible;
}
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; }
.card {
  background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
  padding: 16px 18px; position: relative; z-index: 1;
}
.card:hover, .card:focus-within { z-index: 30; }
.card .label {
  display: flex; align-items: flex-start; gap: 2px;
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--muted); line-height: 1.35;
}
.card .value { font-size: 24px; font-weight: 650; margin-top: 6px; letter-spacing: -0.02em; }
.card .note, .note-block { font-size: 12.5px; color: var(--muted); margin-top: 6px; }
.value.good { color: var(--good); }
.value.warn { color: var(--warn); }
.value.bad { color: var(--bad); }
.bill-strip {
  display: grid; grid-template-columns: 1fr auto 1fr auto 1fr minmax(150px, 0.95fr);
  gap: 10px; align-items: stretch;
}
.bill-stat, .hw-stat {
  background: #f7f9fb; border: 1px solid var(--line); border-radius: 8px; padding: 14px 16px;
}
.bill-stat .k, .hw-stat .k {
  font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted);
}
.bill-stat .v { font-size: 26px; font-weight: 650; margin-top: 6px; }
.hw-stat .v { font-size: 18px; font-weight: 650; margin-top: 6px; }
.bill-stat .v.good, .hw-stat .v.good { color: var(--good); }
.bill-stat .note { font-size: 12.5px; color: var(--muted); margin-top: 4px; }
.bill-arrow {
  display: flex; align-items: center; justify-content: center;
  color: var(--muted); font-size: 22px; font-weight: 600;
}
.hw-strip { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 14px; min-width: 640px; }
th, td { text-align: left; padding: 9px 10px; border-bottom: 1px solid var(--line); }
th { font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge {
  display: inline-block; font-size: 12px; padding: 3px 9px; border-radius: 999px;
  border: 1px solid var(--line); background: #f0f3f6; color: var(--muted); margin-right: 6px;
}
.badge.ok { background: #eaf5ef; border-color: #bfe0cd; color: var(--good); }
.badge.warn { background: #fbf2e3; border-color: #ebd7b0; color: var(--warn); }
.badge.bad { background: #fdecec; border-color: #e8b4b4; color: var(--bad); }
.chart-wrap {
  position: relative;
  width: 100%;
  height: 420px;
  max-height: 420px;
  overflow: hidden;
}
.chart-wrap canvas { display: block; width: 100% !important; height: 100% !important; }
ul { margin: 0; padding-left: 20px; }
li { margin: 6px 0; }
@media (max-width: 980px) {
  .bill-strip { grid-template-columns: 1fr 1fr; }
  .bill-arrow { display: none; }
  .hw-strip { grid-template-columns: 1fr 1fr; }
}
@media (max-width: 820px) {
  .bill-strip, .hw-strip { grid-template-columns: 1fr; }
}
"""


def write_ops_dashboard_html(path: Path | str, payload: dict[str, Any]) -> Path:
    """Write Domino-viewable HTML dashboard from ``somes_ops_dashboard_v1`` JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kpis = payload.get("decision_kpis") or {}
    feedback = payload.get("forecast_vs_actual") or {}
    load = feedback.get("load_forecast") or {}
    pv = feedback.get("pv_forecast") or {}
    pv_m = pv["overall"] if isinstance(pv.get("overall"), dict) else pv
    chart = payload.get("single_chart") or {}
    dispatch = payload.get("dispatch_status") or {}
    alerts_block = payload.get("alerts_and_drift") or {}
    alerts = alerts_block.get("summary") or {}
    latest = alerts_block.get("latest") or []
    drift = alerts_block.get("drift") or {}
    tv = payload.get("technical_validation") or {}
    grid = tv.get("grid_feasibility_check") or {}
    batt = tv.get("battery_constraints_check") or {}
    inv = tv.get("inverter_connection_limits_check") or {}
    volt = tv.get("voltage_band_check") or grid.get("voltage_check") or {}
    dq = payload.get("data_quality") or {}
    labels = chart.get("x") or []
    plan_day = str(labels[0])[:10] if labels else "—"
    generated = str(payload.get("generated_at_utc") or "—")

    grid_only = _num(kpis.get("baseline_grid_only_cost_eur"))
    pv_only = _num(kpis.get("baseline_pv_no_battery_cost_eur"))
    plant = _num(kpis.get("operating_cost_eur_period"))
    save_eur = _num(kpis.get("cost_savings_vs_grid_only_eur"))
    save_pct = _num(kpis.get("cost_savings_vs_grid_only_pct"))
    if grid_only is None and plant is not None and save_eur is not None:
        grid_only = plant + save_eur

    closed = feedback.get("closed_loop_ok")
    crit_n = int(alerts.get("critical") or 0)
    warn_n = int(alerts.get("warning") or 0)
    total_n = int(alerts.get("total") or 0)
    alert_klass = "bad" if crit_n else ("warn" if warn_n else "good")

    bill_html = ""
    if plant is not None:
        bill_html = f"""
    <h2>Účet za elektrinu za horizont {_tip("Účet len zo siete, s FVE, s FVE a batériou. Nie je to návratnosť CAPEX — to je UC3.2 SEED.")}</h2>
    <div class="panel">
      <div class="bill-strip">
        <div class="bill-stat"><div class="k">Bez FVE a batérie</div>
          <div class="v">{_esc(_fmt(grid_only, digits=2, suffix=" €"))}</div>
          <div class="note">Celý odber zo siete</div></div>
        <div class="bill-arrow" aria-hidden="true">→</div>
        <div class="bill-stat"><div class="k">S FVE</div>
          <div class="v">{_esc(_fmt(pv_only, digits=2, suffix=" €"))}</div>
          <div class="note">FVE, bez úložiska</div></div>
        <div class="bill-arrow" aria-hidden="true">→</div>
        <div class="bill-stat"><div class="k">S FVE a batériou</div>
          <div class="v">{_esc(_fmt(plant, digits=2, suffix=" €"))}</div>
          <div class="note">Schválený plán D+1</div></div>
        <div class="bill-stat"><div class="k">Úspora vs. sieť</div>
          <div class="v good">{_esc(_fmt(save_eur, digits=2, suffix=" €"))}</div>
          <div class="note">{_esc(_fmt(save_pct, digits=1, suffix=" %"))}</div></div>
      </div>
    </div>
    """

    kpi_cards = "".join(
        [
            _card(
                "Prevádzkové náklady",
                _esc(_fmt(plant, digits=2, suffix=" €")),
                tip="Nákup zo siete mínus tržba z exportu za horizont plánu.",
            ),
            _card(
                "Vlastná spotreba",
                _esc(_fmt(kpis.get("self_consumption_ratio"), digits=3)),
                note="1,0 = FVE neopustila elektromer",
                tip="Podiel výroby FVE spotrebovanej na mieste.",
            ),
            _card(
                "Špička importu",
                _esc(_fmt(kpis.get("peak_import_kw"), digits=1, suffix=" kW")),
                note=f"limit {_fmt(grid.get('import_limit_kw'), digits=0, suffix=' kW')}",
                tip="Najvyšší 15-minútový nákup zo siete v pláne.",
            ),
            _card(
                "Plán schválený",
                _badge(bool(dispatch.get("approved") or kpis.get("dispatch_approved"))),
                tip="GridFeasibilityPiece plán prijal.",
            ),
            _card(
                "Closed loop",
                _badge(closed, yes="OK", no="akcia"),
                klass="" if closed else "warn",
                tip="Či predikcia vs meranie ostala v tolerancii.",
            ),
            _card(
                "Alerty",
                str(total_n),
                klass=alert_klass,
                note=f"kritické {crit_n} · varovania {warn_n}",
                tip="Anomálie spotreby voči modelu na histórii + D+1 meraní.",
            ),
        ]
    )

    eval_cards = "".join(
        [
            _card("MAE spotreby", _esc(_fmt(load.get("mae"), digits=2, suffix=" kW"))),
            _card("RMSE spotreby", _esc(_fmt(load.get("rmse"), digits=2, suffix=" kW"))),
            _card("MAPE spotreby", _esc(_fmt(load.get("mape_pct"), digits=2, suffix=" %"))),
            _card("MAE výroby", _esc(_fmt(pv_m.get("mae"), digits=2, suffix=" kW"))),
            _card("RMSE výroby", _esc(_fmt(pv_m.get("rmse"), digits=2, suffix=" kW"))),
            _card("MAPE výroby", _esc(_fmt(pv_m.get("mape_pct"), digits=2, suffix=" %"))),
        ]
    )

    hw_html = f"""
    <div class="hw-strip">
      <div class="hw-stat"><div class="k">Batéria Pmax</div><div class="v">{_esc(_fmt(batt.get("pmax_kw"), digits=0, suffix=" kW"))}</div></div>
      <div class="hw-stat"><div class="k">Striedač</div><div class="v">{_esc(_fmt(inv.get("inverter_limit_kw"), digits=0, suffix=" kW"))}</div></div>
      <div class="hw-stat"><div class="k">Limit importu</div><div class="v">{_esc(_fmt(grid.get("import_limit_kw"), digits=0, suffix=" kW"))}</div></div>
      <div class="hw-stat"><div class="k">Limit exportu</div><div class="v">{_esc(_fmt(grid.get("export_limit_kw"), digits=0, suffix=" kW"))}</div></div>
    </div>
    """

    details_rows = [
        ("Import zo siete", _fmt(kpis.get("grid_import_kwh"), digits=1, suffix=" kWh")),
        ("Export do siete", _fmt(kpis.get("grid_export_kwh"), digits=1, suffix=" kWh")),
        ("Nabitie batérie", _fmt(kpis.get("battery_charge_kwh"), digits=1, suffix=" kWh")),
        ("Vybíjanie batérie", _fmt(kpis.get("battery_discharge_kwh"), digits=1, suffix=" kWh")),
        ("Priemerné SOC", _fmt(kpis.get("mean_soc_pct"), digits=1, suffix=" %")),
        ("Plán schválený", "áno" if (dispatch.get("approved") or kpis.get("dispatch_approved")) else "nie"),
    ]
    details_html = "<div class='table-wrap'><table>" + "".join(
        f"<tr><th>{_esc(k)}</th><td class='num'>{_esc(v)}</td></tr>" for k, v in details_rows
    ) + "</table></div>"

    volt_txt = "—"
    if _num(volt.get("min_voltage_pu")) is not None:
        volt_txt = (
            f"OK · U {_fmt(volt.get('min_voltage_pu'), digits=3)}–"
            f"{_fmt(volt.get('max_voltage_pu'), digits=3)} p.u."
        )
    grid_html = (
        "<div class='table-wrap'><table>"
        f"<tr><th>Celkový stav</th><td>{_badge(tv.get('overall_ok'), yes='OK', no='zlyhanie')}</td></tr>"
        f"<tr><th>Batéria (výkon / SOC)</th><td>{_badge(batt.get('ok'), yes='OK', no='porušenie')} · Pmax {_esc(_fmt(batt.get('pmax_kw'), digits=0, suffix=' kW'))}</td></tr>"
        f"<tr><th>Striedač a pripojenie</th><td>{_badge(inv.get('ok'), yes='OK', no='porušenie')}</td></tr>"
        f"<tr><th>Limity importu a exportu</th><td>{_badge(grid.get('ok'), yes='OK', no='porušenie')} · špička {_esc(_fmt(grid.get('peak_grid_flow_kw'), digits=1, suffix=' kW'))}</td></tr>"
        f"<tr><th>Napäťové pásmo</th><td><span class='badge ok'>{_esc(volt_txt)}</span></td></tr>"
        f"<tr><th>Plán schválený na D+1</th><td>{_badge(tv.get('approved_next_day_plan'), yes='OK', no='zamietnutý')}</td></tr>"
        "</table></div>"
    )

    dq_bits = [_badge(str(dq.get("overall_severity") or "").lower() in {"ok", "pass", ""}, yes=str(dq.get("overall_severity") or "ok"), no=str(dq.get("overall_severity") or "—"))]
    for name in dq.get("datasets") or []:
        dq_bits.append(f'<span class="badge">{_esc(name)}</span>')
    dq_recs = dq.get("recommendations") or []
    dq_rec_html = "".join(f"<li>{_esc(r)}</li>" for r in dq_recs) if dq_recs else '<p class="note-block">Bez korekcií vstupu.</p>'

    actions = feedback.get("recommended_actions") or []
    actions_html = (
        "<ul>" + "".join(f"<li>{_esc(a)}</li>" for a in actions) + "</ul>"
        if actions
        else "<p class='note-block'>Žiadne odporúčané akcie.</p>"
    )

    chart_payload = {
        "title": chart.get("title") or "Next-day dispatch",
        "x": labels,
        "series": chart.get("series") or [],
        "compare": {
            "load": feedback.get("load_series") or {},
            "pv": feedback.get("pv_series") or {},
        },
    }
    chart_json = json.dumps(chart_payload, ensure_ascii=False)

    doc = f"""<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SoMES — prevádzkový dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>{CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <p class="kicker">UC3.3 SoMES</p>
      <h1>Prevádzkový dashboard na D+1</h1>
      <p class="sub">Plán dispatchu, kvalita predikcie, alerty a technické kontroly · vygenerované {_esc(generated)}</p>
    </header>
    <div class="meta-strip">
      <span class="meta-k">15 min</span>
      <p>Horizont plánu <strong>{_esc(plan_day)}</strong> · 96 intervalov. Graf vyššie je schválený plán; tabuľka alertov je z merania po dni.</p>
    </div>

    {bill_html}

    <h2>Zhrnutie rozhodnutia {_tip("Hlavné prevádzkové čísla schváleného plánu a stav feedback slučky.")}</h2>
    <div class="cards">{kpi_cards}</div>

    <h2>Zariadenie v tomto behu</h2>
    <div class="panel">{hw_html}</div>

    <h2>Hodnotenie predikcie {_tip("Chyba day-ahead vs nasimulované meranie D+1.")}</h2>
    <div class="cards">{eval_cards}</div>

    <h2>Alerty a drift {_tip("Anomálie spotreby voči reziduálnemu modelu. Kritické = |Δ| ≥ 25 kW a vysoké z.")}</h2>
    <div class="panel">
      <p class="note-block">Spolu {total_n}: {_severity_badge("critical")} {crit_n} · {_severity_badge("warning")} {warn_n}. Namerané vs. reziduálny model (nie day-ahead forecast). Kritické |Δ| typicky ≥ 25 kW.</p>
      {_alert_rows(latest)}
      {_drift_rows(drift.get("departments") or [])}
    </div>

    <h2>Dispatch na horizonte {_tip("Kladné battery kW = vybíjanie. SOC je na pravej osi, aby sa nenaťahoval do kW škály.")}</h2>
    <div class="panel">
      <div class="chart-wrap">
        <canvas id="dispatchChart" height="420"></canvas>
      </div>
      <p class="note-block">Modrá batéria · zelený import · žltý export · ružové SOC % (pravá os).</p>
    </div>

    <h2>Čo graf hovorí</h2>
    <div class="panel">{_reading(payload)}</div>

    {_comparison_panel(feedback)}

    <h2>Podrobnosti energie</h2>
    <div class="panel">{details_html}</div>

    <h2>Technická validácia</h2>
    <div class="panel">{grid_html}</div>

    <h2>Kvalita vstupov</h2>
    <div class="panel">{"".join(dq_bits)}{dq_rec_html}</div>

    <h2>Odporúčané akcie</h2>
    <div class="panel">{actions_html}</div>
  </div>
  <script>
    const payload = {chart_json};
    const labels = (payload.x || []).map((t) => {{
      const s = String(t);
      return s.includes(" ") ? s.split(" ")[1].slice(0, 5) : s;
    }});
    const palette = {{
      battery: "#1a6a8c",
      import: "#b23a3a",
      export: "#2f8f5b",
      soc: "#b4761e",
    }};
    const datasets = (payload.series || []).map((s) => {{
      const name = String(s.name || "");
      const isSoc = /soc/i.test(name) || s.unit === "%";
      const key = /battery/i.test(name) ? "battery" : /import/i.test(name) ? "import" : /export/i.test(name) ? "export" : "soc";
      return {{
        label: name,
        data: s.values || [],
        borderColor: palette[key],
        backgroundColor: "transparent",
        tension: 0.2,
        pointRadius: 0,
        borderWidth: isSoc ? 2 : 1.8,
        borderDash: isSoc ? [5, 4] : [],
        yAxisID: isSoc ? "soc" : "kw",
      }};
    }});
    new Chart(document.getElementById("dispatchChart"), {{
      type: "line",
      data: {{ labels, datasets }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        resizeDelay: 200,
        interaction: {{ mode: "index", intersect: false }},
        plugins: {{ legend: {{ labels: {{ color: "#14181f" }} }} }},
        scales: {{
          x: {{ ticks: {{ color: "#5c6675", maxTicksLimit: 12 }}, grid: {{ color: "#dfe3e9" }} }},
          kw: {{
            type: "linear",
            position: "left",
            title: {{ display: true, text: "kW", color: "#5c6675" }},
            ticks: {{ color: "#5c6675" }},
            grid: {{ color: "#dfe3e9" }},
          }},
          soc: {{
            type: "linear",
            position: "right",
            min: 0,
            max: 100,
            title: {{ display: true, text: "SOC %", color: "#b4761e" }},
            ticks: {{ color: "#b4761e" }},
            grid: {{ drawOnChartArea: false }},
          }},
        }},
      }},
    }});
    const cmp = payload.compare || {{}};
    const cmpCanvas = document.getElementById("compareChart");
    if (cmpCanvas && ((cmp.load && cmp.load.x) || (cmp.pv && cmp.pv.x))) {{
      const src = (cmp.load && (cmp.load.x || []).length) ? cmp.load : cmp.pv;
      const cmpLabels = (src.x || []).map((t) => {{
        const s = String(t);
        return s.includes(" ") ? s.split(" ")[1].slice(0, 5) : s;
      }});
      const cmpSets = [];
      if (cmp.load && (cmp.load.forecast || []).length) {{
        cmpSets.push({{ label: "Spotreba plán (kW)", data: cmp.load.forecast, borderColor: "#1a6a8c", backgroundColor: "transparent", tension: 0.2, pointRadius: 0, borderWidth: 1.7 }});
        cmpSets.push({{ label: "Spotreba meranie (kW)", data: cmp.load.actual, borderColor: "#b23a3a", backgroundColor: "transparent", tension: 0.2, pointRadius: 0, borderWidth: 1.7, borderDash: [4, 3] }});
      }}
      if (cmp.pv && (cmp.pv.forecast || []).length) {{
        cmpSets.push({{ label: "FVE plán (kW)", data: cmp.pv.forecast, borderColor: "#2f8f5b", backgroundColor: "transparent", tension: 0.2, pointRadius: 0, borderWidth: 1.7 }});
        cmpSets.push({{ label: "FVE meranie (kW)", data: cmp.pv.actual, borderColor: "#b4761e", backgroundColor: "transparent", tension: 0.2, pointRadius: 0, borderWidth: 1.7, borderDash: [4, 3] }});
      }}
      new Chart(cmpCanvas, {{
        type: "line",
        data: {{ labels: cmpLabels, datasets: cmpSets }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          resizeDelay: 200,
          interaction: {{ mode: "index", intersect: false }},
          plugins: {{ legend: {{ labels: {{ color: "#14181f" }} }} }},
          scales: {{
            x: {{ ticks: {{ color: "#5c6675", maxTicksLimit: 12 }}, grid: {{ color: "#dfe3e9" }} }},
            y: {{ ticks: {{ color: "#5c6675" }}, grid: {{ color: "#dfe3e9" }}, title: {{ display: true, text: "kW" }} }},
          }},
        }},
      }});
    }}
  </script>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out
