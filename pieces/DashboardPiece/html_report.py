"""Render SoMES operational dashboard payload to a self-contained HTML file for Domino UI."""
from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def _fmt(value: Any, *, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return str(value)


def write_ops_dashboard_html(path: Path | str, payload: dict[str, Any]) -> Path:
    """Write Domino-viewable HTML dashboard from ``somes_ops_dashboard_v1`` JSON."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    kpis = payload.get("decision_kpis") or {}
    feedback = payload.get("forecast_vs_actual") or {}
    load = feedback.get("load_forecast") or {}
    pv = feedback.get("pv_forecast") or {}
    if isinstance(pv.get("overall"), dict):
        pv_m = pv["overall"]
    else:
        pv_m = pv
    chart = payload.get("single_chart") or {}
    dispatch = payload.get("dispatch_status") or {}
    alerts = (payload.get("alerts_and_drift") or {}).get("summary") or {}

    chart_json = json.dumps(
        {
            "title": chart.get("title") or "Next-day dispatch",
            "x": chart.get("x") or [],
            "series": chart.get("series") or [],
        },
        ensure_ascii=False,
    )

    cards = [
        ("Spotreba MAE (kW)", _fmt(load.get("mae"))),
        ("Spotreba RMSE (kW)", _fmt(load.get("rmse"))),
        ("Spotreba MAPE", _fmt(load.get("mape_pct"), suffix=" %")),
        ("Výroba MAE", _fmt(pv_m.get("mae"))),
        ("Výroba RMSE", _fmt(pv_m.get("rmse"))),
        ("Výroba MAPE", _fmt(pv_m.get("mape_pct"), suffix=" %")),
        ("Peak import (kW)", _fmt(kpis.get("peak_import_kw"), digits=1)),
        ("Self-consumption", _fmt(kpis.get("self_consumption_ratio"), digits=3)),
        ("Dispatch OK", "áno" if dispatch.get("overall_ok") else "nie"),
        ("Alerts", str(int(alerts.get("total") or 0))),
    ]
    cards_html = "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )

    actions = feedback.get("recommended_actions") or []
    actions_html = (
        "<ul>" + "".join(f"<li>{html.escape(str(a))}</li>" for a in actions) + "</ul>"
        if actions
        else "<p class='muted'>Žiadne odporúčané akcie.</p>"
    )

    generated = html.escape(str(payload.get("generated_at_utc") or "—"))

    doc = f"""<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SoMES operational dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #1a2332;
      --text: #e7ecf3;
      --muted: #9aa7b8;
      --accent: #3d9cfd;
      --border: #2a3648;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
    }}
    h1 {{ margin: 0 0 4px; font-size: 1.6rem; }}
    .caption {{ color: var(--muted); margin-bottom: 20px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 12px;
      margin-bottom: 24px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
    }}
    .label {{ color: var(--muted); font-size: 0.8rem; margin-bottom: 6px; }}
    .value {{ font-size: 1.25rem; font-weight: 600; }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 20px;
    }}
    canvas {{ width: 100%; height: 420px; }}
    .muted {{ color: var(--muted); }}
  </style>
</head>
<body>
  <h1>SoMES operational dashboard</h1>
  <div class="caption">Vygenerované (UTC): {generated}</div>

  <h2>Kvalita predikcie (MAE / RMSE)</h2>
  <div class="grid">
    {cards_html}
  </div>

  <div class="panel">
    <h2 id="chart-title">Dispatch</h2>
    <canvas id="dispatchChart"></canvas>
  </div>

  <div class="panel">
    <h2>Odporúčané akcie</h2>
    {actions_html}
  </div>

  <script>
    const payload = {chart_json};
    document.getElementById("chart-title").textContent = payload.title || "Dispatch";
    const colors = ["#3d9cfd", "#5ad67d", "#f0b429", "#e85d75"];
    const datasets = (payload.series || []).map((s, i) => ({{
      label: s.name || ("series " + (i + 1)),
      data: s.values || [],
      borderColor: colors[i % colors.length],
      backgroundColor: "transparent",
      tension: 0.2,
      pointRadius: 0,
      borderWidth: 2,
    }}));
    new Chart(document.getElementById("dispatchChart"), {{
      type: "line",
      data: {{ labels: payload.x || [], datasets }},
      options: {{
        responsive: true,
        maintainAspectRatio: false,
        plugins: {{ legend: {{ labels: {{ color: "#e7ecf3" }} }} }},
        scales: {{
          x: {{ ticks: {{ color: "#9aa7b8", maxTicksLimit: 12 }}, grid: {{ color: "#2a3648" }} }},
          y: {{ ticks: {{ color: "#9aa7b8" }}, grid: {{ color: "#2a3648" }} }},
        }},
      }},
    }});
  </script>
</body>
</html>
"""
    out.write_text(doc, encoding="utf-8")
    return out
