from pydantic import BaseModel, Field

METRIC_HELP: dict[str, str] = {
    "savings_period": "Úspora na nákladoch za elektrinu počas simulovaného obdobia.",
    "capex": "Jednorazová investícia do FVE a batérie (€).",
    "payback": "Návratnosť investície v rokoch.",
    "npv": "Čistá súčasná hodnota (NPV) v €.",
    "target_payback": "Cieľová doba návratnosti z formulára (roky).",
    "achieved_payback": "Dosiahnutá návratnosť odporúčaného variantu (roky).",
    "recommended_kwp": "Odporúčaný výkon FVE (kWp).",
    "recommended_kwh": "Odporúčaná kapacita batérie (kWh).",
    "capex_fve_bess": "CAPEX FVE + batéria (€).",
    "annual_savings_inv": "Ročná prevádzková úspora z investičného návrhu (€).",
    "annual_savings": "Odhad ročnej prevádzkovej úspory zo simulácie (€).",
    "npv_inv": "NPV odporúčaného variantu (€).",
    "npv_full": "NPV z časovej simulácie (€).",
    "total_investment": "Celková investícia CAPEX zo simulácie (€).",
    "grid_pick": "Vyberte variant z mriežky FVE × batéria na porovnanie.",
    "grid_fve_kwp": "Výkon FVE vo zvolenom variante mriežky (kWp).",
    "grid_battery_kwh": "Kapacita batérie vo zvolenom variante (kWh).",
    "grid_payback": "Návratnosť vo zvolenom variante (roky).",
    "grid_npv": "NPV vo zvolenom variante (€).",
    "alerts_total": "Počet anomálií spotreby z posledného behu modelu.",
    "alerts_critical": "Kritické odchýlky spotreby oproti predikcii.",
    "alerts_warning": "Varovania — významnejšie odchýlky.",
    "alerts_info": "Informačné upozornenia (menšie odchýlky).",
    "scenario_select": "Scenár simulácie (FVE/batéria) na zobrazenie grafov.",
    "solar_pv_capacity": "Inštalovaný výkon FVE v scenári (kWp).",
    "battery_capacity": "Kapacita batérie v scenári (kWh).",
    "consumption_baseline": "Spotreba bez FVE a batérie za simulované obdobie (kWh).",
    "consumption_with_pv_bess": "Spotreba so FVE a batériou za simulované obdobie (kWh).",
    "consumption_baseline_year": "Ročný ekvivalent spotreby bez FVE/batérie (kWh/rok).",
    "consumption_with_year": "Ročný ekvivalent spotreby s FVE/batériou (kWh/rok).",
    "cost_baseline": "Náklady na elektrinu bez FVE a batérie (€).",
    "cost_with_pv_bess": "Náklady so FVE a batériou (€).",
    "cost_savings": "Úspora nákladov oproti baseline (€).",
    "cost_baseline_year": "Ročný ekvivalent nákladov bez FVE/batérie (€).",
    "cost_with_year": "Ročný ekvivalent nákladov s FVE/batériou (€).",
    "savings_year": "Ročný ekvivalent úspory (€).",
    "payback_grid": "Návratnosť z parametrického odhadu mriežky (roky).",
    "savings_grid": "Ročná úspora z parametrického odhadu (€).",
    "npv_grid": "NPV z parametrického odhadu (€).",
}


class InputModel(BaseModel):
    report_json: str = Field(default="", description="Path to mrk_savings_report.json (legacy/SEED appendix)")
    kpi_results_csv: str = Field(default="", description="Path to kpi_results.csv (legacy)")
    investment_evaluation_csv: str = Field(default="", description="Path to investment_evaluation.csv (SEED appendix)")
    anomaly_alerts_csv: str | None = Field(default=None, description="Optional path to anomaly_alerts.csv")
    drift_report_json: str | None = Field(default=None, description="Optional path to drift_report.json")
    next_day_dispatch_plan_csv: str | None = Field(
        default=None, description="SoMES next-day dispatch plan (enables operational dashboard)"
    )
    operational_kpis_json: str | None = Field(default=None, description="SoMES operational KPIs JSON")
    technical_validation_json: str | None = Field(default=None, description="SoMES technical validation JSON")
    forecast_vs_actual_json: str | None = Field(
        default=None, description="Feedback loop bundle from ModelMonitoringPiece"
    )
    data_quality_report_json: str | None = Field(
        default=None, description="Data validation report from ingestion/preprocessing"
    )


class OutputModel(BaseModel):
    dashboard_data_json: str
