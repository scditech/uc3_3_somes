"""Bilingual SoMES operational form (UC3.2 visual language, SoMES fields)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from common_somes.ui_theme import CSS, LANG_JS, esc, labeled, lang_switch, pack_script
except ModuleNotFoundError:
    from pieces.common_somes.ui_theme import CSS, LANG_JS, esc, labeled, lang_switch, pack_script

# Every visible field has SK copy (HTML default) and an EN overlay key.
EN: dict[str, str] = {
    "kicker": "UC3.3 SoMES",
    "title": "Operational input form",
    "sub": "Values used for this run. Hover the question mark for the full explanation of each field.",
    "sec.site": "Site",
    "sec.site.tip": "Where the offtake sits. Latitude and longitude drive Open-Meteo and SHMÚ.",
    "sec.data": "Load and prices",
    "sec.data.tip": "Metered consumption and the electricity price used in dispatch.",
    "sec.pv": "Photovoltaics",
    "sec.pv.tip": "Installed array used by the physical PVOUT model (Open-Meteo, not a commercial irradiance vendor).",
    "sec.bat": "Battery",
    "sec.bat.tip": "Usable energy, C-rate and starting SOC when BMS telemetry is missing.",
    "sec.grid": "Grid connection",
    "sec.grid.tip": "Import, export and inverter limits at the point of common coupling.",
    "sec.dispatch": "Dispatch and contract",
    "sec.dispatch.tip": "How the next-day plan is solved and the reserved-capacity contract.",
    "sec.sources": "Data sources",
    "sec.sources.tip": "Open-Meteo, SHMÚ and OKTE replace any commercial irradiance feed.",
    "sec.ems": "EMS / BMS / DRMS",
    "sec.ems.tip": "Optional live delivery. The file payload is always written.",
    "f.site_name": "Site name",
    "f.site_name.tip": "Name of the plant or offtake. Shown on the dashboard and in the EMS payload.",
    "f.latitude": "Latitude",
    "f.latitude.tip": "Decimal degrees north. Together with longitude this is the Open-Meteo and SHMÚ query point.",
    "f.longitude": "Longitude",
    "f.longitude.tip": "Decimal degrees east. Must be given together with latitude.",
    "f.timezone": "Timezone",
    "f.timezone.tip": "IANA timezone of the offtake. Load timestamps and OKTE DAM are aligned in this zone, typically Europe/Bratislava.",
    "f.load": "Load file",
    "f.load.tip": "CSV with datetime and load_kw, or department columns that are summed to the site total. Optional price_eur_per_kwh.",
    "f.prices": "Prices file",
    "f.prices.tip": "Optional CSV with datetime and price_eur_per_kwh. If empty, prices in the load file are used, otherwise OKTE DAM is downloaded for the same window.",
    "f.price_source": "Price source used",
    "f.price_source.tip": "Where the commodity price actually came from after validation: load file, extra CSV, or OKTE DAM.",
    "f.horizon": "Datetime window",
    "f.horizon.tip": "First and last timestamp after merge and gap repair. Open-Meteo is fetched for exactly this window.",
    "f.kwp": "Installed PV (kWp)",
    "f.kwp.tip": "Nameplate DC capacity. Physical PVOUT is computed for this size and later scaled only if you change kWp without retraining.",
    "f.tilt": "Tilt (deg)",
    "f.tilt.tip": "Array tilt from horizontal. Used in the Hay–Davies transposition of GHI onto the plane of the array.",
    "f.az": "Azimuth (deg)",
    "f.az.tip": "Array azimuth. 180° is due south in the northern hemisphere.",
    "f.kwh": "Battery energy (kWh)",
    "f.kwh.tip": "Usable AC-side energy of the pack. Dispatch treats this as the SOC 0–100 % range.",
    "f.crate": "C-rate",
    "f.crate.tip": "Charge and discharge power as a fraction of energy. 0.5 C on 400 kWh is 200 kW.",
    "f.soc": "Initial SOC (%)",
    "f.soc.tip": "Starting state of charge when no BMS snapshot is available. Telemetry overrides this.",
    "f.imp": "Import limit (kW)",
    "f.imp.tip": "Maximum power that may be drawn from the grid. The LP dispatch and grid check both honour it.",
    "f.exp": "Export limit (kW)",
    "f.exp.tip": "Maximum power that may be fed into the grid.",
    "f.inv": "Inverter limit (kW)",
    "f.inv.tip": "AC limit of the inverter / power conversion system.",
    "f.method": "Dispatch method",
    "f.method.tip": "auto tries linear programming (HiGHS) and falls back to a greedy heuristic. lp forces the optimiser. greedy skips it.",
    "f.deg": "Degradation cost (€/kWh)",
    "f.deg.tip": "Throughput cost in the dispatch objective so the solver does not cycle the battery for tiny spreads.",
    "f.mrk": "MRK contract (kW)",
    "f.mrk.tip": "Contracted reserved capacity with the DSO. Peak import above this attracts the penalty in the scenario.",
    "f.hours": "Forecast horizon (h)",
    "f.hours.tip": "How many hours ahead the load model is asked to predict. Next-day dispatch uses 24 hours at 15-minute steps (96 intervals).",
    "f.om": "Open-Meteo",
    "f.om.tip": "Public weather archive / historical forecast. Physical PVOUT is computed from GHI, DNI, temperature and wind — not a linear GHI scaling.",
    "f.shmu": "SHMÚ calibration",
    "f.shmu.tip": "Compares modelled GHI with the nearest Slovak Hydrometeorological Institute station. The scale factor trains the UC3.4 error-correction model. If there is no overlap the step is a no-op.",
    "f.okte": "OKTE DAM",
    "f.okte.tip": "Public day-ahead market results (ISOT). Used only when the load file has no usable price column and no extra prices CSV is given.",
    "f.ems": "EMS endpoint",
    "f.ems.tip": "HTTP URL of the energy management system. Empty means the JSON/CSV payload and Modbus map are written but not posted.",
    "f.bms": "BMS",
    "f.bms.tip": "Battery management system. SoMES reads SOC and available charge/discharge from bess_telemetry.csv (or a URL in connectors). There is no live BMS protocol in this run.",
    "f.drms": "DRMS / flexible load",
    "f.drms.tip": "Demand-response style shifting of declared flexible loads into cheaper windows. Configured in flexible_loads.json, not as a market API.",
    "yes": "yes",
    "no": "no",
    "source.strong": "Open-Meteo, SHMÚ and OKTE",
    "source.body": "Public weather, the nearest SHMÚ station and the Slovak day-ahead market. There is no commercial irradiance vendor.",
    "run": "Run SoMES",
    "f.quick": "Quick run (last 7 days of load)",
    "f.quick.tip": "Clips the uploaded or demo load to the last week so Open-Meteo and training finish faster. Uncheck for the full file window.",
    "f.skip_load": "Skip load-forecast training",
    "f.skip_load.tip": "Dispatch then uses the historical load curve itself. Uncheck to train the XGBoost D+1 load model (slower; fills load MAE on the dashboard).",
    "f.skip_ops": "Skip LP dispatch / EMS",
    "f.skip_ops.tip": "Stops after PV inference. Use only when you want model metrics without a battery plan.",
    "sec.run": "Run",
    "sec.run.tip": "Local-only switches. In Domino the DAG always trains load and runs dispatch.",
    "progress": "Computing… Open-Meteo and XGBoost can take several minutes.",
    "progress.title": "SoMES is running",
    "back": "← Input form",
}

SECTIONS = [
    ("sec.site", "Lokalita", "Kde odberné miesto leží. Šírka a dĺžka idú do Open-Meteo a SHMÚ."),
    ("sec.data", "Odber a ceny", "Meraná spotreba a cena elektriny, ktorú používa dispatch."),
    ("sec.pv", "Fotovoltika", "Inštalované pole pre fyzikálny PVOUT (Open-Meteo, nie komerčný dodávateľ žiarenia)."),
    ("sec.bat", "Batéria", "Použiteľná energia, C-rate a počiatočné SOC, ak chýba telemetria BMS."),
    ("sec.grid", "Pripojenie k sieti", "Limity importu, exportu a striedača v mieste pripojenia."),
    ("sec.dispatch", "Dispatch a zmluva", "Ako sa počíta plán na D+1 a zmluva o rezervovanej kapacite."),
    ("sec.sources", "Zdroje dát", "Open-Meteo, SHMÚ a OKTE nahrádzajú komerčný zdroj žiarenia."),
    ("sec.ems", "EMS / BMS / DRMS", "Voliteľné živé doručenie. Súborový payload sa zapíše vždy."),
]


def _yn(flag: bool) -> tuple[str, str]:
    return ("áno", "yes") if flag else ("nie", "no")


def _get(cfg: dict, *path: str, default: Any = "—") -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur if cur not in (None, "") else default


def _row(label_sk: str, i18n: str, tip_sk: str, tip_i18n: str, value: Any) -> str:
    return (
        "<tr>"
        f"<th>{labeled(label_sk, tip_sk, i18n=i18n, i18n_tip=tip_i18n)}</th>"
        f"<td>{esc(value)}</td>"
        "</tr>"
    )


def render_form_html(
    *,
    scenario: dict[str, Any],
    summary: dict[str, Any],
    flags: dict[str, Any],
) -> str:
    site = scenario.get("site") or {}
    pv = scenario.get("pv") or {}
    bat = scenario.get("battery") or {}
    grid = scenario.get("grid") or {}
    disp = scenario.get("dispatch") or {}
    mrk = scenario.get("mrk") or {}
    price_source = summary.get("price_source") or "—"
    window = f"{summary.get('datetime_min', '—')} → {summary.get('datetime_max', '—')}"
    om = _yn(bool(flags.get("enable_open_meteo", True)))
    shmu = _yn(bool(flags.get("enable_shmu", True)))
    okte = _yn(bool(flags.get("enable_okte", True)))

    blocks = [
        (
            "sec.site",
            "Lokalita",
            "Kde odberné miesto leží. Šírka a dĺžka idú do Open-Meteo a SHMÚ.",
            [
                _row("Názov prevádzky", "f.site_name", "Názov závodu alebo odberného miesta. Zobrazí sa na dashboarde a v EMS payloade.", "f.site_name.tip", site.get("name") or scenario.get("description") or "—"),
                _row("Zemepisná šírka", "f.latitude", "Desatinné stupne na sever. Spolu s dĺžkou je to bod dopytu Open-Meteo a SHMÚ.", "f.latitude.tip", site.get("latitude") or flags.get("latitude") or "—"),
                _row("Zemepisná dĺžka", "f.longitude", "Desatinné stupne na východ. Musí byť zadaná spolu so šírkou.", "f.longitude.tip", site.get("longitude") or flags.get("longitude") or "—"),
                _row("Časové pásmo", "f.timezone", "IANA zóna odberu. Pečiatky odberu a OKTE DAM sa na ňu zarovnajú, spravidla Europe/Bratislava.", "f.timezone.tip", scenario.get("timezone") or "Europe/Bratislava"),
            ],
        ),
        (
            "sec.data",
            "Odber a ceny",
            "Meraná spotreba a cena elektriny, ktorú používa dispatch.",
            [
                _row("Súbor odberu", "f.load", "CSV s datetime a load_kw, alebo stĺpce oddelení, ktoré sa sčítajú na súčet lokality. Voliteľne price_eur_per_kwh.", "f.load.tip", (summary.get("input_paths") or {}).get("load_csv") or "—"),
                _row("Súbor cien", "f.prices", "Voliteľný CSV s datetime a price_eur_per_kwh. Ak je prázdny, použijú sa ceny v odbere, inak sa stiahne OKTE DAM na rovnaké okno.", "f.prices.tip", (summary.get("input_paths") or {}).get("prices_csv") or "—"),
                _row("Zdroj ceny", "f.price_source", "Odkiaľ po validácii naozaj prišla komoditná cena: súbor odberu, samostatný CSV, alebo OKTE DAM.", "f.price_source.tip", price_source),
                _row("Časové okno", "f.horizon", "Prvá a posledná pečiatka po zlúčení a oprave medzier. Open-Meteo sa sťahuje presne na toto okno.", "f.horizon.tip", window),
            ],
        ),
        (
            "sec.pv",
            "Fotovoltika",
            "Inštalované pole pre fyzikálny PVOUT (Open-Meteo, nie komerčný dodávateľ žiarenia).",
            [
                _row("Inštalovaný výkon (kWp)", "f.kwp", "Menovitý DC výkon. Fyzikálny PVOUT sa počíta pre túto veľkosť.", "f.kwp.tip", pv.get("installed_kwp")),
                _row("Sklon (°)", "f.tilt", "Sklon poľa od vodorovnej roviny. Používa sa pri transpozícii GHI na rovinu poľa (Hay–Davies).", "f.tilt.tip", pv.get("tilt_deg")),
                _row("Azimut (°)", "f.az", "Azimut poľa. 180° je juh na severnej pologuli.", "f.az.tip", pv.get("azimuth_deg")),
            ],
        ),
        (
            "sec.bat",
            "Batéria",
            "Použiteľná energia, C-rate a počiatočné SOC, ak chýba telemetria BMS.",
            [
                _row("Energia batérie (kWh)", "f.kwh", "Použiteľná AC energia packu. Dispatch berie tento rozsah ako SOC 0–100 %.", "f.kwh.tip", bat.get("energy_kwh") or bat.get("capacity_kWh")),
                _row("C-rate", "f.crate", "Výkon nabíjania a vybíjania ako podiel energie. 0,5 C pri 400 kWh je 200 kW.", "f.crate.tip", bat.get("max_c_rate")),
                _row("Počiatočné SOC (%)", "f.soc", "Počiatočný stav nabitia, ak nie je snímka BMS. Telemetria ho pretlačí.", "f.soc.tip", bat.get("initial_soc_pct")),
            ],
        ),
        (
            "sec.grid",
            "Pripojenie k sieti",
            "Limity importu, exportu a striedača v mieste pripojenia.",
            [
                _row("Limit importu (kW)", "f.imp", "Maximálny odber zo siete. LP dispatch aj kontrola pripojenia ho rešpektujú.", "f.imp.tip", grid.get("import_limit_kw")),
                _row("Limit exportu (kW)", "f.exp", "Maximálny dodáv do siete.", "f.exp.tip", grid.get("export_limit_kw")),
                _row("Limit striedača (kW)", "f.inv", "AC limit striedača / meniča.", "f.inv.tip", grid.get("inverter_limit_kw")),
            ],
        ),
        (
            "sec.dispatch",
            "Dispatch a zmluva",
            "Ako sa počíta plán na D+1 a zmluva o rezervovanej kapacite.",
            [
                _row("Metóda dispatchu", "f.method", "auto skúsi lineárne programovanie (HiGHS) a pri zlyhaní ide na greedy. lp vynúti optimalizér. greedy ho preskočí.", "f.method.tip", disp.get("method") or "auto"),
                _row("Náklad degradácie (€/kWh)", "f.deg", "Náklad na pretočenú energiu v účelovej funkcii, aby sa batéria netočíla pre maličké spready.", "f.deg.tip", disp.get("degradation_cost_eur_per_kwh")),
                _row("Zmluva MRK (kW)", "f.mrk", "Zmluvný rezervovaný výkon u prevádzkovateľa. Špička nad ním ťahá penalizáciu zo scenára.", "f.mrk.tip", mrk.get("contract_kw")),
                _row("Horizont predikcie (h)", "f.hours", "Koľko hodín dopredu má model spotreby predikovať. Plán na D+1 používa 24 hodín po 15 minútach (96 intervalov).", "f.hours.tip", flags.get("horizon_hours") or 24),
            ],
        ),
        (
            "sec.sources",
            "Zdroje dát",
            "Open-Meteo, SHMÚ a OKTE nahrádzajú komerčný zdroj žiarenia.",
            [
                _row("Open-Meteo", "f.om", "Verejný archív / historical-forecast. Fyzikálny PVOUT sa počíta z GHI, DNI, teploty a vetra — nie lineárnym násobkom GHI.", "f.om.tip", om[0]),
                _row("Kalibrácia SHMÚ", "f.shmu", "Porovná modelové GHI s najbližšou stanicou SHMÚ. Faktor škály trénuje korekčný model UC3.4. Bez prekrytia je krok no-op.", "f.shmu.tip", shmu[0]),
                _row("OKTE DAM", "f.okte", "Verejné výsledky denného trhu (ISOT). Použije sa len vtedy, keď v odberi nie je použiteľná cena a nie je zadaný samostatný CSV.", "f.okte.tip", okte[0]),
            ],
        ),
        (
            "sec.ems",
            "EMS / BMS / DRMS",
            "Voliteľné živé doručenie. Súborový payload sa zapíše vždy.",
            [
                _row("Endpoint EMS", "f.ems", "HTTP adresa energy management systému. Prázdne znamená, že JSON/CSV a mapa registrov sa zapíšu, ale nepošlú.", "f.ems.tip", flags.get("ems_endpoint_url") or "—"),
                _row("BMS", "f.bms", "Battery management system. SoMES číta SOC a dostupný výkon z bess_telemetry.csv (alebo URL v konektoroch). Živý protokol BMS v tomto behu nie je.", "f.bms.tip", "súbor / URL telemetrie"),
                _row("DRMS / flexibilný odber", "f.drms", "Presun deklarovaných flexibilných odberov do lacnejších okien. Nastavuje sa v flexible_loads.json, nie ako trhové API.", "f.drms.tip", "flexible_loads.json"),
            ],
        ),
    ]

    parts = []
    for key, title, hint, rows in blocks:
        parts.append(
            f'<h2>{labeled(title, hint, i18n=key, i18n_tip=key + ".tip")}</h2>'
            f'<div class="panel"><table>{"".join(rows)}</table></div>'
        )

    return f"""<!DOCTYPE html>
<html lang="sk">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SoMES — vstupný formulár</title>
  <style>{CSS}</style>
</head>
<body>
  <div class="wrap">
    <header>
      <div class="head-row">
        <div>
          <p class="kicker" data-i18n="kicker">UC3.3 SoMES</p>
          <h1 data-title-sk="Prevádzkový vstupný formulár" data-title-en="Operational input form">Prevádzkový vstupný formulár</h1>
          <p class="sub" data-i18n="sub">Hodnoty použité v tomto behu. Na otáznik ukážte myšou — pri každej položke je úplné vysvetlenie.</p>
        </div>
        {lang_switch()}
      </div>
    </header>
    {"".join(parts)}
  </div>
  {pack_script(EN)}
  <script>{LANG_JS}</script>
</body>
</html>
"""


def write_form_html(path: Path | str, **kwargs: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_form_html(**kwargs), encoding="utf-8")
    return out


def _attr(value: Any) -> str:
    return esc("" if value is None else value)


def _field(
    name: str,
    label_sk: str,
    i18n: str,
    tip_sk: str,
    *,
    value: Any = "",
    kind: str = "text",
    step: str = "any",
    extra: str = "",
) -> str:
    if kind == "file":
        control = f'<input type="file" name="{esc(name)}" accept=".csv,text/csv" {extra}/>'
    elif kind == "select":
        control = extra
    else:
        itype = "number" if kind == "number" else "text"
        step_attr = f' step="{esc(step)}"' if itype == "number" else ""
        control = (
            f'<input type="{itype}" name="{esc(name)}" value="{_attr(value)}"{step_attr} {extra}/>'
        )
    return (
        '<div class="field">'
        f"<label>{labeled(label_sk, tip_sk, i18n=i18n, i18n_tip=i18n + '.tip')}</label>"
        f"{control}"
        "</div>"
    )


def _check(name: str, label_sk: str, i18n: str, tip_sk: str, *, checked: bool) -> str:
    flag = " checked" if checked else ""
    return (
        f'<label class="check"><input type="checkbox" name="{esc(name)}" value="1"{flag}/>'
        f"{labeled(label_sk, tip_sk, i18n=i18n, i18n_tip=i18n + '.tip')}</label>"
    )


def render_interactive_form(defaults: dict[str, Any] | None = None) -> str:
    d = defaults or {}
    site = d.get("site") or {}
    pv = d.get("pv") or {}
    bat = d.get("battery") or {}
    grid = d.get("grid") or {}
    disp = d.get("dispatch") or {}
    mrk = d.get("mrk") or {}
    method = str(disp.get("method") or "auto")
    options = "".join(
        f'<option value="{esc(m)}"{" selected" if m == method else ""}>{esc(m)}</option>'
        for m in ("auto", "lp", "greedy")
    )
    body = f"""
    <div class="source-box">
      <strong data-i18n="source.strong">Open-Meteo, SHMÚ a OKTE</strong>
      <span data-i18n="source.body">Verejné počasie, najbližšia stanica SHMÚ a slovenský denný trh. Komerčný dodávateľ žiarenia v tomto behu nie je.</span>
    </div>
    <form method="post" action="/run" enctype="multipart/form-data" id="somes-form">
      <h2>{labeled("Odber a ceny", "Meraná spotreba a cena elektriny, ktorú používa dispatch.", i18n="sec.data", i18n_tip="sec.data.tip")}</h2>
      <div class="panel grid-2">
        {_field("load_csv", "Súbor odberu (CSV)", "f.load", "CSV s datetime a load_kw, alebo stĺpce oddelení. Prázdne = demo z somes_inputs.", kind="file")}
        {_field("prices_csv", "Súbor cien (CSV, voliteľné)", "f.prices", "datetime + price_eur_per_kwh. Prázdne = ceny v odbere, inak OKTE DAM.", kind="file")}
      </div>
      <h2>{labeled("Lokalita", "Kde odberné miesto leží. Šírka a dĺžka idú do Open-Meteo a SHMÚ.", i18n="sec.site", i18n_tip="sec.site.tip")}</h2>
      <div class="panel grid-2">
        {_field("site_name", "Názov prevádzky", "f.site_name", "Názov závodu alebo odberného miesta.", value=site.get("name") or "Demo plant")}
        {_field("timezone", "Časové pásmo", "f.timezone", "IANA zóna odberu, spravidla Europe/Bratislava.", value=d.get("timezone") or "Europe/Bratislava")}
        {_field("latitude", "Zemepisná šírka", "f.latitude", "Desatinné stupne na sever.", value=site.get("latitude") or 48.17, kind="number")}
        {_field("longitude", "Zemepisná dĺžka", "f.longitude", "Desatinné stupne na východ.", value=site.get("longitude") or 17.07, kind="number")}
      </div>
      <h2>{labeled("Fotovoltika", "Inštalované pole pre fyzikálny PVOUT (Open-Meteo, nie komerčný dodávateľ žiarenia).", i18n="sec.pv", i18n_tip="sec.pv.tip")}</h2>
      <div class="panel grid-2">
        {_field("installed_kwp", "Inštalovaný výkon (kWp)", "f.kwp", "Menovitý DC výkon.", value=pv.get("installed_kwp") or 400, kind="number")}
        {_field("tilt_deg", "Sklon (°)", "f.tilt", "Sklon poľa od vodorovnej roviny.", value=pv.get("tilt_deg") or 15, kind="number")}
        {_field("azimuth_deg", "Azimut (°)", "f.az", "180° je juh na severnej pologuli.", value=pv.get("azimuth_deg") or 180, kind="number")}
      </div>
      <h2>{labeled("Batéria", "Použiteľná energia, C-rate a počiatočné SOC, ak chýba telemetria BMS.", i18n="sec.bat", i18n_tip="sec.bat.tip")}</h2>
      <div class="panel grid-2">
        {_field("energy_kwh", "Energia batérie (kWh)", "f.kwh", "Použiteľná AC energia packu.", value=bat.get("energy_kwh") or bat.get("capacity_kWh") or 400, kind="number")}
        {_field("max_c_rate", "C-rate", "f.crate", "0,5 C pri 400 kWh je 200 kW.", value=bat.get("max_c_rate") or 0.5, kind="number")}
        {_field("initial_soc_pct", "Počiatočné SOC (%)", "f.soc", "Ak nie je snímka BMS. Telemetria ho pretlačí.", value=bat.get("initial_soc_pct") or 50, kind="number")}
      </div>
      <h2>{labeled("Pripojenie k sieti", "Limity importu, exportu a striedača v mieste pripojenia.", i18n="sec.grid", i18n_tip="sec.grid.tip")}</h2>
      <div class="panel grid-2">
        {_field("import_limit_kw", "Limit importu (kW)", "f.imp", "Maximálny odber zo siete.", value=grid.get("import_limit_kw") or 420, kind="number")}
        {_field("export_limit_kw", "Limit exportu (kW)", "f.exp", "Maximálny dodáv do siete.", value=grid.get("export_limit_kw") or 350, kind="number")}
        {_field("inverter_limit_kw", "Limit striedača (kW)", "f.inv", "AC limit striedača / meniča.", value=grid.get("inverter_limit_kw") or 400, kind="number")}
        {_field("contract_kw", "Zmluva MRK (kW)", "f.mrk", "Zmluvný rezervovaný výkon u prevádzkovateľa.", value=mrk.get("contract_kw") or 420, kind="number")}
      </div>
      <h2>{labeled("Dispatch a zmluva", "Ako sa počíta plán na D+1 a zmluva o rezervovanej kapacite.", i18n="sec.dispatch", i18n_tip="sec.dispatch.tip")}</h2>
      <div class="panel grid-2">
        {_field("dispatch_method", "Metóda dispatchu", "f.method", "auto skúsi LP (HiGHS) a pri zlyhaní greedy.", kind="select", extra=f'<select name="dispatch_method">{options}</select>')}
        {_field("degradation_cost_eur_per_kwh", "Náklad degradácie (€/kWh)", "f.deg", "Aby sa batéria netočíla pre maličké spready.", value=disp.get("degradation_cost_eur_per_kwh") or 0.01, kind="number")}
        {_field("ems_endpoint_url", "Endpoint EMS", "f.ems", "HTTP adresa EMS. Prázdne = súborový payload bez POST.", value=d.get("ems_endpoint_url") or "")}
      </div>
      <h2>{labeled("Zdroje dát", "Open-Meteo, SHMÚ a OKTE nahrádzajú komerčný zdroj žiarenia.", i18n="sec.sources", i18n_tip="sec.sources.tip")}</h2>
      <div class="panel">
        {_check("enable_open_meteo", "Open-Meteo", "f.om", "Verejný archív. Fyzikálny PVOUT z GHI, DNI, teploty a vetra.", checked=True)}
        {_check("enable_shmu", "Kalibrácia SHMÚ", "f.shmu", "Porovná modelové GHI s najbližšou stanicou SHMÚ.", checked=True)}
        {_check("enable_okte", "OKTE DAM", "f.okte", "Keď v odbere nie je cena a nie je zadaný samostatný CSV.", checked=True)}
      </div>
      <h2>{labeled("Beh", "Lokálne spínače. V Dominio DAG vždy trénuje odber a spúšťa dispatch.", i18n="sec.run", i18n_tip="sec.run.tip")}</h2>
      <div class="panel">
        {_check("quick", "Rýchly beh (posledných 7 dní loadu)", "f.quick", "Skráti odber na posledný týždeň, aby Open-Meteo a tréning stihli rýchlejšie.", checked=True)}
        {_check("skip_load", "Preskočiť tréning load forecastu", "f.skip_load", "Dispatch potom berie historickú krivku odberu. Odškrtnite pre XGB D+1 (pomalšie, doplní MAE spotreby).", checked=True)}
        {_check("skip_ops", "Preskočiť LP dispatch / EMS", "f.skip_ops", "Zastaví sa po inferencii FVE. Len metriky modelu, bez plánu batérie.", checked=False)}
        <p style="margin:18px 0 0"><button class="btn" type="submit" data-i18n="run">Spustiť SoMES</button></p>
      </div>
    </form>
    <script>
      document.getElementById("somes-form").addEventListener("submit", function () {{
        var b = this.querySelector("button[type=submit]");
        if (b) {{ b.disabled = true; b.textContent = (window.UC32_LANG === "en") ? "Running…" : "Počítam…"; }}
      }});
    </script>
    """
    try:
        from common_somes.ui_theme import page_shell
    except ModuleNotFoundError:
        from pieces.common_somes.ui_theme import page_shell

    en = dict(EN)
    en["sub"] = (
        "Set the site and plant, optionally upload your own load. "
        "Hover the question mark for the full explanation of each field."
    )
    return page_shell(
        title="Prevádzkový vstupný formulár",
        title_en="Operational input form",
        kicker="UC3.3 SoMES",
        sub="Vyplňte lokalitu a techniku, prípadne nahrajte vlastný odber. Na otáznik ukážte myšou — pri každej položke je vysvetlenie.",
        body=body,
        en=en,
    )


def render_progress_html(*, piece: str = "", n: int = 0, error: str = "", done: bool = False) -> str:
    try:
        from common_somes.ui_theme import page_shell
    except ModuleNotFoundError:
        from pieces.common_somes.ui_theme import page_shell

    if error:
        status = f'<div class="panel"><p class="value bad">{esc(error)}</p><p class="nav-back"><a href="/" data-i18n="back">← Vstupný formulár</a></p></div>'
        extra = ""
    elif done:
        status = '<div class="panel"><p data-i18n="done">Hotovo. Presmerovávam na dashboard…</p></div>'
        extra = '<script>window.location.replace("/dashboard");</script>'
    else:
        label = esc(piece or "…")
        status = (
            f'<div class="panel"><p data-i18n="progress">Počítam… Open-Meteo a XGBoost môžu trvať niekoľko minút.</p>'
            f'<p class="note-block">{label} ({int(n)})</p><progress max="24" value="{min(n, 24)}"></progress></div>'
        )
        extra = """<script>
        setTimeout(function () { window.location.reload(); }, 1200);
        </script>"""
    en = dict(EN)
    en["done"] = "Done. Redirecting to the dashboard…"
    en["progress"] = "Computing… Open-Meteo and XGBoost can take several minutes."
    return page_shell(
        title="SoMES beží",
        title_en="SoMES is running",
        kicker="UC3.3 SoMES",
        sub="Rovnaký beh ako v Dominio, len lokálne.",
        body=status,
        en=en,
        extra_js=extra,
    )
