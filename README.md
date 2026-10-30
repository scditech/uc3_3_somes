# UC3.3 SoMES — Domino pieces

Operational next-day microgrid optimisation (load / PV / prices / BESS dispatch)
for Domino, with **OneData** I/O.

Current import is a **28-node** DAG: load forecast, Open-Meteo PVOUT chain,
connectors, LP dispatch, grid feasibility, EMS/BEMS, anomaly/drift alerts,
dashboard.

## User parameters (FVE + battery)

There is no form piece. Site / PV / battery come from **`scenario.yaml`**:

`onedata:///SCDI/UC3.3_SOMES/inputs/scenario.yaml`

| Block | Typical fields |
|-------|----------------|
| `pv` | `installed_kwp`, `tilt_deg`, `azimuth_deg` |
| `battery` | `energy_kwh` / `capacity_kWh`, C-rate, efficiencies, initial SoC |
| `site` | `latitude`, `longitude` |
| `grid` | import/export/inverter/connection limits |

Edit that file in OneData and re-run the workflow. Piece inputs in the Domino
editor can still override fallbacks (e.g. Open-Meteo lat/lon/kWp).

## Register pieces repository

1. In Domino: Pieces repositories → add GitHub repo `scditech/uc3_3_somes`
2. CI builds and publishes `ghcr.io/scditech/uc3_3_somes:<VERSION>-group0`

## Import workflow

1. Open **Workflow editor** (workspace where this repo is registered)
2. Import → from file:
   - [`domino_import/uc33_somes.customization`](domino_import/uc33_somes.customization) — OneData paths, token-free
   - [`domino_import/uc33_somes.spice.customization`](domino_import/uc33_somes.spice.customization) — same DAG with GHCR `source_image`
3. **Settings**: name letters/numbers/`_` only; **Storage Source = Local**
   (results mount; inputs/outputs still use OneData URLs)
4. Create → Run

## OneData layout (space `SCDI`)

| Role | Path |
|------|------|
| Inputs | `onedata:///SCDI/UC3.3_SOMES/inputs/` |
| Outputs | `onedata:///SCDI/UC3.3_SOMES/outputs/<run_id>/<PieceName>/` |

Expected under `inputs/`: `load_input/`, `prices.csv`, `history_by_department.csv`,
`predict_in/load.csv`, `measured_last_day.csv`, `scenario.yaml`,
`solargis_irradiance.csv`, `weather_history.csv`, `connectors/`, `model_registry/`.

**Do not put an OneData token in code.** Set `onedata_token` as a Domino/SPICE
repository secret (`pieces/common/onedata_defaults.py` default is empty).

## Local Domino overlay image

```bash
docker build -f dependencies/Dockerfile.somes-local -t uc33-somes-local:latest .
```
