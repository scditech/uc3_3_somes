# UC3.3 SoMES — Domino pieces

Operational next-day microgrid optimisation (load / PV / prices / BESS dispatch) for Domino, with **OneData** I/O.

## Register pieces repository

1. In Domino: Pieces repositories → add GitHub repo `scditech/uc3_3_somes`
2. CI builds and publishes `ghcr.io/scditech/uc3_3_somes:0.2.1-group0`

## Import workflow into the editor

1. Open **Workflow editor** (workspace where this repo is registered)
2. Click **Import → from file**
3. Upload [`domino_import/uc33_somes.customization`](domino_import/uc33_somes.customization)  
   (OneData variant is the same file; also available as `uc33_somes_onedata.customization`)
4. **Settings**:
   - Name: letters/numbers/`_` only (e.g. `SoMESOneData`)
   - **Storage Source = Local** (Domino results mount; inputs/outputs still use OneData URLs)
5. Create → Run

The import is the full 18-node SoMES DAG (D+1 load forecast, Open-Meteo / OKTE connectors, LP dispatch, technical validation, EMS output, dashboard).

## OneData layout (space `SCDI`)

| Role | Path |
|------|------|
| Inputs | `onedata:///SCDI/UC3.3_SOMES/inputs/` |
| Outputs | `onedata:///SCDI/UC3.3_SOMES/outputs/<run_id>/<PieceName>/` |

Seed inputs (from a machine with OneData access):

```bash
python scripts/seed_somes_onedata.py
```

Expected under `inputs/`: `load_input/`, `prices.csv`, `history_by_department.csv`, `predict_in/load.csv`, `measured_last_day.csv`, `scenario.yaml`, `solargis_irradiance.csv`, `weather_history.csv`, `connectors/`, `model_registry/`.

OneData token defaults live in `pieces/common/onedata_defaults.py` (same pattern as UC3.2). Override via Domino piece secrets if needed.

## Local Domino overlay image (optional)

```bash
docker build -f dependencies/Dockerfile.somes-local -t somes-local:0.2.1 .
```
