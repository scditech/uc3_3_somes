# UC3.3 SoMES — Domino pieces

Operational next-day microgrid optimisation (load / PV / prices / BESS dispatch) for Domino.

## Register pieces repository

1. In Domino: Pieces repositories → add GitHub repo `scditech/uc3_3_somes`
2. CI builds and publishes `ghcr.io/scditech/uc3_3_somes:0.2.0-group0`

## Import workflow into the editor

1. Open **Workflow editor**
2. Click **Import → from file**
3. Upload [`domino_import/uc33_somes.customization`](domino_import/uc33_somes.customization)

The import contains the full 18-node SoMES DAG (D+1 load forecast, Open-Meteo / OKTE connectors, LP dispatch, technical validation, EMS output, dashboard).

## Runtime inputs (shared storage)

Default path used by the imported nodes: `/home/shared_storage/somes/`

- `load_input/load.csv`, `prices.csv`, `history_by_department.csv`
- `predict_in/load.csv`, `measured_last_day.csv`
- `scenario.yaml`, `solargis_irradiance.csv`, `weather_history.csv`
- `connectors/` — PV measurements, BESS telemetry, grid constraints, flexible loads
