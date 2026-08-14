Import: uc33_somes.customization (OneData) — UC3.3 SoMES + UC3.4 PVOUT

Space: SCDI
Inputs:  onedata:///SCDI/UC3.3_SOMES/inputs/
Outputs: onedata:///SCDI/UC3.3_SOMES/outputs/

PV data: Open-Meteo by user latitude/longitude (no commercial SolarGIS files).
Chain: Open-Meteo -> preprocess -> train -> error-correction -> staged inference
       -> evaluate/explain/aggregate -> virtual_solar -> BatterySim (+ SoMES ops).

Domino Settings before Create:
- Storage Source = Local
- Name: letters/numbers/_ only
- Piece image: ghcr.io/scditech/uc3_3_somes:0.2.3-group0

Also available as: uc33_somes_onedata.customization / uc33_somes.json (same content).
