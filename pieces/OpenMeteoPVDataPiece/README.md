# OpenMeteoPVDataPiece

Generate PV irradiance / PVOUT time series from the [Open-Meteo archive API](https://open-meteo.com/en/docs/historical-weather-api) (no API key). Outputs match the column layout expected by **DataPreprocessingPiece**.

## Typical wiring

```
OpenMeteoPVDataPiece.file_path
  → DataPreprocessingPiece.data_path
```

Site latitude/longitude/kWp/tilt should come from `scenario_yaml` (OneData).
