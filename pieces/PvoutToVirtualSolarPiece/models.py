from pydantic import BaseModel, Field


class InputModel(BaseModel):
    forecast_csv_path: str = Field(
        description="Path to UC3.4 InferencePiece forecast CSV (final_forecast / PVOUT)."
    )


class OutputModel(BaseModel):
    message: str
    virtual_solar_csv: str = Field(description="SoMES virtual_solar.csv for BatterySimPiece")
