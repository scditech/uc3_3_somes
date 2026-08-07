from pydantic import BaseModel, Field

try:
    from common.onedata_models import OneDataSecretsModel, RunIdInputMixin
except ModuleNotFoundError:
    from pieces.common.onedata_models import OneDataSecretsModel, RunIdInputMixin



class InputModel(RunIdInputMixin):
    """
    Input model for Fetch Energy Data Piece
    """

    load_csv: str = Field(
        default="/home/shared_storage/load.csv",
        description="Path to load CSV file or a directory with load*.csv files"
    )

    prices_csv: str = Field(
        default="/home/shared_storage/prices.csv",
        description="Path to prices CSV file"
    )


class SecretsModel(OneDataSecretsModel):
    pass


class OutputModel(BaseModel):
    run_id: str = Field(default="", description="OneData run folder id")
    """
    Output model for Fetch Energy Data Piece
    """

    message: str = Field(default="")
    output_path: str = Field(default="")
