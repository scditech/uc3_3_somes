from pydantic import BaseModel, Field


class InputModel(BaseModel):
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


class OutputModel(BaseModel):
    """
    Output model for Fetch Energy Data Piece
    """

    message: str = Field(default="")
    output_path: str = Field(default="")
