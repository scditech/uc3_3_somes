from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SecretsModel(BaseModel):
    onedata_onezone_host: Optional[str] = Field(
        default=None,
        description="Onedata Onezone host.",
    )
    onedata_token: Optional[str] = Field(
        default=None,
        description="Onedata access token.",
    )


class InputModel(BaseModel):
    pass


class OutputModel(BaseModel):
    pass
