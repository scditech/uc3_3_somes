from pydantic import BaseModel, Field


class InputModel(BaseModel):
    next_day_dispatch_plan_csv: str = Field(description="Approved next-day dispatch plan CSV")
    technical_validation_json: str = Field(description="Technical validation bundle JSON")
    operational_kpis_json: str = Field(default="", description="Optional operational KPIs JSON")
    flexible_load_activation_csv: str = Field(default="", description="Optional flexible load activation CSV")
    ems_endpoint_url: str = Field(
        default="",
        description="HTTP endpoint for EMS/BEMS POST (file + register map always written)",
    )
    auth_mode: str = Field(default="none", description="none | bearer | basic | api_key")
    auth_token: str = Field(default="", description="Bearer token or API key value")
    auth_username: str = Field(default="", description="Username for basic auth")
    auth_password: str = Field(default="", description="Password for basic auth")
    api_key_header: str = Field(default="X-API-Key", description="Header name used in api_key mode")
    request_timeout_s: int = Field(default=15, description="Per-attempt HTTP timeout")
    max_retries: int = Field(default=3, description="Delivery attempts before giving up")
    backoff_seconds: float = Field(default=1.5, description="Base for exponential backoff between attempts")
    require_delivery: bool = Field(
        default=False, description="Fail the piece when an endpoint is configured but delivery is not acknowledged"
    )
    emit_register_map: bool = Field(default=True, description="Write a Modbus/IEC-104 point map next to the payload")


class OutputModel(BaseModel):
    message: str
    ems_bems_payload_json: str
    ems_bems_schedule_csv: str
    ems_bems_ack_json: str = ""
    ems_bems_register_map_csv: str = ""
    delivered: bool = False
