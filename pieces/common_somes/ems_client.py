"""EMS/BEMS delivery adapter — authenticated REST push with retry and acknowledgement."""
from __future__ import annotations

import base64
import json
import random
import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from urllib import error as urlerror
from urllib import request as urlrequest

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


def build_auth_headers(
    *,
    auth_mode: str = "none",
    token: str = "",
    username: str = "",
    password: str = "",
    api_key_header: str = "X-API-Key",
) -> dict[str, str]:
    mode = (auth_mode or "none").lower()
    if mode == "bearer" and token:
        return {"Authorization": f"Bearer {token}"}
    if mode == "basic" and username:
        raw = f"{username}:{password}".encode("utf-8")
        return {"Authorization": f"Basic {base64.b64encode(raw).decode('ascii')}"}
    if mode == "api_key" and token:
        return {api_key_header: token}
    return {}


def interpret_ack(status: int, body: str) -> dict[str, Any]:
    """Decide whether the EMS actually accepted the schedule, not just the HTTP call."""
    ack: dict[str, Any] = {"http_status": status, "accepted": 200 <= status < 300, "raw": body[:2000]}
    try:
        parsed = json.loads(body) if body.strip() else {}
    except json.JSONDecodeError:
        ack["ack_format"] = "non_json"
        return ack
    ack["ack_format"] = "json"
    ack["payload"] = parsed
    if isinstance(parsed, dict):
        for key in ("accepted", "success", "ok"):
            if isinstance(parsed.get(key), bool):
                ack["accepted"] = ack["accepted"] and parsed[key]
        status_field = str(parsed.get("status", "")).lower()
        if status_field in {"rejected", "error", "failed"}:
            ack["accepted"] = False
        ack["ems_reference"] = parsed.get("id") or parsed.get("reference") or parsed.get("request_id")
        ack["rejected_instructions"] = parsed.get("rejected") or parsed.get("errors")
    return ack


def post_schedule(
    url: str,
    payload: dict[str, Any],
    *,
    auth_mode: str = "none",
    token: str = "",
    username: str = "",
    password: str = "",
    api_key_header: str = "X-API-Key",
    timeout: int = 15,
    max_retries: int = 3,
    backoff_seconds: float = 1.5,
    sleep=time.sleep,
) -> dict[str, Any]:
    """POST with exponential backoff + jitter. Returns a full delivery record."""
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    headers.update(
        build_auth_headers(
            auth_mode=auth_mode,
            token=token,
            username=username,
            password=password,
            api_key_header=api_key_header,
        )
    )
    body = json.dumps(payload).encode("utf-8")
    attempts: list[dict[str, Any]] = []

    for attempt in range(1, max(1, max_retries) + 1):
        started = time.monotonic()
        try:
            req = urlrequest.Request(url, data=body, headers=headers, method="POST")
            with urlrequest.urlopen(req, timeout=timeout) as resp:
                text = resp.read().decode("utf-8", errors="replace")
                ack = interpret_ack(resp.status, text)
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": ack["http_status"],
                    "latency_ms": round((time.monotonic() - started) * 1000, 1),
                    "accepted": ack["accepted"],
                }
            )
            if ack["accepted"] or ack["http_status"] not in RETRYABLE_STATUS:
                return {
                    "delivered": bool(ack["accepted"]),
                    "endpoint": url,
                    "auth_mode": auth_mode,
                    "attempts": attempts,
                    "acknowledgement": ack,
                    "delivered_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            error = f"retryable HTTP {ack['http_status']}"
        except urlerror.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else ""
            ack = interpret_ack(exc.code, text)
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": exc.code,
                    "latency_ms": round((time.monotonic() - started) * 1000, 1),
                    "accepted": False,
                }
            )
            if exc.code not in RETRYABLE_STATUS:
                return {
                    "delivered": False,
                    "endpoint": url,
                    "auth_mode": auth_mode,
                    "attempts": attempts,
                    "acknowledgement": ack,
                    "error": f"HTTP {exc.code}",
                    "delivered_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            error = f"retryable HTTP {exc.code}"
        except (urlerror.URLError, TimeoutError, OSError) as exc:
            attempts.append(
                {
                    "attempt": attempt,
                    "http_status": None,
                    "latency_ms": round((time.monotonic() - started) * 1000, 1),
                    "accepted": False,
                    "error": str(exc),
                }
            )
            error = str(exc)

        if attempt < max_retries:
            sleep(backoff_seconds * (2 ** (attempt - 1)) + random.uniform(0, 0.25))

    return {
        "delivered": False,
        "endpoint": url,
        "auth_mode": auth_mode,
        "attempts": attempts,
        "acknowledgement": None,
        "error": error,
        "delivered_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_register_map(schedule: pd.DataFrame, *, start_register: int = 40001) -> pd.DataFrame:
    """Point/register map so a Modbus or IEC-104 gateway can consume the same plan."""
    points = [
        ("battery_setpoint_kw", "holding", "kW", 0.1),
        ("soc_target_pct", "holding", "%", 0.1),
        ("grid_import_kw", "holding", "kW", 0.1),
        ("grid_export_kw", "holding", "kW", 0.1),
    ]
    rows: list[dict[str, Any]] = []
    register = start_register
    for name, kind, unit, scale in points:
        if name not in schedule.columns:
            continue
        rows.append(
            {
                "signal": name,
                "register": register,
                "register_type": kind,
                "data_type": "int16",
                "unit": unit,
                "scale_factor": scale,
                "min_value": round(float(pd.to_numeric(schedule[name], errors="coerce").min()), 3),
                "max_value": round(float(pd.to_numeric(schedule[name], errors="coerce").max()), 3),
                "steps": int(len(schedule)),
            }
        )
        register += 1
    rows.append(
        {
            "signal": "schedule_valid",
            "register": register,
            "register_type": "coil",
            "data_type": "bool",
            "unit": "-",
            "scale_factor": 1,
            "min_value": 0,
            "max_value": 1,
            "steps": 1,
        }
    )
    return pd.DataFrame(rows)
