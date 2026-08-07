"""SoMES Output API / EMS-BEMS integration interface (module 17)."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import pandas as pd

try:
    from domino.base_piece import BasePiece
except ModuleNotFoundError:
    from local_compat.base_piece import BasePiece

from .models import InputModel, OutputModel


class EmsBemsOutputPiece(BasePiece):
    def piece_function(self, input_data: InputModel) -> OutputModel:
        repo_root = Path(__file__).resolve().parents[2]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        from pieces.common_somes.architecture import build_ems_bems_payload, write_json
        from pieces.common_somes.ems_client import build_register_map, post_schedule

        out_dir = Path(self.results_path or ".")
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "ems_bems.log"

        def _log(msg: str) -> None:
            text = f"[EmsBemsOutputPiece] {msg}"
            print(text, flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

        try:
            plan = pd.read_csv(input_data.next_day_dispatch_plan_csv, parse_dates=["datetime"])
            validation = json.loads(Path(input_data.technical_validation_json).read_text(encoding="utf-8"))
            kpis = {}
            if input_data.operational_kpis_json and Path(input_data.operational_kpis_json).is_file():
                kpis = json.loads(Path(input_data.operational_kpis_json).read_text(encoding="utf-8"))
            flex = None
            if input_data.flexible_load_activation_csv and Path(input_data.flexible_load_activation_csv).is_file():
                flex = pd.read_csv(input_data.flexible_load_activation_csv)

            payload = build_ems_bems_payload(
                next_day_plan=plan,
                technical_validation=validation,
                operational_kpis=kpis,
                flexible_schedule=flex,
            )
            out_json = write_json(out_dir / "ems_bems_payload.json", payload)
            # machine-readable schedule (one row per instruction)
            sched = pd.DataFrame(payload["dispatch_instructions"])
            out_csv = out_dir / "ems_bems_schedule.csv"
            sched.to_csv(out_csv, index=False)

            register_path = ""
            if input_data.emit_register_map:
                register_map = build_register_map(sched)
                register_path = str(out_dir / "ems_bems_register_map.csv")
                register_map.to_csv(register_path, index=False)

            if not input_data.ems_endpoint_url:
                delivery = {
                    "delivered": False,
                    "status": "no_endpoint_configured",
                    "note": "payload, schedule and register map are available for pull-based integration",
                }
            elif not payload.get("approved"):
                delivery = {
                    "delivered": False,
                    "status": "blocked_not_approved",
                    "note": "technical validation rejected the plan; nothing was sent to the EMS",
                }
                _log("Delivery blocked: plan is not approved by technical validation")
            else:
                delivery = post_schedule(
                    input_data.ems_endpoint_url,
                    payload,
                    auth_mode=input_data.auth_mode,
                    token=input_data.auth_token,
                    username=input_data.auth_username,
                    password=input_data.auth_password,
                    api_key_header=input_data.api_key_header,
                    timeout=int(input_data.request_timeout_s),
                    max_retries=int(input_data.max_retries),
                    backoff_seconds=float(input_data.backoff_seconds),
                )
                _log(
                    f"Delivery attempts={len(delivery.get('attempts', []))} "
                    f"delivered={delivery.get('delivered')} error={delivery.get('error')}"
                )

            ack_path = write_json(out_dir / "ems_bems_ack.json", delivery)
            delivered = bool(delivery.get("delivered"))
            if input_data.require_delivery and input_data.ems_endpoint_url and not delivered:
                raise RuntimeError(f"EMS delivery not acknowledged: {delivery.get('error') or delivery.get('status')}")

            status = delivery.get("status") or ("acknowledged" if delivered else "not_acknowledged")
            _log(f"EMS/BEMS payload written, delivery={status}")
            return OutputModel(
                message=f"EMS/BEMS interface ready ({status})",
                ems_bems_payload_json=str(out_json),
                ems_bems_schedule_csv=str(out_csv),
                ems_bems_ack_json=str(ack_path),
                ems_bems_register_map_csv=register_path,
                delivered=delivered,
            )
        except Exception as exc:
            (out_dir / "ems_bems_error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            _log(f"ERROR: {exc}")
            raise
