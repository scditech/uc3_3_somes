"""Create + run SoMES OneData customization in local Domino and report task states."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CUSTOM = Path(r"C:\Users\NTB\Desktop\uc33_somes_onedata.customization")
API = "http://localhost:8000"
EMAIL = "admin@email.com"
PASSWORD = "admin"
WORKSPACE_ID = 4
OUT_DIR = Path(r"C:\Users\NTB\Desktop\somes_domino_import_test")
# Must match the piece repository registered in workspace "test".
# Local Docker has this tag retargeted to the OneData-enabled overlay build.
SOURCE_IMAGE = "ghcr.io/scditech/uc3_3_somes:0.2.1-group0"


def api(method: str, path: str, token: str | None = None, body: dict | None = None):
    data = None if body is None else json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(API + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            return resp.status, (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")


def task_id(piece_name: str, node_id: str) -> str:
    return f"{piece_name[:10]}_{node_id.split('_', 1)[-1].replace('-', '')}"


def kwarg_from_spec(spec: dict) -> dict | None:
    if not isinstance(spec, dict):
        return {"fromUpstream": False, "upstreamTaskId": None, "upstreamArgument": None, "value": spec}
    from_up = bool(spec.get("fromUpstream"))
    up_id = spec.get("upstreamId") or None
    up_arg = spec.get("upstreamArgument") or None
    value = spec.get("value")
    if not from_up and value in ("", None) and not isinstance(value, (bool, int, float)):
        return None
    if from_up and not up_id and not up_arg:
        return None
    return {
        "fromUpstream": from_up,
        "upstreamTaskId": up_id if from_up else None,
        "upstreamArgument": up_arg if from_up else None,
        "value": value if not from_up else None,
    }


def build_payload(custom: dict, wf_name: str) -> dict:
    pieces = custom["workflowPieces"]
    pdata = custom["workflowPiecesData"]
    nodes = custom["workflowNodes"]
    edges = custom["workflowEdges"]

    node_piece = {nid: p["name"] for nid, p in pieces.items()}
    node_tid = {nid: task_id(name, nid) for nid, name in node_piece.items()}
    local_storage = {"source": "Local", "mode": "Read/Write", "provider_options": {}}

    tasks = {}
    for nid, piece in pieces.items():
        name = piece["name"]
        tid = node_tid[nid]
        inputs = {}
        for key, spec in (pdata[nid].get("inputs") or {}).items():
            mapped = kwarg_from_spec(spec)
            if mapped is not None:
                inputs[key] = mapped
        deps = sorted(
            {
                mapped["upstreamTaskId"]
                for mapped in inputs.values()
                if mapped.get("fromUpstream") and mapped.get("upstreamTaskId")
            }
        )
        cr = pdata[nid].get("containerResources") or {}
        cpu = cr.get("cpu", 500)
        mem = cr.get("memory", 2048)
        if isinstance(cpu, dict):
            cpu_req, cpu_lim = cpu.get("min", 100), cpu.get("max", 1000)
        else:
            cpu_req = cpu_lim = int(cpu)
        if isinstance(mem, dict):
            mem_req, mem_lim = mem.get("min", 128), mem.get("max", 4096)
        else:
            mem_req = mem_lim = int(mem)
        tasks[tid] = {
            "task_id": tid,
            "piece": {"name": name, "source_image": SOURCE_IMAGE},
            "dependencies": deps,
            "piece_input_kwargs": inputs,
            "workflow_shared_storage": local_storage,
            "container_resources": {
                "requests": {"cpu": float(cpu_req), "memory": float(mem_req)},
                "limits": {"cpu": float(cpu_lim), "memory": float(mem_lim)},
                "use_gpu": bool(cr.get("useGpu", False)),
            },
        }

    ui_nodes = {node_tid[node["id"]]: node for node in nodes}
    ui_edges = []
    for e in edges:
        src = node_tid[e["source"]]
        dst = node_tid[e["target"]]
        ui_edges.append(
            {
                "source": src,
                "sourceHandle": f"source-{src}",
                "target": dst,
                "targetHandle": f"target-{dst}",
                "id": f"e-{src}-{dst}",
                "markerEnd": e.get("markerEnd") or {"type": "arrowclosed", "width": 20, "height": 20},
            }
        )

    return {
        "workflow": {
            "name": wf_name,
            "startDateTime": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
            "scheduleInterval": "none",
            "selectStartDate": "now",
            "selectEndDate": "never",
            "catchup": False,
            "generateReport": False,
            "description": "SoMES OneData verification run",
        },
        "tasks": tasks,
        "ui_schema": {"nodes": ui_nodes, "edges": ui_edges},
        "forageSchema": custom,
    }


def airflow_task_states(dag_id: str, run_id: str) -> list[tuple[str, str]]:
    import subprocess

    sql = (
        "SELECT task_id, state FROM task_instance "
        f"WHERE dag_id='{dag_id}' AND run_id='{run_id}' ORDER BY task_id;"
    )
    out = subprocess.check_output(
        ["docker", "exec", "airflow-postgres", "psql", "-U", "airflow", "-d", "airflow", "-t", "-A", "-F", "|", "-c", sql],
        text=True,
        errors="replace",
    )
    rows = []
    for line in out.splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        tid, state = line.split("|", 1)
        rows.append((tid, state))
    return rows


def main() -> int:
    custom = json.loads(CUSTOM.read_text(encoding="utf-8"))
    code, login = api("POST", "/auth/login", body={"email": EMAIL, "password": PASSWORD})
    if code != 200:
        print("LOGIN_FAIL", code, login)
        return 1
    token = login["access_token"]

    wf_name = f"SoMESOneData{datetime.now(timezone.utc).strftime('%m%d%H%M')}"
    payload = build_payload(custom, wf_name)
    (OUT_DIR / "onedata_create_payload.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    code, resp = api("POST", f"/workspaces/{WORKSPACE_ID}/workflows", token=token, body=payload)
    (OUT_DIR / "onedata_create_result.json").write_text(
        json.dumps({"code": code, "resp": resp}, indent=2, default=str), encoding="utf-8"
    )
    print("CREATE", code, wf_name)
    if code not in (200, 201) or not isinstance(resp, dict):
        print(str(resp)[:2000])
        return 2
    wf_id = resp["id"]
    print("workflow_id", wf_id)

    # Wait until creation settles, then trigger
    run = None
    for i in range(30):
        time.sleep(5)
        c, r = api("POST", f"/workspaces/{WORKSPACE_ID}/workflows/{wf_id}/runs", token=token, body={})
        print("TRIGGER", i, c, str(r)[:120])
        if c in (200, 201, 202):
            break
        c2, runs = api("GET", f"/workspaces/{WORKSPACE_ID}/workflows/{wf_id}/runs", token=token)
        data = (runs or {}).get("data") if isinstance(runs, dict) else None
        if data:
            run = data[0]
            print("auto-run appeared", run.get("state"), run.get("workflow_run_id"))
            break

    # Poll
    for i in range(90):
        time.sleep(10)
        c, runs = api("GET", f"/workspaces/{WORKSPACE_ID}/workflows/{wf_id}/runs", token=token)
        data = (runs or {}).get("data") if isinstance(runs, dict) else None
        if not data:
            print(i, "no runs")
            continue
        run = data[0]
        state = run.get("state")
        run_id = run.get("workflow_run_id")
        dag = run.get("workflow_uuid")
        print(i, "state", state, "run", run_id)
        if state in ("success", "failed"):
            rows = airflow_task_states(dag, run_id) if dag and run_id else []
            summary = {"workflow_id": wf_id, "name": wf_name, "run": run, "tasks": rows}
            (OUT_DIR / "onedata_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("==== TASKS ====")
            for tid, st in rows:
                print(f"{st:20} {tid}")
            failed = [t for t, s in rows if s == "failed"]
            ok = [t for t, s in rows if s == "success"]
            print(f"success={len(ok)} failed={len(failed)} total={len(rows)}")
            return 0 if state == "success" else 3
    print("TIMEOUT")
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
