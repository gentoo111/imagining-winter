"""Post E1 summary metrics to MLflow experiment last-embers-diffusion (one run per method x setting)."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from mlflow_live import _post, _get
EXP = "last-embers-diffusion"
try:
    eid = _get("experiments/get-by-name", {"experiment_name": EXP})["experiment"]["experiment_id"]
except Exception:
    eid = _post("experiments/create", {"name": EXP})["experiment_id"]
for setting in sys.argv[1:]:
    S = json.load(open(f"out/summary_{setting}.json"))
    for m, v in S.items():
        r = _post("runs/create", {"experiment_id": eid, "run_name": f"e1-{m}-{setting}", "start_time": int(time.time() * 1000),
                                  "tags": [{"key": "project", "value": "last-embers"}, {"key": "stage", "value": "E1"},
                                           {"key": "method", "value": m}, {"key": "setting", "value": setting}]})
        rid = r["run"]["info"]["run_id"]
        ts = int(time.time() * 1000)
        metrics = [{"key": k, "value": float(x[0]), "timestamp": ts, "step": 0} for k, x in v.items()]
        _post("runs/log-batch", {"run_id": rid, "metrics": metrics})
        _post("runs/update", {"run_id": rid, "status": "FINISHED", "end_time": ts})
        print(setting, m, rid)
