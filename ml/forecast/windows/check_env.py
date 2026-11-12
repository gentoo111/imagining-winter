"""Windows e1 environment report (S1-1a). Read-only: writes D:\\LastEmbers\\e1\\setup.json.
Run with the e1 venv:  D:\\LastEmbers\\e1\\.venv\\Scripts\\python.exe ml\\forecast\\windows\\check_env.py"""
import json, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
E1 = os.path.abspath(os.path.join(HERE, "..", "..", ".."))        # D:\LastEmbers\e1
DATA = os.path.join(E1, "ml", "forecast", "data")


def procs():
    """Command lines of all processes (PowerShell CIM; tasklist has no command line)."""
    ps = "Get-CimInstance Win32_Process | ForEach-Object { $_.Name + '|' + $_.CommandLine }"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def size(p):
    return os.path.getsize(p) if os.path.exists(p) else 0


import torch  # noqa: E402

pl = procs()
rep = {
    "torch": torch.__version__,
    "cuda_available": bool(torch.cuda.is_available()),
    "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "cuda_mem_mib": (torch.cuda.get_device_properties(0).total_memory >> 20) if torch.cuda.is_available() else 0,
    "python": sys.version.split()[0],
    "venv_path": sys.prefix,
    "numpy": __import__("numpy").__version__,
    "balance_json_bytes": size(os.path.join(E1, "data", "balance.json")),
    "forecast_py_files": len([f for f in os.listdir(os.path.join(E1, "ml", "forecast")) if f.endswith(".py")]),
}
for sp in ("train", "val", "test"):
    rep[f"endurance_murky_{sp}_bytes"] = size(os.path.join(DATA, f"endurance_murky_{sp}.bin"))
rep["dppo_workers"] = sum(1 for ln in pl if ln.lower().startswith("python") and "dppo.py worker" in ln)
rep["phd_smoke_running"] = any("smoke.yml" in ln for ln in pl)
rep["checked"] = time.strftime("%Y-%m-%d %H:%M:%S")
out = os.path.join(E1, "setup.json")
with open(out, "w") as f:
    json.dump(rep, f, indent=1)
print(json.dumps(rep, indent=1))
