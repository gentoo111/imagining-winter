# S1-1a: idempotent setup of the Last Embers e1 environment on Windows (D:\LastEmbers\e1).
# Separate from the PhD pilot: own uv venv, own data, own queue. Sources and data are
# scp'd from the main Mac beforehand (see docs/ml/windows/README.md).
$ErrorActionPreference = 'Stop'
$E1 = 'D:\LastEmbers\e1'
foreach ($d in 'ml\forecast\data', 'ml\forecast\runs', 'ml\forecast\out', 'data', 'out') { New-Item -ItemType Directory -Force "$E1\$d" | Out-Null }
$py = "$E1\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { uv venv "$E1\.venv" --python 3.12 }
# Same torch build as the PhD pilot uses (2.14.1+cu132) so uv reuses its cached wheel.
uv pip install --python $py 'torch==2.14.1+cu132' numpy --index-url https://download.pytorch.org/whl/cu132 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
& $py "$E1\ml\forecast\windows\check_env.py"
