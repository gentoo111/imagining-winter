# S2-0: build the frozen Rust sim (lastembers_sim, commit 4195eb6) on Windows and check it bitwise
# against the main Mac's frozen .so. Everything lives in D:\LastEmbers\e1\sim (own rustup/cargo,
# own target dir, module installed to sim\site); the PhD pilot and the old D:\LastEmbers\{venv,
# rustup,cargo} are not touched. Source = `git archive 4195eb6 simrs data` unpacked to sim\src
# (with sim\src\COMMIT), never this repo's working tree. See docs/ml/windows/README.md (S2-0).
#
#   powershell -NoProfile -ExecutionPolicy Bypass -File build_sim.ps1 [-SkipBuild] [-Hours 1.5]
param([switch]$SkipBuild, [double]$Hours = 0, [string]$Notes = '')
$ErrorActionPreference = 'Stop'
$SIM = 'D:\LastEmbers\e1\sim'
$PY = 'D:\LastEmbers\e1\.venv\Scripts\python.exe'
$env:PYTHONUTF8 = '1'
$env:RUSTUP_HOME = "$SIM\rustup"
$env:CARGO_HOME = "$SIM\cargo"
$env:CARGO_TARGET_DIR = "$SIM\target"
$env:PATH = "$SIM\cargo\bin;" + $env:PATH
# pyo3-ffi's raw-dylib python3x.dll import makes rustc call `dlltool.exe` from PATH on the gnu target.
# sim\tools holds ONLY llvm-dlltool (as dlltool.exe) + the DLLs it needs, copied from the 10-28
# llvm-mingw. The llvm-mingw bin itself is never put on PATH: its gcc.exe would replace rustc's
# self-contained linker/CRT, and with it the libm that sin/cos/ln resolve to.
$MG = 'D:\LastEmbers\llvm-mingw\llvm-mingw-20261020-ucrt-x86_64\bin'
if (-not (Test-Path "$SIM\tools\dlltool.exe")) {
    New-Item -ItemType Directory -Force "$SIM\tools" | Out-Null
    Copy-Item "$MG\llvm-dlltool.exe" "$SIM\tools\dlltool.exe"
    Copy-Item "$MG\libLLVM-*.dll", "$MG\libc++.dll", "$MG\libunwind.dll", "$MG\libwinpthread-1.dll" "$SIM\tools\"
}
$env:PATH = "$SIM\tools;" + $env:PATH
$TOOLCHAIN = '1.99.1-x86_64-pc-windows-gnu'   # same rustc (58cf8f9c3) that built the main Mac's frozen .so
[System.Diagnostics.Process]::GetCurrentProcess().PriorityClass = 'BelowNormal'

function Wait-PhD {
    $said = $false
    while ((Get-ChildItem 'D:\PhD\pilot\queue\running', 'D:\PhD\pilot\queue\pending' -File -ErrorAction SilentlyContinue) -or
           (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'nh-run' })) {
        if (-not $said) { "WAIT phd queue busy"; $said = $true }
        Start-Sleep 60
    }
}

$commit = (Get-Content "$SIM\src\COMMIT" -Raw).Trim()
if ($commit -notmatch '^4195eb6') { throw "sim\src is not the frozen 4195eb6 source ($commit)" }

if (-not $SkipBuild) {
    Wait-PhD
    if (-not (Test-Path "$SIM\cargo\bin\rustup.exe")) {
        $ProgressPreference = 'SilentlyContinue'
        Invoke-WebRequest -UseBasicParsing 'https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-gnu/rustup-init.exe' -OutFile "$SIM\rustup-init.exe"
        & "$SIM\rustup-init.exe" -y --no-modify-path --profile minimal --default-host x86_64-pc-windows-gnu --default-toolchain $TOOLCHAIN
        if ($LASTEXITCODE) { throw "rustup-init exit $LASTEXITCODE" }
    }
    & rustup default $TOOLCHAIN
    uv pip install --python $PY 'maturin==1.15.0'
    if ($LASTEXITCODE) { throw "maturin install exit $LASTEXITCODE" }
    Remove-Item -Recurse -Force "$SIM\wheels" -ErrorAction SilentlyContinue
    & 'D:\LastEmbers\e1\.venv\Scripts\maturin.exe' build --release -m "$SIM\src\simrs\py\Cargo.toml" -i $PY --out "$SIM\wheels"
    if ($LASTEXITCODE) { throw "maturin build exit $LASTEXITCODE" }
    $whl = (Get-ChildItem "$SIM\wheels\*.whl" | Select-Object -First 1).FullName
    Remove-Item -Recurse -Force "$SIM\site" -ErrorAction SilentlyContinue
    uv pip install --python $PY --no-deps --target "$SIM\site" $whl
    if ($LASTEXITCODE) { throw "install exit $LASTEXITCODE" }
}

# Parity: trace the Windows build, compare with the main Mac's trace (scp'd to sim\trace_mac.json).
$rustc = (& rustc -V) -join ' '
# Optional wider sweep (seeds 61-1000, main-Mac trace scp'd to sim\trace_mac_ext.json) -> parity_extended.json.
if (Test-Path "$SIM\trace_mac_ext.json") {
    & $PY -B "$SIM\sim_trace.py" --site "$SIM\site" --data "$SIM\src\data" --out "$SIM\trace_win_ext.json" --seeds 61-1000
    if ($LASTEXITCODE) { throw "ext trace exit $LASTEXITCODE" }
    & $PY -B "$SIM\sim_parity.py" --ref "$SIM\trace_mac_ext.json" --test "$SIM\trace_win_ext.json" --out "$SIM\parity_extended.json" --commit $commit --toolchain $rustc
    "extended parity exit=$LASTEXITCODE"
}
& $PY -B "$SIM\sim_trace.py" --site "$SIM\site" --data "$SIM\src\data" --out "$SIM\trace_win.json" --seeds 1-60
if ($LASTEXITCODE) { throw "trace exit $LASTEXITCODE" }
& $PY -B "$SIM\sim_parity.py" --ref "$SIM\trace_mac.json" --test "$SIM\trace_win.json" --out "$SIM\parity.json" `
    --commit $commit --toolchain "$rustc; maturin 1.15.0; llvm-mingw 20261020 dlltool" --hours $Hours --notes $Notes
"parity exit=$LASTEXITCODE"
