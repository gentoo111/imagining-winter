# Imagining Winter: code and data

Code, data generator, particle-filter oracle, forecasters and evaluation scripts for

> Zhenzhou (Luke) Shi. *Imagining Winter: Generative Storm Forecasts and the Value
> of Foresight in a Chaotic Survival Game.* arXiv preprint, 2026.

- Code: https://github.com/gentoo111/imagining-winter (this repository), MIT License (`LICENSE`).
- Data, forecasts and model checkpoints: release assets of
  https://github.com/gentoo111/imagining-winter/releases/tag/v1.0, CC BY 4.0 (`LICENSE-DATA.md`);
  checksums in `SHA256SUMS` on the release page.
- `RELEASE_COMMIT`: the commit of the research repository this tree was exported from.

## Frozen simulator

All results use the simulator of game commit `4195eb6` (tag `paper-e1-frozen`; one chaotic weather).
`simrs/` and `data/` in this repository are that commit's files; the only addition is
`simrs/core/src/bin/weather_gen.rs` (weather-only data export, with the optional offline wind-noise
switch of the observation-model ablation). The Godot game itself is not part of this release.

## Layout

| Path | Contents |
|---|---|
| `simrs/core` | Rust simulation core (bit-exact port of the game simulation): Lorenz-63 weather in Q16 fixed point, precursors, storm rule, colony simulation used for RL |
| `simrs/core/src/bin/weather_gen.rs` | weather-only data for the forecasting experiments |
| `simrs/py` | Python bindings of the simulator (PyO3, `maturin`) used by PPO |
| `data/*.json` | game data read by the simulator (balance, buildings, expeditions, technology) |
| `ml/forecast/pf.py` | bootstrap particle-filter oracle on the exact integer dynamics; `pf_selfcheck.py` PIT/rank self-check |
| `ml/forecast/baselines.py` | climatology, clock hazard and scripted-warning baselines |
| `ml/forecast/models.py`, `train.py` | GRU landing-time classifier and the conditional diffusion forecasters (v2 U-Net, v3 transformer) |
| `ml/forecast/evaluate.py`, `metrics.py`, `path_baselines.py` | evaluation: Brier skill, CRPS, log loss, W1, KL, energy and variogram scores, analogue path generators |
| `ml/forecast/value_curve.py`, `figures_paper.py` | relative economic value and the paper's figures and generated tables |
| `ml/forecast/windows/` | GPU/CPU job scripts used on the training host (200-seed evaluation, ablation, learning curve, v3 selection and seeds, sample count, wind noise, rungs 3b/3a′) |
| `ml/env.py`, `ml/dppo.py`, `ml/train.py` | PPO environment wrapper and distributed PPO used for RQ2; `ml/forecast/storm_e0.py`, `storm_seeds.py`, `probe_e0.py` evaluate the ladder |
| `docs/paper/` | LaTeX source of the paper |

## Reproducing

Forecasting data (Rust, about 0.15 s for 500 seeds × 120 days):

```sh
cd simrs && cargo build --release -p lastembers_core --bin weather_gen
# weather_gen <data dir> <out> <seed_from> <n> <days> <profile> <noise> [wind_noise=<p>]
./target/release/weather_gen ../data ../ml/forecast/data/test_endurance_murky.bin 220000 500 120 endurance murky
```

Seed splits: forecaster training 200000–203999, validation 210000–210199, test 220000–220499;
RL evaluation 300000–300999. Then, in `ml/forecast/`:

```sh
TORCH_THREADS=2 python train.py gru endurance murky 16000 runs/gru16k_endurance_murky.pt
DEVICE=cuda VAL_EVERY=2000 python train.py diff endurance murky 100000 runs/diff2_endurance_murky.pt          # v2 (U-Net)
DEVICE=cuda V3_CFG=C LR=1e-3 TF32=1 SEED=1 VAL_EVERY=1000 python train.py diff endurance murky 100000 runs/diff3_endurance_murky_s1.pt
python value_curve.py && python figures_paper.py tables200
```

The 200-seed evaluation (200 paths, DDIM 50 for the main results), the ablation, the learning curve,
v3 selection and seeds, the sample-count study and the wind-noise ablation were run by the job scripts in
`ml/forecast/windows/` (`jobs/*.cmd`, `condor/*`), which record the exact commands and environment
variables; the `*.pt.log` files in `e1-checkpoints.tar` record each training run.

## Release assets (v1.0)

| Asset | Contents |
|---|---|
| `e1-checkpoints.tar` | GRU (three settings), diffusion v2 (12k-step ablation model, best learning-curve checkpoint at step 88k, ablation models) and v3 (three training seeds, selection run, marginal control, secondary-setting and wind-noise seed 1) checkpoints with training logs; PPO policies of RQ2 (`ml/forecast/out/release_ckpt/ppo/`: stage-11 start, final model of every rung × three runs, last checkpoints of run 1 for the drift analysis) |
| `e1-forecasts-eval200.tar` | per-forecast outputs of every method on the 200 test seeds × 113 issue times (three settings) and their summaries |
| `e1-results.tar` | result JSON behind every table and figure (RQ1 and RQ2) |

In the JSON and log files of the assets, absolute home-directory prefixes of provenance paths (e.g. the checkpoint each RQ2 evaluation loaded) are written as `~/`; nothing else is changed.

## Citation

```bibtex
@misc{shi2026imagining,
  author = {Shi, Zhenzhou},
  title  = {Imagining Winter: Generative Storm Forecasts and the Value of Foresight in a Chaotic Survival Game},
  year   = {2026},
  note   = {arXiv preprint}
}
```
