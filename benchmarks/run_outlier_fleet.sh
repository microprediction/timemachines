#!/bin/zsh
# Gentle, resumable outlier-benchmark fleet (see RESULTS.md §8).
#
# Every harness checkpoints per series and skips cached work, so this script
# can be killed and re-run at any time and only pays for what is missing.
# Stages run SEQUENTIALLY at low priority — with WORKERS=2 the whole fleet
# never occupies more than ~2 cores. Raise WORKERS when the machine is free:
#     WORKERS=8 ./benchmarks/run_outlier_fleet.sh
#
# Do not launch with nohup on this machine (SIP strips DYLD_* for protected
# binaries); use `./benchmarks/run_outlier_fleet.sh > fleet.log 2>&1 &`.

cd "$(dirname "$0")/.."
WORKERS=${WORKERS:-2}
PY=.venv/bin/python
PYTSB=.venv-tsb/bin/python
# stages are independent and resumable: log a failure, keep going
N() { nice -n 19 "$@" || echo "!! stage failed: $*"; }

echo "=== outlier fleet: workers=$WORKERS $(date) ==="

# 1. TSB-AD-U tuning split, both memory configs (48 series each; tune here only)
N $PY benchmarks/tsb_ad_run.py --split tuning --workers $WORKERS --scale-alpha 0.01 --det-alpha 0.005
N $PY benchmarks/tsb_ad_run.py --split tuning --workers $WORKERS --scale-alpha 0.03 --det-alpha 0.02
N $PYTSB benchmarks/tsb_ad_eval.py --split tuning --tag laplace_k3_sa0.01_da0.005
N $PYTSB benchmarks/tsb_ad_eval.py --split tuning --tag laplace_k3_sa0.03_da0.02

# 2. FRED v2 rank-percentile + detection delay (150 series each)
N $PY benchmarks/fred_anomaly_v2.py --limit 150 --workers $WORKERS
N $PY benchmarks/detection_delay.py --limit 150 --workers $WORKERS

# 3. TSB-AD-U eval split (350 series, the long leg). Config frozen from the
#    tuning read — EDIT the alphas below after step 1 before trusting this.
N $PY benchmarks/tsb_ad_run.py --split eval --workers $WORKERS --scale-alpha 0.01 --det-alpha 0.005
N $PYTSB benchmarks/tsb_ad_eval.py --split eval --tag laplace_k3_sa0.01_da0.005

# 4. UCR full-250 re-run under skaters 0.13.0 (GPD tails default) — longest leg
N $PY benchmarks/ucr_run.py --workers $WORKERS --scale-alpha 0.01 --det-alpha 0.005

echo "=== fleet complete $(date) ==="
