# HypSEE

Semi-supervised graph classification with hierarchical hypergraph structure learning.

## Environment

```bash
conda activate HEAL   # or your PyG / PyTorch environment
pip install torch torch-geometric wandb
wandb login
mkdir -p logs
```

## Training modes

| Mode | Command | W&B |
|------|---------|-----|
| Plain | `python main.py --data_name PROTEINS` | off |
| Single run + log | `python main.py --use_wandb --data_name PROTEINS` | one run |
| Hyperparameter sweep | `python main.py --sweep ...` | bayes sweep |

Sweep uses `configs/sweep_base.py` (`--sweep_profile base`) by default. Hyperparameters are sampled by W&B; **`epochs` is fixed to 100** per trial via `FIXED_RUN_DEFAULTS` in the base profile (not searched).

## W&B sweep (nohup)

Long sweeps should run with `nohup` so training continues after SSH disconnect.  
Naming convention: `SWEEP_NAME="${DATASET}_${SWEEP_TAG}"`.

Monitor:

```bash
tail -f logs/sweep_PROTEINS_base_v0628_*.log
kill $(cat logs/sweep_PROTEINS_base_v0628.pid)
cat logs/sweep_registry.log
```

### Template

```bash
DATASET="PROTEINS"
SWEEP_TAG="base_v0628"          # custom suffix; sweep_name = <dataset>_<tag>
SWEEP_NAME="${DATASET}_${SWEEP_TAG}"

LOG_FILE="logs/sweep_${SWEEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

CUDA_VISIBLE_DEVICES=0 nohup python -u main.py \
  --sweep \
  --data_name "${DATASET}" \
  --sweep_name "${SWEEP_NAME}" \
  --sweep_profile base \
  --sweep_iters 500 \
  --runs 5 \
  --epochs 100 \
  --wandb_project HypSEE \
  > "${LOG_FILE}" 2>&1 &

SWEEP_PID=$!
echo "${SWEEP_PID}" > "logs/sweep_${SWEEP_NAME}.pid"
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid=${SWEEP_PID}  sweep_name=${SWEEP_NAME}  log=${LOG_FILE}" >> logs/sweep_registry.log
echo "Sweep PID: ${SWEEP_PID} (sweep_name: ${SWEEP_NAME})"
```

Or use the helper script (same layout):

```bash
chmod +x scripts/run_sweep_nohup.sh
DATASET=PROTEINS SWEEP_TAG=base_v0628 ./scripts/run_sweep_nohup.sh
```

### PROTEINS

```bash
DATASET="IMDB-BINARY"
SWEEP_TAG="base_v0628"
SWEEP_NAME="${DATASET}_${SWEEP_TAG}"

LOG_FILE="logs/sweep_${SWEEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup /home/zengguangjie/anaconda3/envs/HEAL/bin/python -u main.py \
  --sweep \
  --data_name "${DATASET}" \
  --sweep_name "${SWEEP_NAME}" \
  --sweep_profile base \
  --sweep_iters 500 \
  --runs 5 \
  --epochs 100 \
  --wandb_project HypSEE \
  > "${LOG_FILE}" 2>&1 &

SWEEP_PID=$!
echo "${SWEEP_PID}" > "logs/sweep_${SWEEP_NAME}.pid"
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid=${SWEEP_PID}  sweep_name=${SWEEP_NAME}  log=${LOG_FILE}" >> logs/sweep_registry.log
echo "Sweep PID: ${SWEEP_PID} (sweep_name: ${SWEEP_NAME})"
```

### IMDB-BINARY

```bash
DATASET="IMDB-BINARY"
SWEEP_TAG="base_v0628"
SWEEP_NAME="${DATASET}_${SWEEP_TAG}"

LOG_FILE="logs/sweep_${SWEEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup /home/zengguangjie/anaconda3/envs/HEAL/bin/python -u main.py \
  --sweep \
  --data_name "${DATASET}" \
  --sweep_name "${SWEEP_NAME}" \
  --sweep_profile base \
  --sweep_iters 500 \
  --runs 5 \
  --epochs 100 \
  --wandb_project HypSEE \
  > "${LOG_FILE}" 2>&1 &

SWEEP_PID=$!
echo "${SWEEP_PID}" > "logs/sweep_${SWEEP_NAME}.pid"
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid=${SWEEP_PID}  sweep_name=${SWEEP_NAME}  log=${LOG_FILE}" >> logs/sweep_registry.log
echo "Sweep PID: ${SWEEP_PID} (sweep_name: ${SWEEP_NAME})"
```

### COLLAB

Larger graphs; reduce `batch_size` if OOM.

```bash
DATASET="COLLAB"
SWEEP_TAG="base_v0628"
SWEEP_NAME="${DATASET}_${SWEEP_TAG}"

LOG_FILE="logs/sweep_${SWEEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup /home/zengguangjie/anaconda3/envs/HEAL/bin/python -u main.py \
  --sweep \
  --data_name "${DATASET}" \
  --sweep_name "${SWEEP_NAME}" \
  --sweep_profile base \
  --sweep_iters 500 \
  --runs 5 \
  --epochs 100 \
  --batch_size 32 \
  --wandb_project HypSEE \
  > "${LOG_FILE}" 2>&1 &

SWEEP_PID=$!
echo "${SWEEP_PID}" > "logs/sweep_${SWEEP_NAME}.pid"
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid=${SWEEP_PID}  sweep_name=${SWEEP_NAME}  log=${LOG_FILE}" >> logs/sweep_registry.log
echo "Sweep PID: ${SWEEP_PID} (sweep_name: ${SWEEP_NAME})"
```

### REDDIT-BINARY

```bash
DATASET="REDDIT-BINARY"
SWEEP_TAG="base_v0628"
SWEEP_NAME="${DATASET}_${SWEEP_TAG}"

LOG_FILE="logs/sweep_${SWEEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup /home/zengguangjie/anaconda3/envs/HEAL/bin/python -u main.py \
  --sweep \
  --data_name "${DATASET}" \
  --sweep_name "${SWEEP_NAME}" \
  --sweep_profile base \
  --sweep_iters 500 \
  --runs 5 \
  --epochs 100 \
  --wandb_project HypSEE \
  > "${LOG_FILE}" 2>&1 &

SWEEP_PID=$!
echo "${SWEEP_PID}" > "logs/sweep_${SWEEP_NAME}.pid"
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid=${SWEEP_PID}  sweep_name=${SWEEP_NAME}  log=${LOG_FILE}" >> logs/sweep_registry.log
echo "Sweep PID: ${SWEEP_PID} (sweep_name: ${SWEEP_NAME})"
```

### Quick smoke test

`quick` profile overrides **`epochs=2`**, `runs=1`.

```bash
DATASET="PROTEINS"
SWEEP_TAG="quick_smoke"
SWEEP_NAME="${DATASET}_${SWEEP_TAG}"

LOG_FILE="logs/sweep_${SWEEP_NAME}_$(date +%Y%m%d_%H%M%S).log"

nohup /home/zengguangjie/anaconda3/envs/HEAL/bin/python -u main.py \
  --sweep \
  --data_name "${DATASET}" \
  --sweep_name "${SWEEP_NAME}" \
  --sweep_profile quick \
  --sweep_iters 3 \
  --wandb_project HypSEE \
  > "${LOG_FILE}" 2>&1 &

SWEEP_PID=$!
echo "${SWEEP_PID}" > "logs/sweep_${SWEEP_NAME}.pid"
echo "$(date '+%Y-%m-%d %H:%M:%S')  pid=${SWEEP_PID}  sweep_name=${SWEEP_NAME}  log=${LOG_FILE}" >> logs/sweep_registry.log
echo "Sweep PID: ${SWEEP_PID} (sweep_name: ${SWEEP_NAME})"
```

### Sweep notes

- Sweep does **not** load `configs/{data_name}.json` (`use_config_file=False`); only CLI defaults + sampled parameters apply.
- `feat_str` is in the search space; each value caches preprocessed features under `data/{dataset}/processed/data_{feat_str}.pt`.
- `num_edges1` is set equal to sampled `num_edges2` automatically.
- Metric optimized: `final/avg_test_acc` (maximize).
- `epochs=100` is enforced by `FIXED_RUN_DEFAULTS` in `configs/sweep_base.py` during base-profile sweeps.
- `nohup` prevents hangup on SSH disconnect; reboot or `kill` still stops the job.

## Plain / single-run examples

```bash
# Default training (300 epochs)
python main.py --data_name PROTEINS --epochs 300

# Use dataset JSON hyperparameters
python main.py --data_name IMDB-BINARY --use_config_file

# Single run with W&B logging
python main.py --use_wandb --data_name PROTEINS --epochs 100 --wandb_name proteins_run1
```

## Project layout

```
main.py              # entry point (plain / --use_wandb / --sweep)
Exp.py               # training loop
configs/sweep_*.py   # W&B sweep profiles
configs/*.json       # per-dataset defaults (--use_config_file)
scripts/             # launch helpers (run_sweep_nohup.sh)
data/                # TU datasets + precomputed hypergraphs
models/              # HypSEE model
logs/                # nohup logs, pid files, sweep_registry.log
```
