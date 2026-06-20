# RoboTwin Benchmark Evaluation

Drive a **running** sana-wam policy server from the RoboTwin simulator. These
scripts only cover the RoboTwin (client) side of the loop — start the server
first (see [Start the server](#3-start-the-server)).

## Files

| File | Description |
|---|---|
| `single_eval.sh` | Run evaluation on a single task against an already-running server. |
| `eval_policy_wrapper.py` | Runs RoboTwin's `script/eval_policy.py` without its fragile render self-test. |
| `sana_wam2robotwin_interface.py` | RoboTwin `ModelClient` — calls the server's `/predict` and `/reset` HTTP endpoints. |
| `policy_config.yml` | Client config template; `host` / `http_port` are injected at runtime by `single_eval.sh`. |
| `step_limits.yml` | Per-task `step_lim` overrides (see below). |
| `export_results_csv.py` | Parse a log directory's `summary.tsv` + per-task `Success rate` lines into a CSV. |

## Per-task step_lim overrides

RoboTwin ships upstream per-task step limits in `task_config/_eval_step_limit.yml`.
To tweak them without patching the RoboTwin source tree, edit
[`step_limits.yml`](step_limits.yml) in this directory:

```yaml
adjust_bottle: 160
open_laptop: 288
put_bottles_dustbin: 640
```

Semantics:

- Any task listed here overrides RoboTwin's upstream value for that task.
- Tasks not listed keep RoboTwin's original value (which itself falls back to
  `1000` when the upstream file also lacks the task).
- The file is loaded once at adapter import and applied at the first step of each
  episode via `TASK_ENV.step_lim = <override>`. Edits only take effect in a fresh
  eval process — restart the evaluation after tweaking values.
- Leaving the file empty (comments only) reproduces stock RoboTwin behavior.

## Environment Setup

### 1. Install the RoboTwin environment

Follow the [official RoboTwin installation guide](https://github.com/RoboTwin-Platform/RoboTwin)
to clone the repo, create the Conda environment, install dependencies, and
download assets. You should end up with a working RoboTwin Conda environment and
a local checkout of the RoboTwin repository.

### 2. Match the checkpoint action/state mode

`policy_config.yml` must match the checkpoint's saved `config.yaml`:

- `dataloader.action_mode: eef` / `architecture.state_dim: 20` → keep
  `action_type: ee`, `state_dim: 20`.
- `dataloader.action_mode: joint` / `architecture.state_dim: 14` → set
  `action_type: qpos`, `state_dim: 14`.
- Keep `send_state: true` for any checkpoint with
  `architecture.use_proprioception: true`; the adapter fails fast if the
  extracted RoboTwin state dimension is wrong.

### 3. Start the server

Start the sana-wam policy server separately, from the project root, before
running any evaluation (it can live on a remote machine — just make sure the
host/IP and HTTP port are reachable):

```bash
uv run python scripts/deploy.py --ckpt-dir /path/to/ckpt_dir --device cuda:0
```

The greedy AR policy is mandatory: the engine is a stateful closed-loop with a
persistent KV cache, so async / receding-horizon / temporal-ensemble modes break
cache alignment.

## Usage

```bash
bash single_eval.sh <task_name> <task_config> <ckpt_setting> <gpu_id> [http_port] [host]
```

| Argument | Description |
|---|---|
| `task_name` | RoboTwin task name (e.g. `adjust_bottle`). |
| `task_config` | `demo_clean` or `demo_randomized`. |
| `ckpt_setting` | Label written into result filenames (e.g. `sana_wam`). |
| `gpu_id` | CUDA device for the RoboTwin simulator. |
| `http_port` | Server HTTP port (default: `8848`). |
| `host` | Server address (default: `127.0.0.1`). |

**Example:**

```bash
bash single_eval.sh adjust_bottle demo_clean sana_wam 0 8848 127.0.0.1
```

**Environment variables read by `single_eval.sh`:**

| Variable | Required | Description |
|---|---|---|
| `ROBOTWIN_PATH` | yes | RoboTwin repository root. |
| `ROBOTWIN_PYTHON` | no (default `python`) | Python interpreter for the RoboTwin env. |
| `ROBOTWIN_HTTP_PORT` | no (default `8848`) | Server HTTP port (overridden by the 5th positional arg). |
| `ROBOTWIN_POLICY_HOST` | no (default `127.0.0.1`) | Server host (overridden by the 6th positional arg). |
| `ROBOTWIN_TEST_NUM` | no (default upstream `100`) | Cap on eval episodes for smoke runs. |
| `POLICY_CONFIG_PATH` | no | Override the `policy_config.yml` template path. |

## Export evaluation results to CSV

`export_results_csv.py` combines a log directory's `summary.tsv` and each
per-task log's last `Success rate` line into a single CSV. It searches nested
`node*/worker*/*.log` files when `summary.tsv` is absent and strips ANSI escape
sequences before parsing.

```bash
python benchmarks/robotwin/export_results_csv.py /path/to/log_dir -o /path/to/results.csv
```

If `-o` is omitted the default output is `<log_dir>/results.csv`. Pass `--strict`
to return a non-zero exit code when any task log is missing or lacks a parseable
success rate.

## FAQ

### Render Error (headless servers)

On a headless Linux box the SAPIEN renderer fails to find an X display and
raises `Render Error`. Start a virtual framebuffer (Xvfb):

```bash
sudo apt-get install -y xvfb   # if not installed yet
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
bash single_eval.sh adjust_bottle demo_clean sana_wam 0 8848 127.0.0.1
```

Or, in one step, use `xvfb-run`:

```bash
xvfb-run -a bash single_eval.sh adjust_bottle demo_clean sana_wam 0 8848 127.0.0.1
```

### `policy_config.yml` options

```yaml
# Observation settings
# The client always forwards RoboTwin's head / left / right cameras to the server.
# The server inspects the checkpoint's saved config.yaml to decide:
#   - multiview=false → single-view preprocessing using head_camera only
#   - multiview=true  → composed into the L-shape multi-view layout used at training time
# The client no longer needs to configure camera selection or resolution.
send_state: true          # Include the proprio state vector in the /predict request.
state_dim: 20             # Fail-fast expected dim. 20 for eef/ee, 14 for joint/qpos.
request_timeout: 300      # HTTP timeout in seconds.

# Action settings (must match the `action_mode` used at training time).
action_type: ee           # ee   — EEF mode (action_mode: eef at train time, default).
                          #        Server returns 20D (xyz + rot6d + grip) × 2,
                          #        auto-converted to 16D (xyz + quat + grip) × 2 before dispatch.
                          # qpos — Joint-angle mode (action_mode: joint at train time).
                          #        Server returns 14D, passed straight to take_action.

# action_indices: null    # Optional index reordering for the returned action vector (null = no reorder).
```
