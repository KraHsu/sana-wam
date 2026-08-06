# LIBERO benchmark adapter

This directory connects the LIBERO simulator to an **already-running**
sana-wam policy server. The simulator remains isolated here; repository-level
LIBERO data and server launchers are listed below and do not install LIBERO or
start training implicitly.

## Current status and evidence boundary

This initial integration is an adapter/scaffold only. It does not establish a
LIBERO success rate and does not make the current RoboTwin checkpoints suitable
for LIBERO:

- RoboTwin joint checkpoints emit 14D bimanual joint actions.
- RoboTwin EEF checkpoints emit 20D bimanual absolute-pose actions.
- LIBERO requires a 7D single-arm relative action.

Those action spaces have different dimensions and semantics. The adapter must
therefore fail closed on a non-7D policy response; it must never slice, pad, or
reinterpret a RoboTwin checkpoint. A meaningful rollout requires a checkpoint
trained with the LIBERO observation, 8D proprioception, 7D relative-action, and
normalization contracts described below.

The external LIBERO source is pinned to commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01`; the launcher rejects any other
checkout HEAD.

## Files

| File | Purpose |
|---|---|
| `policy_config.yml` | Frozen client-side observation, action, gripper, and suite-budget contract. |
| `sana_wam2libero_interface.py` | Thin HTTP client and pure LIBERO conversions; imports no simulator package. |
| `eval_policy.py` | Deterministic LIBERO task/initial-state rollout loop; imports LIBERO lazily. |
| `single_eval.sh` | Validates the launch arguments and enters the dedicated simulator environment. |
| `requirements-data.txt` | PyArrow pin for a separately authorized native-data environment update. |

## Frozen benchmark contract

### Observations

LIBERO images are rotated 180 degrees before transmission to match the
orientation used by the standard LIBERO training/evaluation pipelines. The
server payload mapping is:

| LIBERO observation | sana-wam payload |
|---|---|
| `agentview_image` | `head_camera` |
| `robot0_eye_in_hand_image` | `left_wrist_camera` |
| no third LIBERO view | `right_wrist_camera: null` |

The sana-wam checkpoint contract in this project uses exactly 8D
proprioception:

```text
[eef_xyz(3), eef_quaternion_as_axis_angle(3), gripper_qpos(2)]
```

This is a project-specific choice aligned with the LeRobot LIBERO data already
present on H200. It is not a claim that LIBERO exposes only one possible robot
state representation. Quaternion conversion deliberately matches the
robosuite/GR00T data-conversion formula byte-for-semantics: it does not replace
`q` by `-q` when `w < 0`.

The policy checkpoint's single-view/multi-view preprocessing remains
server-owned. Its training configuration must use the same two-view mapping and
missing-right-camera convention as the client. Stored LeRobot MP4s are already
in canonical training orientation and are decoded without rotation; only live
simulator images are rotated 180 degrees by the benchmark adapter.

Action and proprioception have separate normalization contracts. Actions use
the `libero_relative_eef` 7D stats entry, while the 8D state uses
`libero_eef_axis_angle_gripper`. Deployment reports both actually constructed
normalizers through `/info`; a config declaration without matching runtime
normalizers is rejected before simulator construction.

### Actions

The server must return exactly seven finite values:

```text
[delta_xyz(3), delta_axis_angle(3), open_gripper(1)]
```

The first six relative-motion values pass through unchanged. Only the gripper
coordinate is converted. The model/data convention is `open=1`, `closed=0`;
LIBERO's environment convention is `open=-1`, `closed=+1`. With the frozen 0.5
threshold, the mapping is:

```text
env_gripper = -1  if model_open_gripper > 0.5
               +1  otherwise
```

The client sends the current observation to the policy server on every
environment step. It does not create a second action queue: action chunking and
stateful cache alignment belong to the sana-wam server.

Before any simulator environment or output root is created, the client binds
`/info.inference_runtime.deployment_identity` to the complete
`expected_server_contract` in `policy_config.yml`. A compatible checkpoint must
save that mapping at `dataloader.benchmark_contract`; its derived action/state
dimensions, dataloader type, action mode, normalization mode, multiview flag,
and camera layout must agree. This rejects both RoboTwin checkpoints and a
coincidentally 7D checkpoint with different semantics.

### Suites and rollout budgets

Every episode first executes 5 stabilization steps with the 7D zero action
`[0, 0, 0, 0, 0, 0, 0]`. Direct runner use defaults to 20 fixed initial
states per task; `single_eval.sh` makes the requested count explicit. The
task-step budgets are:

| Suite | Tasks | Maximum policy steps |
|---|---:|---:|
| `libero_spatial` | 10 | 220 |
| `libero_object` | 10 | 280 |
| `libero_goal` | 10 | 300 |
| `libero_10` | 10 | 520 |

These suite-specific limits are the common VLA evaluation caps used by this
project, not LIBERO's lifelong-learning default of 600 steps. `libero_90` is
allowed only as a dataset/pretraining suite here and is deliberately rejected
by the closed-loop launcher.

The runner uses LIBERO's fixed task initial states. It issues exactly one named
server reset per episode, then records per-episode output and an aggregate
summary under the selected output directory. Every reset key includes a fresh
run nonce and trial index, duplicate resets are rejected, and a stable explicit
model-noise seed is derived for each episode. Every `/predict` response must
echo the active episode key, model seed, and exact next server step. Requests
for more trials than available fixed init states fail instead of silently
counting repeated states.

Each unique result root ends with `TERMINAL.json`: `COMPLETE` only after the
expected episode count and summary are written, otherwise `FAILED` on a caught
exception. A missing terminal file is incomplete evidence and must not be
reported as a benchmark result.

The remaining frozen simulator identity is:

| Field | Value |
|---|---:|
| LIBERO task-order index | 0 |
| Environment seed | 0 |
| Render height | 256 |
| Render width | 256 |

## Training data path

The repository now includes a native LeRobot v2.1 loader and a deployable AR
baseline source template:

```text
src/sana_wam/dataloader/libero_dataset.py
src/sana_wam/dataloader/libero_stats.py
src/sana_wam/train/libero_contract.py
src/sana_wam/deploy/libero_model_loader.py
src/sana_wam/deploy/libero_policy_server.py
scripts/build_libero_stats.py
scripts/smoke_libero_data.py
scripts/train_libero.py
scripts/deploy_libero.py
configs/benchmarks/libero/train_libero_ar_baseline.yaml
```

The loader reads `observation.state[t] -> action[t]` exactly. It does not reuse
RoboTwin's absolute-state `t+1` label convention. Numeric rows are loaded lazily
with optional PyArrow, and the two AV1 videos are decoded with PyAV. Install the
data-only optional dependency in the sana-wam training environment before an
authorized data smoke:

```bash
uv pip install --python .venv/bin/python -r benchmarks/libero/requirements-data.txt
```

This dependency is kept outside `pyproject.toml` and `uv.lock` because those
files are inputs to frozen predecessor evidence. The recorded H200 CPU smoke
used the pinned data-only requirement; each new environment must still install
and verify it explicitly.

The subsequent full-2B, real-sample, update-free GPU construction smoke is
recorded in
[`docs/libero/LIBERO_AR_REAL_GPU_UPDATE_FREE_SMOKE_20260806.md`](../../docs/libero/LIBERO_AR_REAL_GPU_UPDATE_FREE_SMOKE_20260806.md).
It validates one finite AR architecture forward, not training or benchmark
quality.

Normalization stats can be materialized without decoding Parquet or video:

```bash
python scripts/build_libero_stats.py \
  --output /DATA/share/LIBERO/sana_wam_libero_all_minmax_stats.npy \
  /DATA/share/LIBERO/libero/libero_spatial_no_noops_1.0.0_lerobot \
  /DATA/share/LIBERO/libero/libero_object_no_noops_1.0.0_lerobot \
  /DATA/share/LIBERO/libero/libero_goal_no_noops_1.0.0_lerobot \
  /DATA/share/LIBERO/libero/libero_10_no_noops_1.0.0_lerobot
```

The current H200 Goal snapshot has a corrupted wrist video for episode 82.
The Isaac-GR00T patch available locally belongs to a different 169-frame
revision, while this snapshot has 129 state/action rows, so it must not be
copied over the shared data. The frozen initial training contract excludes the
complete episode (`libero_goal_no_noops_1.0.0_lerobot:82`). The loader rejects
using that snapshot without the exclusion and also rejects the incompatible
patch. No shared dataset file is modified.

The generic Trainer and policy-server sources are likewise frozen predecessor
inputs. LIBERO therefore enters through the dedicated launchers, which validate
the 7D/8D contract and install a composite deploy normalizer: 7D action
history/output uses action stats, while 8D proprioception uses independent state
stats. Do not launch a LIBERO checkpoint through the generic deploy entrypoint.

The fixed CPU real-data smoke and its evidence boundary are recorded in
[`docs/libero/LIBERO_CPU_REAL_DATA_SMOKE_20260806.md`](../../docs/libero/LIBERO_CPU_REAL_DATA_SMOKE_20260806.md).

## Environment

Keep LIBERO and MuJoCo in a dedicated Python environment. The sana-wam project
environment uses Python 3.12, whereas the pinned LIBERO/robosuite stack commonly
uses Python 3.10 and older dependency versions. Do not install the simulator
stack into the main sana-wam environment.

Three variables are mandatory:

```bash
export LIBERO_PATH=/path/to/LIBERO          # checkout containing libero/
export LIBERO_PYTHON=/path/to/env/bin/python
export LIBERO_CONFIG_PATH=/path/to/libero-config-directory
```

On the current H200 host, an existing read-only-compatible combination is:

```bash
export LIBERO_PATH=/home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO
export LIBERO_PYTHON=/home/zch/workspace/starVLA/.venv-libero/bin/python
export LIBERO_CONFIG_PATH=/path/to/pinned-libero-config
```

`LIBERO_CONFIG_PATH` must already exist and contain `config.yaml`. Requiring it
prevents LIBERO's import-time first-run prompt from interactively creating or
rewriting `~/.libero`. For the checkout above, a dedicated config has this
shape (replace the dataset path if necessary):

```yaml
benchmark_root: /home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO/libero/libero
bddl_files: /home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO/libero/libero/bddl_files
init_states: /home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO/libero/libero/init_files
assets: /home/zch/workspace/Isaac-GR00T/external_dependencies/LIBERO/libero/libero/assets
datasets: /DATA/share/LIBERO/libero
```

The checkout and environment remain external inputs; a reproducible run must
record their resolved paths and versions. `single_eval.sh` supplies
`LIBERO_PATH` through `PYTHONPATH`, selects EGL rendering, and confines the
simulator process with `CUDA_VISIBLE_DEVICES`. It does not alter the GPU or
process used by the policy server.

## Usage

Start a sana-wam server separately with a compatible LIBERO-native checkpoint.
Then run:

```bash
bash benchmarks/libero/single_eval.sh \
  <suite> <task_id|all> <num_trials> <simulator_gpu_id> [http_port] [host]
```

For example, the following selects task 0 of LIBERO-Spatial, runs two initial
states, places the simulator on physical GPU 3, and connects to an existing
server at `127.0.0.1:8848`:

```bash
LIBERO_OUTPUT_DIR=/DATA/share/sana_wam_libero_smoke/example \
bash benchmarks/libero/single_eval.sh \
  libero_spatial 0 2 3 8848 127.0.0.1
```

`task_id` may be `all` or `0..N-1` for the selected suite. The default output
directory is `benchmarks/libero/results`; set `LIBERO_OUTPUT_DIR` to a unique
large-storage path for actual runs. `LIBERO_POLICY_CONFIG` can select an
alternate explicitly pinned client config.

The script only connects to a server. It never starts one, loads a checkpoint,
trains a model, or turns an adapter smoke into a benchmark claim.
