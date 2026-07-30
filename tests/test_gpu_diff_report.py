"""CPU checks for the GPU differential report helpers."""

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_diff_script():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "diff_openwam_sanawam_gpu.py"
    )
    spec = importlib.util.spec_from_file_location("_gpu_diff_report", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delta_metrics_are_json_serializable():
    module = _load_diff_script()
    left = np.zeros((2, 3), dtype=np.float32)
    right = left.copy()
    right[1, 2] = 0.25

    metrics = module._delta_metrics(left, right)

    assert metrics["worst_index"] == [1, 2]
    assert metrics["max_abs"] == 0.25
    json.dumps(metrics)
