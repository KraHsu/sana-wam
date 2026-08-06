from __future__ import annotations

import ast
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "scripts/smoke_libero_ar_real_update_gpu.py"
CONFIG = ROOT / "configs/benchmarks/libero/train_libero_ar_baseline.yaml"


def _calls(tree: ast.AST) -> set[str]:
    result = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        parts = []
        while isinstance(function, ast.Attribute):
            parts.append(function.attr)
            function = function.value
        if isinstance(function, ast.Name):
            parts.append(function.id)
        if parts:
            result.add(".".join(reversed(parts)))
    return result


def test_libero_source_template_freezes_t1_and_future_exact_stats_path() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert config["model"]["video_backbone"][
        "continuous_timestep_conditioning"
    ] is True
    assert config["dataloader"]["action_stats_path"].endswith(
        "sana_wam_libero_train_all4_excl_goal82_stats_v1.npy"
    )
    assert config["training"]["action_stats_sha256"] is None
    assert config["training"]["preserve_frozen_input_grad_modules"] == [
        "video_backbone"
    ]


def test_one_update_runner_has_narrow_execution_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = _calls(tree)

    assert "architecture.prepare_inputs" in calls
    assert "architecture.compute_loss" in calls
    assert "loss.backward" in calls
    assert "optimizer.step" in calls
    assert "trainer._set_training_mode" in calls
    assert "trainer.train" not in calls
    assert "torch.save" not in calls
    assert "save_config" not in calls
    assert "manage_checkpoints" not in calls
    assert "NONFORMAL_METADATA_BOOTSTRAP_ONLY" in source
    assert "continuous_timestep_conditioning" in source
    assert "fractional_count" in source
    assert "isinstance(positional[1], torch.Tensor)" in source
    assert "isinstance(token_timestep, torch.Tensor)" in source
    assert "_create_run_root" in calls
    assert "_freeze_run_root" in calls
    assert "_assert_source_tree_clean" in calls
    assert "FAILED.json" in source
    assert "require_materialized_stats=False" in source
    assert "cfg.training.action_stats_sha256 =" not in source
    assert '"model.video_backbone.init_dit_from"' in source
    assert "int(cfg.training.seed) != args.seed" in source
    assert "len(parameter_ids) != len(set(parameter_ids))" in source
    assert "torch.isfinite(parameter).all" in source
    assert '"pre_clip_global_norm"' in source
    assert '"clipped_global_norm"' not in source
    assert "child.chmod(0o400)" in source
    assert "os.fchmod(descriptor, 0o500)" in source
    assert "EXPECTED_SANA_ASSET_SHA256" in source
    assert "EXPECTED_GEMMA_ASSET_SHA256" in source
    assert "EXPECTED_SPATIAL_SAMPLE_ASSET_SHA256" in source
    assert "EXPECTED_PRESERVE_FROZEN_INPUT_GRAD_MODULES" in source
    assert '"_sana_wam_no_grad_wrapped"' in source
    assert 'observed_checkpoint_names != ["SANA_Video_2B_480p.pth"]' in source
    assert "episode_000000.mp4" in source
    assert "external_assets = _verify_external_assets(cfg)" in source
    assert source.index(
        "external_assets = _verify_external_assets(cfg)"
    ) < source.index("gpu_before = _assert_idle_gpu")
