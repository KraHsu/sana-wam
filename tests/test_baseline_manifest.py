"""Integrity checks for the canonical AR reference baseline."""

import hashlib
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "docs" / "baselines" / "ar_lownoise_seedfixed.manifest.yaml"
HISTORY113_DEPLOY = ROOT / "configs" / "deploy_ar_lownoise_history113.yaml"


def test_reference_config_matches_recorded_hash():
    manifest = OmegaConf.load(MANIFEST)
    config_path = ROOT / str(manifest.training.config_copy)
    digest = hashlib.sha256(config_path.read_bytes()).hexdigest()
    assert digest == manifest.training.config_sha256


def test_seed_fixed_metric_is_internally_consistent():
    manifest = OmegaConf.load(MANIFEST)
    evaluation = manifest.evaluation
    assert evaluation.completed_episodes == 86
    assert evaluation.successes == 8
    expected = evaluation.successes / evaluation.completed_episodes
    assert abs(evaluation.success_rate - expected) < 1e-10
    assert evaluation.completed_episodes < evaluation.requested_episodes

    native = manifest.native_evaluation
    assert native.exit_code == 0
    assert native.completed_episodes == native.requested_episodes == 100
    assert native.successes == 8
    assert native.failures == 92
    assert native.success_rate == native.successes / native.completed_episodes == 0.08
    assert native.unique_episode_seeds == 100
    assert len(set(native.successful_seeds)) == native.successes
    assert native.video_artifacts == 100
    assert native.repository_step_limit_overrides == 0
    assert native.step_limit_source == "robotwin_upstream_default"
    assert native.comparison_to_historical.fisher_exact_two_sided_p > 0.05


def test_native_deploy_overlay_matches_recorded_eval():
    manifest = OmegaConf.load(MANIFEST)
    deploy = OmegaConf.load(ROOT / str(manifest.evaluation.deploy_config))
    assert manifest.status == "native_inference_verified"
    assert (
        deploy.checkpoint_name
        == manifest.native_port.loader_verification.checkpoint_name
    )
    assert deploy.inference.seed is None
    assert deploy.inference.video_steps == manifest.evaluation.video_steps == 10
    assert deploy.inference.action_steps == manifest.evaluation.action_steps == 10
    assert deploy.policy.execute_horizon is None
    assert deploy.policy.history_len == manifest.evaluation.policy_history_len == 10
    assert deploy.policy.temporal_ensemble is True
    assert manifest.evaluation.temporal_ensemble_effective is False
    assert deploy.server.host == "127.0.0.1"
    deploy_digest = hashlib.sha256(
        (ROOT / str(manifest.evaluation.deploy_config)).read_bytes()
    ).hexdigest()
    assert deploy_digest == manifest.native_evaluation.current_deploy_config_sha256
    assert manifest.native_evaluation.deploy_config_sha256_at_run != deploy_digest
    assert manifest.evaluation.configured_schedule_label == "sync"
    assert (
        manifest.evaluation.effective_denoise_order
        == "chunk0_action_only_then_later_chunks_video_then_action"
    )
    assert deploy.inference.shift == 5.0
    assert manifest.evaluation.effective_video_shift == 3.0
    assert manifest.evaluation.effective_action_shift == 5.0


def test_native_checkpoint_schema_audit_is_strict_and_complete():
    native = OmegaConf.load(MANIFEST).native_port
    assert native.loader_verification.strict is True
    assert native.loader_verification.result == "passed"
    assert native.loader_verification.loaded_state_keys == 1741
    assert (
        native.checkpoint_schema.expected_keys
        == native.checkpoint_schema.checkpoint_keys
        == 1741
    )
    assert native.checkpoint_schema.missing_keys == 0
    assert native.checkpoint_schema.unexpected_keys == 0
    assert native.checkpoint_schema.shape_mismatches == 0
    assert native.gpu_smoke_verification.final_engine_step_c == 2
    assert native.gpu_smoke_verification.finite_actions is True
    assert native.differential_verification.result == "PASS_BITWISE"
    assert native.differential_verification.physical_actions_bitwise_equal is True
    assert native.differential_verification.normalized_max_abs_error == 0.0
    assert native.cpu_regression_tests.result == "passed"
    assert native.cpu_regression_tests.passed == 180


def test_history113_experiment_is_a_single_variable_deploy_ablation():
    manifest = OmegaConf.load(MANIFEST)
    canonical = OmegaConf.load(ROOT / str(manifest.evaluation.deploy_config))
    experiment = OmegaConf.load(HISTORY113_DEPLOY)

    assert canonical.policy.history_len == 10
    assert experiment.policy.history_len == 113
    assert experiment.policy.execute_horizon is None
    assert experiment.policy.temporal_ensemble is True

    # Normalize the single A/B field before checking the remaining config.
    experiment.policy.history_len = canonical.policy.history_len
    assert OmegaConf.to_container(experiment, resolve=True) == OmegaConf.to_container(
        canonical, resolve=True
    )

    followup = manifest.followup_evaluations.history113
    assert followup.single_variable == "policy.history_len"
    assert followup.confirmation.completed_episodes == 100
    assert followup.confirmation.successes == 10
    assert followup.confirmation.success_rate == 0.10
    assert followup.confirmation.comparison_to_canonical.rate_difference == 0.02
    assert followup.confirmation.comparison_to_canonical.fisher_exact_two_sided_p > 0.05
    assert followup.confirmation.comparison_to_canonical.paired_mcnemar_exact_p > 0.05
