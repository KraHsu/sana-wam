"""Growing-history (variable-length, frame-0-anchored) windowing + clean-prefix.

The growing-history logic is pure index/mask math, so the core is unit-tested on
a bare ``RoboTwinDataset`` instance (``__new__`` bypasses the data-loading
``__init__``). An optional integration test runs against the real RoboTwin
dataset when ``/DATA/share/RoboTwin2.0/dataset`` is present.
"""

from __future__ import annotations

import os

import pytest

from sana_wam.dataloader.robotwin_dataset import RoboTwinDataset

_DATA_ROOT = "/DATA/share/RoboTwin2.0/dataset"
_HAS_DATA = os.path.isdir(_DATA_ROOT)


def _bare(**attrs):
    """A RoboTwinDataset with only the attributes the helpers under test read."""
    ds = RoboTwinDataset.__new__(RoboTwinDataset)
    defaults = dict(
        growing_history=True,
        history_min_frames=5,
        history_stride=4,
        gdn_chunk_size=3,
        video_stride=4,
        temporal_compression=4,
        causal_temporal=True,
        split="train",
    )
    defaults.update(attrs)
    for k, v in defaults.items():
        setattr(ds, k, v)
    return ds


def test_max_raw_window_on_grid_lands_on_stride_and_temporal_grid():
    ds = _bare()
    # longest episode 117 → smallest L>=117 with (L-1)%4==0 and (nvf-1)%4==0
    L = ds._max_raw_window_on_grid(117)
    assert L >= 117
    assert (L - 1) % ds.video_stride == 0
    nvf = len(range(0, L, ds.video_stride))
    assert (nvf - 1) % ds.temporal_compression == 0


def test_video_len_to_latent_causal():
    ds = _bare()
    # causal Wan VAE: T_lat = 1 + (T_vid - 1)//4
    assert ds._video_len_to_latent(1) == 1
    assert ds._video_len_to_latent(33) == 1 + 32 // 4
    assert ds._video_len_to_latent(29) == 1 + 28 // 4


def test_clean_prefix_is_chunk_aligned_and_leaves_one_supervised_chunk():
    ds = _bare(gdn_chunk_size=3)
    # build a sample with plenty of valid history (full 33-frame clip → T_lat=9)
    ds._video_sample_indices = list(range(0, 129, 4))  # 33 indices
    P = ds._sample_clean_prefix_latent(ep_idx=0, logical_len=129, actual_valid_len=129)
    t_lat = ds._video_len_to_latent(33)  # 9
    assert P % ds.gdn_chunk_size == 0
    assert 0 <= P <= t_lat - ds.gdn_chunk_size  # at least one supervised chunk remains


def test_clean_prefix_bootstrap_returns_zero_for_short_history():
    ds = _bare(gdn_chunk_size=3)
    ds._video_sample_indices = list(range(0, 129, 4))
    # only ~2 valid video frames ⇒ T_lat=1 < chunk ⇒ no room for a clean chunk
    P = ds._sample_clean_prefix_latent(ep_idx=0, logical_len=5, actual_valid_len=5)
    assert P == 0


def test_clean_prefix_is_deterministic_per_sample():
    ds = _bare()
    ds._video_sample_indices = list(range(0, 129, 4))
    a = ds._sample_clean_prefix_latent(0, 129, 129)
    b = ds._sample_clean_prefix_latent(0, 129, 129)
    assert a == b  # reproducible given (ep_idx, logical_len, split)


def test_non_growing_clean_prefix_is_zero():
    ds = _bare(growing_history=False)
    ds._video_sample_indices = list(range(0, 129, 4))
    assert ds._sample_clean_prefix_latent(0, 129, 129) == 0


def test_proprio_raw_index_tracks_latest_observed_frame():
    # tc=4, video_stride=4 ⇒ each extra clean latent advances the current frame by 16.
    ds = _bare(video_stride=4, temporal_compression=4)
    # P_lat<=1 (legacy single-observation / bootstrap): current state is frame 0.
    assert ds._proprio_raw_index(0, actual_valid_len=129) == 0
    assert ds._proprio_raw_index(1, actual_valid_len=129) == 0
    # Growing-history: latest observed = last clean latent (P_lat-1) → tc*(P_lat-1)*stride.
    assert ds._proprio_raw_index(2, actual_valid_len=129) == 16   # 4*1*4
    assert ds._proprio_raw_index(3, actual_valid_len=129) == 32   # 4*2*4
    # Clamped into the valid window (never points at a padded/fabricated frame).
    assert ds._proprio_raw_index(9, actual_valid_len=20) == 19


def test_proprio_raw_index_zero_for_non_growing():
    ds = _bare(growing_history=False)
    # Non-growing always has P_lat=0 ⇒ current state stays at frame 0 (window start).
    assert ds._proprio_raw_index(0, actual_valid_len=49) == 0


@pytest.mark.skipif(not _HAS_DATA, reason="RoboTwin dataset not present")
def test_growing_history_integration_lift_pot():
    from omegaconf import OmegaConf

    from sana_wam.dataloader.robotwin_dataset import MultiTaskRoboTwinDataset

    cfg = OmegaConf.create(
        dict(
            dataset_dir=_DATA_ROOT,
            robot="aloha-agilex",
            variant="clean_50",
            action_mode="eef",
            task_name=None,
            train_tasks=["lift_pot"],
            num_frames=49,
            video_stride=4,
            height=384,
            width=320,
            multiview=True,
            camera_layout=["head_camera", "left_camera", "right_camera"],
            target_camera="head_camera",
            normalize_mode="min-max",
            val_ratio=0.0,
            filter_static_segments=False,
            growing_history=True,
            history_min_frames=5,
            history_stride=8,
            gdn_chunk_size=3,
        )
    )
    ds = MultiTaskRoboTwinDataset.from_config(cfg, split="train")
    sub = ds._sub_datasets[0]
    # raw buffer spans the longest episode
    assert sub._raw_window_len >= max(sub._episode_lengths)
    assert (sub._raw_window_len - 1) % sub.video_stride == 0
    # all windows anchored at frame 0 with increasing logical length
    for ep_idx, start, k in sub._window_index[:50]:
        assert start == 0
        assert k >= sub.history_min_frames

    chunk = sub.gdn_chunk_size
    seen_valid = set()
    for i in range(0, len(ds), max(1, len(ds) // 40)):
        s = ds[i]
        P = int(s["num_clean_prefix_latent"])
        assert P % chunk == 0
        assert len(s["video"]) == sub.num_video_frames  # fixed buffer (pad-to-max-T)
        seen_valid.add(int(s["video_mask"].sum()))
    # growing windows exercise a range of valid lengths, not a single fixed one
    assert len(seen_valid) > 3
