"""Pipeline-specific transforms that bridge between model-agnostic data and
pipeline-specific conditioning fields.

These transforms are applied at the training pipeline level (not in the
dataset), keeping the dataset output model-agnostic.
"""

from sana_wam.dataloader.transforms.base import ModalityTransform


class FirstFrameConditioningTransform(ModalityTransform):
    """Add first-frame conditioning fields to a sample.

    Reads ``data["video"]`` and derives:
    - ``first_frame_image``: the first frame, used as a first-frame /
      reference-image condition by backbones that support it. This transform
      stays backend-agnostic; the mapping to the backbone's input happens at
      the pipeline boundary (see the video backbone adapter).

    Args:
        use_first_frame_as_reference: If True, set ``first_frame_image``
            to ``[video[0]]``. If False, set to None.
    """

    def __init__(self, use_first_frame_as_reference: bool = True):
        super().__init__(apply_to=["video"])
        self.use_first_frame_as_reference = use_first_frame_as_reference

    def apply(self, data: dict) -> dict:
        if "first_frame_image" not in data:
            if self.use_first_frame_as_reference and "video" in data and data["video"]:
                data["first_frame_image"] = [data["video"][0]]
            else:
                data["first_frame_image"] = None

        return data
