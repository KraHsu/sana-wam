"""Shared test fixtures and sys.path setup for sana-wam.

Puts ``third_party/Sana`` and ``src/`` on sys.path so unit tests resolve both
the SANA upstream ``diffusion.*`` package and the ``sana_wam`` package without
an editable install. Mini-backbone tests need ``third_party/Sana`` importable
(they build the real ``SanaMSVideo`` factory) but NOT the 2B weights.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
SANA_ROOT = PROJECT_ROOT / "third_party" / "Sana"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))  # so `from tests.<mod> import ...` resolves
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if SANA_ROOT.is_dir() and str(SANA_ROOT) not in sys.path:
    sys.path.insert(0, str(SANA_ROOT))

# Fire the SANA mmcv 1.x → mmengine compatibility shim before any test's
# top-level ``import diffusion...`` probe. Guarded so missing mmcv/SANA does
# not block CPU-only collection.
try:  # noqa: SIM105
    import sana_wam.model.video_backbone.sana  # noqa: F401
except Exception:
    pass


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test requires GPU")
    config.addinivalue_line("markers", "data: test requires RoboTwin data")
