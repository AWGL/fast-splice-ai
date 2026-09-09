"""FastSpliceAI - an accelerated CPU inference engine for the SpliceAI models."""
__version__ = "0.1.0"

from .annotator import Annotator          # noqa: F401
from .engine import Ensemble              # noqa: F401
from .score import get_delta_scores       # noqa: F401
from .api import SpliceAI, Variant, parse_variant, score_variants  # noqa: F401
