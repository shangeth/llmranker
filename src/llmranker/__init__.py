from importlib.metadata import PackageNotFoundError, version

from .benchmark import compare_rankers
from .llm import LLMConfig, LLMResponse
from .metrics import RankingMetrics
from .rankers import (
    BaseRanker,
    ListwiseRanker,
    PairwiseRanker,
    PointwiseRanker,
    SetwiseRanker,
    TourRankRanker,
)
from .types import Candidate

try:
    __version__ = version("llmranker")
except PackageNotFoundError:
    # Not installed (e.g. running from a source checkout with no build metadata).
    __version__ = "0.0.0+unknown"

__all__ = [
    "BaseRanker",
    "Candidate",
    "LLMConfig",
    "LLMResponse",
    "ListwiseRanker",
    "PairwiseRanker",
    "PointwiseRanker",
    "RankingMetrics",
    "SetwiseRanker",
    "TourRankRanker",
    "compare_rankers",
]
