from importlib.metadata import PackageNotFoundError, version

from .benchmark import compare_rankers
from .llm import LLMConfig, LLMResponse
from .metrics import RankingMetrics
from .rankers import (
    BaseRanker,
    CascadeRanker,
    ListwiseRanker,
    PairwiseRanker,
    PointwiseRanker,
    SetwiseRanker,
    TourRankRanker,
)
from .types import Candidate, Ranker

try:
    __version__ = version("llmranker")
except PackageNotFoundError:
    # Not installed (e.g. running from a source checkout with no build metadata).
    __version__ = "0.0.0+unknown"

__all__ = [
    "BaseRanker",
    "Candidate",
    "CascadeRanker",
    "LLMConfig",
    "LLMResponse",
    "ListwiseRanker",
    "PairwiseRanker",
    "PointwiseRanker",
    "Ranker",
    "RankingMetrics",
    "SetwiseRanker",
    "TourRankRanker",
    "compare_rankers",
]
