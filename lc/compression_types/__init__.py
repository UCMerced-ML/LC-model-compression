#!/usr/bin/env python3
from .base_types import CompressionTypeBase, PruningTypeBase
from .quantization import AdaptiveQuantization, ScaledBinaryQuantization, ScaledTernaryQuantization, BinaryQuantization, OptimalAdaptiveQuantization
from .low_rank import LowRank, RankSelection
from .pruning import ConstraintL0Pruning, ConstraintL1Pruning, PenaltyL0Pruning, PenaltyL1Pruning
