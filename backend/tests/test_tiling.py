from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

VENDOR_ROOT = Path(__file__).resolve().parents[1] / "vendor" / "talk_combined"
sys.path.insert(0, str(VENDOR_ROOT))

from talc_analysis.inference import TileWindow, merge_tile_predictions  # noqa: E402


def test_center_weighted_vote_suppresses_internal_tile_edge() -> None:
    left = TileWindow(0, 0, 4, 1)
    right = TileWindow(2, 0, 4, 1)
    result = merge_tile_predictions(
        6,
        1,
        [
            (left, np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)),
            (right, np.zeros((1, 4), dtype=np.float32)),
        ],
        threshold=0.5,
        blend_margin=2,
    )

    assert result.mask.tolist() == [[0, 0, 0, 0, 0, 0]]
    assert result.positive_votes[0, 3] == 1
    assert result.vote_count[0, 3] == 2


def test_unweighted_merge_keeps_original_hard_voting_contract() -> None:
    window = TileWindow(0, 0, 1, 1)
    result = merge_tile_predictions(
        1,
        1,
        [
            (window, np.array([[0.9]], dtype=np.float32)),
            (window, np.array([[0.1]], dtype=np.float32)),
        ],
        threshold=0.5,
    )

    assert result.mask.item() == 1
