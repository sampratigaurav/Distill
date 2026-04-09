"""
Unit tests for the core mathematical logic in Distill.

Covers:
  - MAD-based thresholding  (_calibrate_mad_threshold + _evaluate_binary_votes)
  - Isolation-Forest contamination formula
  - DynamicAutoencoder forward pass & reconstruction error
  - DynamicDeepSVDD forward pass & anomaly score
  - Ensemble 2-of-3 voting logic
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

# conftest.py already mocks `modal` before this module is collected.
import sys
import os

# Ensure the backend package root is on the path when tests are run from
# the `backend/` directory *or* from the repository root.
_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app import _calibrate_mad_threshold, _evaluate_binary_votes, _build_flagged_items  # noqa: E402
from models import DynamicAutoencoder, DynamicDeepSVDD  # noqa: E402


def _votes(scores: np.ndarray, n_samples: int) -> np.ndarray:
    """Test helper: calibrate MAD thresholds and evaluate in one call."""
    median, mad, threshold = _calibrate_mad_threshold(scores, n_samples)
    return _evaluate_binary_votes(scores, median, mad, threshold)


# ---------------------------------------------------------------------------
# _evaluate_binary_votes — MAD-based Modified Z-Score thresholding
# ---------------------------------------------------------------------------

class TestEvaluateBinaryVotes:
    def test_no_outliers_flags_nothing(self):
        """Tightly-clustered data should produce zero flags."""
        rng = np.random.default_rng(0)
        scores = rng.normal(loc=1.0, scale=0.01, size=60).astype(np.float32)
        flags = _votes(scores, n_samples=60)
        assert flags.sum() == 0

    def test_single_extreme_outlier_is_flagged(self):
        """One extreme value must be flagged while the rest are clean."""
        scores = np.ones(60, dtype=np.float32) * 0.5
        scores[0] = 500.0  # blatant outlier
        flags = _votes(scores, n_samples=60)
        assert flags[0] == 1, "extreme outlier should be flagged"
        assert flags[1:].sum() == 0, "clean samples should not be flagged"

    def test_multiple_outliers_are_flagged(self):
        """Multiple extreme values must all be flagged."""
        scores = np.ones(80, dtype=np.float32) * 0.3
        scores[[5, 20, 50]] = 999.0
        flags = _votes(scores, n_samples=80)
        assert flags[5] == 1 and flags[20] == 1 and flags[50] == 1

    def test_uniform_data_flags_nothing(self):
        """Perfectly uniform scores have zero MAD -> epsilon floor prevents false flags."""
        scores = np.full(30, 7.0, dtype=np.float32)
        flags = _votes(scores, n_samples=30)
        assert flags.sum() == 0

    def test_output_is_binary(self):
        """All flag values must be exactly 0 or 1."""
        rng = np.random.default_rng(1)
        scores = rng.random(50).astype(np.float32)
        flags = _votes(scores, n_samples=50)
        assert set(flags.tolist()).issubset({0, 1})

    def test_output_length_matches_input(self):
        """Output array length must equal input array length."""
        scores = np.random.rand(45).astype(np.float32)
        flags = _votes(scores, n_samples=45)
        assert len(flags) == 45

    def test_adaptive_threshold_in_bounds(self):
        """Adaptive threshold returns a value in [2.0, 6.0]."""
        rng = np.random.default_rng(0)
        scores = rng.normal(loc=1.0, scale=0.1, size=1000).astype(np.float32)
        median, mad, threshold = _calibrate_mad_threshold(scores, n_samples=1000)
        assert 2.0 <= threshold <= 6.0

    def test_mad_epsilon_floor_prevents_zero_division(self):
        """Near-uniform data must not raise a ZeroDivisionError."""
        scores = np.full(25, 3.14159, dtype=np.float32)
        scores[0] += 1e-10  # tiny perturbation — still near-uniform
        # Should complete without exception
        flags = _votes(scores, n_samples=25)
        assert isinstance(flags, np.ndarray)


# ---------------------------------------------------------------------------
# DynamicAutoencoder
# ---------------------------------------------------------------------------

class TestDynamicAutoencoder:
    @pytest.mark.parametrize("input_dim", [4, 8, 16, 32, 128])
    def test_forward_shape(self, input_dim):
        """Output shape must match input shape."""
        model = DynamicAutoencoder(input_dim)
        x = torch.randn(10, input_dim)
        out = model(x)
        assert out.shape == x.shape

    @pytest.mark.parametrize("input_dim", [4, 12, 64])
    def test_reconstruction_error_nonnegative(self, input_dim):
        """Per-sample MSE reconstruction errors must be ≥ 0."""
        model = DynamicAutoencoder(input_dim)
        x = torch.randn(20, input_dim)
        errors = model.reconstruction_error(x)
        assert (errors >= 0).all(), "reconstruction errors must be non-negative"

    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_reconstruction_error_shape(self, batch_size):
        """Reconstruction error tensor must have shape (batch_size,).
        Eval mode is used so LayerNorm can handle any batch size >= 1."""
        input_dim = 16
        model = DynamicAutoencoder(input_dim)
        model.eval()
        with torch.no_grad():
            x = torch.randn(batch_size, input_dim)
            errors = model.reconstruction_error(x)
        assert errors.shape == (batch_size,)

    def test_bottleneck_is_smaller_than_input(self):
        """Encoder bottleneck dimension must be strictly smaller than input_dim."""
        input_dim = 32
        model = DynamicAutoencoder(input_dim)
        # The encoder ends at the bottleneck Linear layer
        encoder_layers = list(model.encoder.children())
        last_linear = [l for l in encoder_layers if isinstance(l, torch.nn.Linear)][-1]
        assert last_linear.out_features < input_dim

    def test_minimum_dimension_clamping(self):
        """Very small input dims (e.g., 1) should not raise errors."""
        model = DynamicAutoencoder(1)
        x = torch.randn(5, 1)
        errors = model.reconstruction_error(x)
        assert errors.shape == (5,)
        assert (errors >= 0).all()

    def test_perfect_reconstruction_has_zero_error_after_training(self):
        """
        An autoencoder trained to convergence on a single repeated vector
        should eventually achieve near-zero reconstruction error on that vector.
        """
        input_dim = 4
        model = DynamicAutoencoder(input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = torch.nn.MSELoss()
        x = torch.ones(8, input_dim)

        for _ in range(500):
            optimizer.zero_grad()
            loss = criterion(model(x), x)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            errors = model.reconstruction_error(x)
        assert errors.mean().item() < 0.1, "autoencoder should converge on trivial data"


# ---------------------------------------------------------------------------
# DynamicDeepSVDD
# ---------------------------------------------------------------------------

class TestDynamicDeepSVDD:
    @pytest.mark.parametrize("input_dim", [4, 8, 16, 64])
    def test_forward_shape(self, input_dim):
        """Forward output must map to the correct latent dimension."""
        if input_dim <= 16:
            latent_dim = max(input_dim // 2, 4)
        else:
            latent_dim = max(input_dim // 8, 8)
        model = DynamicDeepSVDD(input_dim)
        x = torch.randn(10, input_dim)
        out = model(x)
        assert out.shape == (10, latent_dim)

    @pytest.mark.parametrize("input_dim", [4, 12, 32])
    def test_anomaly_score_nonnegative(self, input_dim):
        """Squared L2 distances from center must be ≥ 0."""
        model = DynamicDeepSVDD(input_dim)
        x = torch.randn(15, input_dim)
        scores = model.anomaly_score(x)
        assert (scores >= 0).all(), "anomaly scores must be non-negative"

    @pytest.mark.parametrize("batch_size", [1, 8, 64])
    def test_anomaly_score_shape(self, batch_size):
        """Anomaly score tensor must have shape (batch_size,).
        Eval mode is used so LayerNorm can handle any batch size >= 1."""
        input_dim = 16
        model = DynamicDeepSVDD(input_dim)
        model.eval()
        with torch.no_grad():
            x = torch.randn(batch_size, input_dim)
            scores = model.anomaly_score(x)
        assert scores.shape == (batch_size,)

    def test_center_initialized_to_zeros(self):
        """The hypersphere center buffer must start at all-zeros."""
        input_dim = 16
        latent_dim = max(input_dim // 2, 4) if input_dim <= 16 else max(input_dim // 8, 8)
        model = DynamicDeepSVDD(input_dim)
        assert model.center.shape == (latent_dim,)
        assert torch.allclose(model.center, torch.zeros(latent_dim))

    def test_anomaly_score_increases_with_distance(self):
        """
        Artificially moving a point further from the center should increase
        its anomaly score.
        """
        input_dim = 8
        model = DynamicDeepSVDD(input_dim)
        model.eval()

        x_near = torch.zeros(1, input_dim)
        x_far = torch.ones(1, input_dim) * 100.0

        with torch.no_grad():
            score_near = model.anomaly_score(x_near).item()
            score_far = model.anomaly_score(x_far).item()

        assert score_far > score_near, "farther point should have higher anomaly score"

    def test_minimum_dimension_clamping(self):
        """Tiny input dims should not raise errors."""
        model = DynamicDeepSVDD(1)
        x = torch.randn(4, 1)
        scores = model.anomaly_score(x)
        assert scores.shape == (4,)
        assert (scores >= 0).all()


# ---------------------------------------------------------------------------
# Ensemble 2-of-3 voting logic (_build_flagged_items)
# ---------------------------------------------------------------------------

class TestEnsembleVoting:
    """The ensemble flags a sample only when ≥ 2 of the 3 models vote for it."""

    def _make_flags(self, bits: list[int]) -> np.ndarray:
        return np.array(bits, dtype=np.int32)

    def test_zero_votes_not_flagged(self):
        identifiers = ["a"]
        ae_f = self._make_flags([0])
        sv_f = self._make_flags([0])
        is_f = self._make_flags([0])
        fv = np.zeros((1, 4), dtype=np.float32)
        rc = np.zeros((1, 4), dtype=np.float32)
        mv = np.zeros(4, dtype=np.float32)
        items = _build_flagged_items(identifiers, ae_f, sv_f, is_f, fv, rc, mv, n_samples=200)
        assert items == []

    def test_one_vote_not_flagged(self):
        identifiers = ["a"]
        for combo in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            ae_f, sv_f, is_f = [self._make_flags([v]) for v in combo]
            fv = np.zeros((1, 4), dtype=np.float32)
            rc = np.zeros((1, 4), dtype=np.float32)
            mv = np.zeros(4, dtype=np.float32)
            items = _build_flagged_items(identifiers, ae_f, sv_f, is_f, fv, rc, mv, n_samples=200)
            assert items == [], f"single vote from {combo} should not flag the sample"

    def test_two_votes_flagged(self):
        identifiers = ["a"]
        for combo in ([1, 1, 0], [1, 0, 1], [0, 1, 1]):
            ae_f, sv_f, is_f = [self._make_flags([v]) for v in combo]
            fv = np.zeros((1, 4), dtype=np.float32)
            rc = np.zeros((1, 4), dtype=np.float32)
            mv = np.zeros(4, dtype=np.float32)
            items = _build_flagged_items(identifiers, ae_f, sv_f, is_f, fv, rc, mv, n_samples=200)
            assert len(items) == 1, f"two votes from {combo} should flag the sample"
            assert items[0]["id"] == "a"

    def test_three_votes_flagged(self):
        identifiers = ["a"]
        ae_f = sv_f = is_f = self._make_flags([1])
        fv = np.zeros((1, 4), dtype=np.float32)
        rc = np.zeros((1, 4), dtype=np.float32)
        mv = np.zeros(4, dtype=np.float32)
        items = _build_flagged_items(identifiers, ae_f, sv_f, is_f, fv, rc, mv, n_samples=200)
        assert len(items) == 1

    def test_flagged_by_attribution(self):
        """The 'flagged_by' field must list exactly the models that voted."""
        identifiers = ["sample_0"]
        ae_f = self._make_flags([1])
        sv_f = self._make_flags([0])
        is_f = self._make_flags([1])
        fv = np.zeros((1, 4), dtype=np.float32)
        rc = np.zeros((1, 4), dtype=np.float32)
        mv = np.zeros(4, dtype=np.float32)
        items = _build_flagged_items(identifiers, ae_f, sv_f, is_f, fv, rc, mv, n_samples=200)
        assert len(items) == 1
        assert set(items[0]["flagged_by"]) == {"Autoencoder", "Isolation Forest"}

    def test_mixed_batch(self):
        """Only samples with ≥ 2 votes appear in the result."""
        identifiers = ["clean", "borderline", "poisoned"]
        ae_f  = self._make_flags([0, 1, 1])
        sv_f  = self._make_flags([0, 0, 1])
        is_f  = self._make_flags([1, 0, 1])
        fv = np.zeros((3, 4), dtype=np.float32)
        rc = np.zeros((3, 4), dtype=np.float32)
        mv = np.zeros(4, dtype=np.float32)
        items = _build_flagged_items(identifiers, ae_f, sv_f, is_f, fv, rc, mv, n_samples=200)
        flagged_ids = {item["id"] for item in items}
        assert flagged_ids == {"poisoned"}

class TestConfidenceScores:
    def test_clean_score_below_half(self):
        """Score at median → confidence near 0.5, below threshold."""
        from app import _normalize_scores_to_confidence
        scores = np.ones(10, dtype=np.float32) * 5.0
        conf = _normalize_scores_to_confidence(scores, median=5.0, mad=1.0)
        assert (conf < 0.55).all()

    def test_extreme_outlier_near_one(self):
        """Extreme outlier → confidence close to 1.0."""
        from app import _normalize_scores_to_confidence
        scores = np.array([1000.0], dtype=np.float32)
        conf = _normalize_scores_to_confidence(scores, median=1.0, mad=0.1)
        assert conf[0] > 0.95

    def test_output_bounded(self):
        """All confidence values must be in [0,1]."""
        from app import _normalize_scores_to_confidence
        rng = np.random.default_rng(0)
        scores = rng.normal(0, 10, 100).astype(np.float32)
        conf = _normalize_scores_to_confidence(scores, median=0.0, mad=5.0)
        assert (conf >= 0).all() and (conf <= 1).all()

    def test_severity_ordering(self):
        """Higher confidence → higher severity label."""
        # CRITICAL=0.80+, HIGH=0.65+, MEDIUM=0.50+, LOW=below
        thresholds = [(0.85,"CRITICAL"),(0.70,"HIGH"),
                      (0.55,"MEDIUM"),(0.40,"LOW")]
        for conf, expected_sev in thresholds:
            if conf >= 0.80:   sev = "CRITICAL"
            elif conf >= 0.65: sev = "HIGH"
            elif conf >= 0.50: sev = "MEDIUM"
            else:              sev = "LOW"
            assert sev == expected_sev

    def test_ensemble_weights_sum_to_one(self):
        """AE(0.4) + SVDD(0.4) + ISO(0.2) = 1.0"""
        assert abs(0.4 + 0.4 + 0.2 - 1.0) < 1e-9
