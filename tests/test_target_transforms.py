"""
Tests for target transform roundtrip correctness.

Verifies that:
1. Forward then inverse transform recovers original prices
2. Multi-step reconstruction is mathematically correct
3. Backward compat with use_percentage_returns works
"""
import pytest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'modules', 'data_science'))

from forecast_utils import (
    get_target_transform,
    to_target,
    from_target,
    from_target_cumulative,
    VALID_TARGET_TRANSFORMS
)


class TestTargetTransformBasics:
    """Test basic transform functionality."""
    
    def test_valid_transforms_defined(self):
        """Ensure all valid transforms are listed."""
        assert 'price' in VALID_TARGET_TRANSFORMS
        assert 'pct_change' in VALID_TARGET_TRANSFORMS
        assert 'log_return' in VALID_TARGET_TRANSFORMS
        assert len(VALID_TARGET_TRANSFORMS) == 3
    
    def test_get_target_transform_default(self):
        """Default should be 'price'."""
        config = {}
        assert get_target_transform(config) == 'price'
    
    def test_get_target_transform_explicit(self):
        """Explicit target_transform setting."""
        config = {'data_science': {'target_transform': 'log_return'}}
        assert get_target_transform(config) == 'log_return'
    
    def test_get_target_transform_backward_compat(self):
        """Legacy use_percentage_returns should map to pct_change."""
        config = {'data_science': {'use_percentage_returns': True}}
        assert get_target_transform(config) == 'pct_change'
        
        config = {'data_science': {'use_percentage_returns': False}}
        assert get_target_transform(config) == 'price'
    
    def test_get_target_transform_priority(self):
        """target_transform takes priority over use_percentage_returns."""
        config = {'data_science': {'target_transform': 'log_return', 'use_percentage_returns': True}}
        assert get_target_transform(config) == 'log_return'


class TestPriceTransform:
    """Test identity transform (no change)."""
    
    def test_price_roundtrip(self):
        """Price transform should be identity."""
        prices = np.array([[100, 102, 105], [200, 198, 195]])
        context = np.array([99, 199])
        
        transformed = to_target(prices, 'price', context)
        np.testing.assert_array_equal(transformed, prices)
        
        recovered = from_target(transformed, 'price', context)
        np.testing.assert_array_equal(recovered, prices)


class TestPctChangeTransform:
    """Test percentage change transform."""
    
    def test_pct_change_forward(self):
        """Forward transform: (p - p0) / p0."""
        context = np.array([100.0])
        prices = np.array([[105.0, 110.0, 120.0]])  # +5%, +10%, +20%
        
        transformed = to_target(prices, 'pct_change', context)
        
        expected = np.array([[0.05, 0.10, 0.20]])
        np.testing.assert_allclose(transformed, expected, rtol=1e-10)
    
    def test_pct_change_inverse(self):
        """Inverse transform: p0 * (1 + pct)."""
        context = np.array([100.0])
        pct_changes = np.array([[0.05, 0.10, 0.20]])
        
        recovered = from_target(pct_changes, 'pct_change', context)
        
        expected = np.array([[105.0, 110.0, 120.0]])
        np.testing.assert_allclose(recovered, expected, rtol=1e-10)
    
    def test_pct_change_roundtrip(self):
        """Full roundtrip should recover original prices."""
        context = np.array([100.0, 200.0, 50.0])
        prices = np.array([
            [102.0, 105.0, 110.0],
            [190.0, 185.0, 200.0],
            [52.0, 55.0, 48.0]
        ])
        
        transformed = to_target(prices, 'pct_change', context)
        recovered = from_target(transformed, 'pct_change', context)
        
        np.testing.assert_allclose(recovered, prices, rtol=1e-10)
    
    def test_pct_change_negative(self):
        """Negative returns should work correctly."""
        context = np.array([100.0])
        prices = np.array([[95.0, 90.0, 80.0]])  # -5%, -10%, -20%
        
        transformed = to_target(prices, 'pct_change', context)
        expected = np.array([[-0.05, -0.10, -0.20]])
        np.testing.assert_allclose(transformed, expected, rtol=1e-10)
        
        recovered = from_target(transformed, 'pct_change', context)
        np.testing.assert_allclose(recovered, prices, rtol=1e-10)


class TestLogReturnTransform:
    """Test log return transform."""
    
    def test_log_return_forward(self):
        """Forward transform: log(p / p0)."""
        context = np.array([100.0])
        prices = np.array([[105.0, 110.0, 120.0]])
        
        transformed = to_target(prices, 'log_return', context)
        
        expected = np.log(prices / context[:, None])
        np.testing.assert_allclose(transformed, expected, rtol=1e-10)
    
    def test_log_return_inverse(self):
        """Inverse transform: p0 * exp(log_ret)."""
        context = np.array([100.0])
        log_rets = np.array([[0.05, 0.10, 0.20]])
        
        recovered = from_target(log_rets, 'log_return', context)
        
        expected = context[:, None] * np.exp(log_rets)
        np.testing.assert_allclose(recovered, expected, rtol=1e-10)
    
    def test_log_return_roundtrip(self):
        """Full roundtrip should recover original prices."""
        context = np.array([100.0, 200.0, 50.0])
        prices = np.array([
            [102.0, 105.0, 110.0],
            [190.0, 185.0, 200.0],
            [52.0, 55.0, 48.0]
        ])
        
        transformed = to_target(prices, 'log_return', context)
        recovered = from_target(transformed, 'log_return', context)
        
        np.testing.assert_allclose(recovered, prices, rtol=1e-10)
    
    def test_log_return_negative_returns(self):
        """Negative log returns (price decreases)."""
        context = np.array([100.0])
        prices = np.array([[95.0, 90.0, 80.0]])
        
        transformed = to_target(prices, 'log_return', context)
        # log_return should be negative
        assert np.all(transformed < 0)
        
        recovered = from_target(transformed, 'log_return', context)
        np.testing.assert_allclose(recovered, prices, rtol=1e-10)


class TestCumulativeTransform:
    """Test cumulative (chained) transform for recursive forecasts."""
    
    def test_cumulative_pct_change(self):
        """Cumulative pct_change reconstruction."""
        context = np.array([100.0])
        
        # Day 1: +5%, Day 2: +3% from day 1, Day 3: -2% from day 2
        step_changes = np.array([[0.05, 0.03, -0.02]])
        
        recovered = from_target_cumulative(step_changes, 'pct_change', context)
        
        # Day 1: 100 * 1.05 = 105
        # Day 2: 105 * 1.03 = 108.15
        # Day 3: 108.15 * 0.98 = 105.987
        expected = np.array([[105.0, 108.15, 105.987]])
        np.testing.assert_allclose(recovered, expected, rtol=1e-10)
    
    def test_cumulative_log_return(self):
        """Cumulative log_return reconstruction."""
        context = np.array([100.0])
        
        # Step log returns
        step_log_rets = np.array([[0.05, 0.03, -0.02]])
        
        recovered = from_target_cumulative(step_log_rets, 'log_return', context)
        
        # Day 1: 100 * exp(0.05)
        # Day 2: Day1 * exp(0.03)
        # Day 3: Day2 * exp(-0.02)
        expected_day1 = 100 * np.exp(0.05)
        expected_day2 = expected_day1 * np.exp(0.03)
        expected_day3 = expected_day2 * np.exp(-0.02)
        expected = np.array([[expected_day1, expected_day2, expected_day3]])
        
        np.testing.assert_allclose(recovered, expected, rtol=1e-10)
    
    def test_cumulative_price_identity(self):
        """Cumulative price should just return prices."""
        context = np.array([100.0])
        prices = np.array([[105.0, 108.0, 106.0]])
        
        recovered = from_target_cumulative(prices, 'price', context)
        np.testing.assert_array_equal(recovered, prices)


class TestEdgeCases:
    """Test edge cases and numerical stability."""
    
    def test_zero_context_price(self):
        """Handle zero context price gracefully."""
        context = np.array([0.0, 100.0])
        prices = np.array([[1.0, 2.0], [105.0, 110.0]])
        
        # Should not raise - uses epsilon internally
        transformed = to_target(prices, 'pct_change', context)
        assert not np.any(np.isnan(transformed))
        assert not np.any(np.isinf(transformed))
    
    def test_single_sample(self):
        """Single sample should work."""
        context = np.array([100.0])
        prices = np.array([[105.0]])
        
        transformed = to_target(prices, 'pct_change', context)
        recovered = from_target(transformed, 'pct_change', context)
        
        np.testing.assert_allclose(recovered, prices, rtol=1e-10)
    
    def test_large_horizon(self):
        """Large horizon should work."""
        n_samples = 10
        horizon = 30
        context = np.linspace(100, 200, n_samples)
        
        # Generate random prices
        np.random.seed(42)
        prices = context[:, None] * (1 + np.random.randn(n_samples, horizon) * 0.01)
        
        for transform in ['pct_change', 'log_return']:
            transformed = to_target(prices, transform, context)
            recovered = from_target(transformed, transform, context)
            np.testing.assert_allclose(recovered, prices, rtol=1e-9)


class TestMathematicalProperties:
    """Test mathematical properties of transforms."""
    
    def test_log_return_sum_property(self):
        """Sum of log returns equals log of final/initial."""
        context = np.array([100.0])
        prices = np.array([[100.0, 105.0, 110.0, 108.0]])  # Include context as t=0
        
        # Calculate step-by-step log returns
        log_rets = np.log(prices[0, 1:] / prices[0, :-1])
        
        # Sum should equal log(final/initial)
        total_log_ret = np.sum(log_rets)
        expected = np.log(108.0 / 100.0)
        
        np.testing.assert_allclose(total_log_ret, expected, rtol=1e-10)
    
    def test_pct_change_vs_log_return(self):
        """For small changes, pct_change ≈ log_return."""
        context = np.array([100.0])
        # Small change
        prices = np.array([[100.5, 101.0]])
        
        pct = to_target(prices, 'pct_change', context)
        log_ret = to_target(prices, 'log_return', context)
        
        # Should be approximately equal for small changes
        np.testing.assert_allclose(pct, log_ret, rtol=0.01)
    
    def test_pct_change_diverges_for_large_changes(self):
        """For large changes, pct_change != log_return."""
        context = np.array([100.0])
        # Large change: +50%
        prices = np.array([[150.0]])
        
        pct = to_target(prices, 'pct_change', context)[0, 0]
        log_ret = to_target(prices, 'log_return', context)[0, 0]
        
        assert pct == 0.5
        assert log_ret == pytest.approx(np.log(1.5), rel=1e-10)
        assert abs(pct - log_ret) > 0.05  # Significant difference


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
