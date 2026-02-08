"""Tests for vhap/util/vector_ops.py - Vector math operations."""

import pytest
import torch

from vhap.util.vector_ops import dot, reflect, length, safe_normalize, to_hvec


class TestDot:
    def test_parallel_vectors(self):
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[1.0, 0.0, 0.0]])
        result = dot(x, y)
        assert torch.allclose(result, torch.tensor([[1.0]]))

    def test_perpendicular_vectors(self):
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[0.0, 1.0, 0.0]])
        result = dot(x, y)
        assert torch.allclose(result, torch.tensor([[0.0]]))

    def test_antiparallel_vectors(self):
        x = torch.tensor([[1.0, 0.0, 0.0]])
        y = torch.tensor([[-1.0, 0.0, 0.0]])
        result = dot(x, y)
        assert torch.allclose(result, torch.tensor([[-1.0]]))

    def test_batch(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        result = dot(x, y)
        assert result.shape == (2, 1)
        assert torch.allclose(result, torch.tensor([[1.0], [5.0]]))


class TestReflect:
    def test_reflect_against_normal(self):
        x = torch.tensor([[1.0, -1.0, 0.0]])
        n = torch.tensor([[0.0, 1.0, 0.0]])
        result = reflect(x, n)
        expected = torch.tensor([[-1.0, 1.0, 0.0]])
        # reflect formula: 2*dot(x,n)*n - x = 2*(-1)*[0,1,0] - [1,-1,0] = [-1,-1,0]
        # Wait, checking: 2*dot(x,n) = 2*(-1) = -2; -2*n = [0,-2,0]; [0,-2,0] - [1,-1,0] = [-1,-1,0]
        expected = torch.tensor([[-1.0, -1.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_reflect_along_normal(self):
        x = torch.tensor([[0.0, 1.0, 0.0]])
        n = torch.tensor([[0.0, 1.0, 0.0]])
        result = reflect(x, n)
        # 2*dot(x,n)*n - x = 2*1*[0,1,0] - [0,1,0] = [0,1,0]
        expected = torch.tensor([[0.0, 1.0, 0.0]])
        assert torch.allclose(result, expected)


class TestLength:
    def test_unit_vector(self):
        x = torch.tensor([[1.0, 0.0, 0.0]])
        result = length(x)
        assert torch.allclose(result, torch.tensor([[1.0]]))

    def test_3d_vector(self):
        x = torch.tensor([[3.0, 4.0, 0.0]])
        result = length(x)
        assert torch.allclose(result, torch.tensor([[5.0]]))

    def test_zero_vector(self):
        x = torch.tensor([[0.0, 0.0, 0.0]])
        result = length(x)
        # Should not be NaN due to eps clamping
        assert not torch.isnan(result).any()

    def test_batch(self):
        x = torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 4.0]])
        result = length(x)
        expected = torch.tensor([[1.0], [5.0]])
        assert torch.allclose(result, expected)


class TestSafeNormalize:
    def test_unit_vector(self):
        x = torch.tensor([[5.0, 0.0, 0.0]])
        result = safe_normalize(x)
        expected = torch.tensor([[1.0, 0.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_general_vector(self):
        x = torch.tensor([[3.0, 4.0, 0.0]])
        result = safe_normalize(x)
        expected = torch.tensor([[0.6, 0.8, 0.0]])
        assert torch.allclose(result, expected)

    def test_normalized_has_unit_length(self):
        x = torch.randn(10, 3)
        result = safe_normalize(x)
        lengths = length(result)
        assert torch.allclose(lengths, torch.ones(10, 1), atol=1e-5)

    def test_zero_vector_no_nan(self):
        x = torch.tensor([[0.0, 0.0, 0.0]])
        result = safe_normalize(x)
        assert not torch.isnan(result).any()


class TestToHvec:
    def test_homogeneous_point(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = to_hvec(x, w=1.0)
        expected = torch.tensor([[1.0, 2.0, 3.0, 1.0]])
        assert torch.allclose(result, expected)

    def test_homogeneous_direction(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        result = to_hvec(x, w=0.0)
        expected = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
        assert torch.allclose(result, expected)

    def test_batch(self):
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = to_hvec(x, w=1.0)
        assert result.shape == (2, 4)
        assert torch.allclose(result[:, 3], torch.tensor([1.0, 1.0]))
