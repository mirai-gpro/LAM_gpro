"""Tests for lam/datasets/cam_utils.py - Camera math utilities."""

import pytest
import torch
import math

from lam.datasets.cam_utils import (
    compose_extrinsic_R_T,
    compose_extrinsic_RT,
    decompose_extrinsic_R_T,
    decompose_extrinsic_RT,
    get_normalized_camera_intrinsics,
    build_camera_principle,
    build_camera_standard,
    center_looking_at_camera_pose,
    surrounding_views_linspace,
    create_intrinsics,
)


class TestComposeDecompose:
    def test_compose_extrinsic_RT_identity(self):
        RT = torch.eye(4)[:3, :].unsqueeze(0)  # (1, 3, 4)
        E = compose_extrinsic_RT(RT)
        assert E.shape == (1, 4, 4)
        expected = torch.eye(4).unsqueeze(0)
        assert torch.allclose(E, expected)

    def test_compose_extrinsic_R_T(self):
        R = torch.eye(3).unsqueeze(0)  # (1, 3, 3)
        T = torch.tensor([[1.0, 2.0, 3.0]])  # (1, 3)
        E = compose_extrinsic_R_T(R, T)
        assert E.shape == (1, 4, 4)
        assert torch.allclose(E[0, :3, :3], torch.eye(3))
        assert torch.allclose(E[0, :3, 3], torch.tensor([1.0, 2.0, 3.0]))
        assert torch.allclose(E[0, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]))

    def test_decompose_extrinsic_RT(self):
        E = torch.eye(4).unsqueeze(0)
        RT = decompose_extrinsic_RT(E)
        assert RT.shape == (1, 3, 4)
        assert torch.allclose(RT, E[:, :3, :])

    def test_decompose_extrinsic_R_T(self):
        E = torch.eye(4).unsqueeze(0)
        E[0, :3, 3] = torch.tensor([5.0, 6.0, 7.0])
        R, T = decompose_extrinsic_R_T(E)
        assert R.shape == (1, 3, 3)
        assert T.shape == (1, 3)
        assert torch.allclose(R, torch.eye(3).unsqueeze(0))
        assert torch.allclose(T, torch.tensor([[5.0, 6.0, 7.0]]))

    def test_compose_decompose_roundtrip(self):
        R = torch.randn(2, 3, 3)
        T = torch.randn(2, 3)
        E = compose_extrinsic_R_T(R, T)
        R_out, T_out = decompose_extrinsic_R_T(E)
        assert torch.allclose(R, R_out)
        assert torch.allclose(T, T_out)

    def test_compose_decompose_RT_roundtrip(self):
        RT = torch.randn(3, 3, 4)
        E = compose_extrinsic_RT(RT)
        RT_out = decompose_extrinsic_RT(E)
        assert torch.allclose(RT, RT_out)

    def test_batch_compose(self):
        RT = torch.randn(5, 3, 4)
        E = compose_extrinsic_RT(RT)
        assert E.shape == (5, 4, 4)
        # Last row should be [0, 0, 0, 1] for all
        for i in range(5):
            assert torch.allclose(E[i, 3, :], torch.tensor([0.0, 0.0, 0.0, 1.0]))


class TestNormalizedIntrinsics:
    def test_basic_normalization(self):
        # (N, 3, 2): [[fx, fy], [cx, cy], [w, h]]
        intrinsics = torch.tensor([[[100.0, 200.0], [50.0, 100.0], [200.0, 400.0]]])
        fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
        assert torch.allclose(fx, torch.tensor([0.5]))  # 100/200
        assert torch.allclose(fy, torch.tensor([0.5]))  # 200/400
        assert torch.allclose(cx, torch.tensor([0.25]))  # 50/200
        assert torch.allclose(cy, torch.tensor([0.25]))  # 100/400

    def test_square_image(self):
        intrinsics = torch.tensor([[[512.0, 512.0], [256.0, 256.0], [512.0, 512.0]]])
        fx, fy, cx, cy = get_normalized_camera_intrinsics(intrinsics)
        assert torch.allclose(fx, torch.tensor([1.0]))
        assert torch.allclose(fy, torch.tensor([1.0]))
        assert torch.allclose(cx, torch.tensor([0.5]))
        assert torch.allclose(cy, torch.tensor([0.5]))


class TestCreateIntrinsics:
    def test_with_c(self):
        intr = create_intrinsics(f=100.0, c=50.0, w=200.0, h=200.0)
        assert intr.shape == (3, 2)
        assert torch.allclose(intr[0], torch.tensor([0.5, 0.5]))  # fx/w, fy/h
        assert torch.allclose(intr[1], torch.tensor([0.25, 0.25]))  # cx/w, cy/h
        assert torch.allclose(intr[2], torch.tensor([1.0, 1.0]))  # normalized w, h

    def test_with_cx_cy(self):
        intr = create_intrinsics(f=200.0, cx=100.0, cy=100.0, w=400.0, h=400.0)
        assert intr.shape == (3, 2)

    def test_c_and_cx_cy_conflict(self):
        with pytest.raises(AssertionError):
            create_intrinsics(f=100.0, c=50.0, cx=50.0, cy=50.0)

    def test_no_c_no_cx_cy_raises(self):
        with pytest.raises(AssertionError):
            create_intrinsics(f=100.0)


class TestCenterLookingAt:
    def test_basic_pose(self):
        pos = torch.tensor([[0.0, 0.0, 2.0]])
        ext = center_looking_at_camera_pose(pos)
        assert ext.shape == (1, 3, 4)
        # Camera at (0,0,2) looking at origin, z-axis should point away from origin
        assert ext[0, 2, 3] == 2.0  # z position

    def test_multiple_cameras(self):
        pos = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        ext = center_looking_at_camera_pose(pos)
        assert ext.shape == (3, 3, 4)


class TestSurroundingViews:
    def test_single_view(self):
        ext = surrounding_views_linspace(1, radius=2.0, height=0.8)
        assert ext.shape == (1, 3, 4)

    def test_multiple_views(self):
        ext = surrounding_views_linspace(8, radius=2.0, height=0.8)
        assert ext.shape == (8, 3, 4)

    def test_radius_constraint(self):
        n = 4
        ext = surrounding_views_linspace(n, radius=3.0, height=0.0)
        # All cameras should be at distance 3 from origin
        positions = ext[:, :, 3]  # (N, 3)
        dists = positions.norm(dim=-1)
        assert torch.allclose(dists, torch.tensor([3.0] * n), atol=1e-5)

    def test_zero_views_raises(self):
        with pytest.raises(AssertionError):
            surrounding_views_linspace(0)


class TestBuildCamera:
    def test_build_camera_principle_shape(self):
        RT = torch.eye(4)[:3, :].unsqueeze(0)  # (1, 3, 4)
        intrinsics = torch.tensor([[[512.0, 512.0], [256.0, 256.0], [512.0, 512.0]]])
        result = build_camera_principle(RT, intrinsics)
        assert result.shape == (1, 16)  # 12 + 4

    def test_build_camera_standard_shape(self):
        RT = torch.eye(4)[:3, :].unsqueeze(0)  # (1, 3, 4)
        intrinsics = torch.tensor([[[512.0, 512.0], [256.0, 256.0], [512.0, 512.0]]])
        result = build_camera_standard(RT, intrinsics)
        assert result.shape == (1, 25)  # 16 + 9
