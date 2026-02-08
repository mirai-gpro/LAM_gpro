"""Tests for vhap/util/camera.py - Camera utilities."""

import pytest
import torch
import numpy as np

from vhap.util.camera import (
    gram_schmidt_orthogonalization,
    projection_from_intrinsics,
    OrbitCamera,
)


class TestGramSchmidt:
    def test_identity_matrix(self):
        M = torch.eye(3).clone()
        result = gram_schmidt_orthogonalization(M)
        assert torch.allclose(result, torch.eye(3), atol=1e-6)

    def test_orthonormality(self):
        M = torch.randn(3, 3)
        result = gram_schmidt_orthogonalization(M)
        # Check orthogonality: M^T @ M should be close to identity
        product = result.T @ result
        assert torch.allclose(product, torch.eye(3), atol=1e-5)

    def test_columns_unit_length(self):
        M = torch.randn(3, 3)
        result = gram_schmidt_orthogonalization(M)
        for c in range(3):
            col_norm = result[:, c].norm()
            assert torch.allclose(col_norm, torch.tensor(1.0), atol=1e-5)


class TestProjectionFromIntrinsics:
    def test_basic_projection(self):
        K = np.array([[[500, 0, 320], [0, 500, 240], [0, 0, 1]]], dtype=np.float64)
        proj = projection_from_intrinsics(K, image_size=(480, 640), near=0.1, far=100)
        assert proj.shape == (1, 4, 4)
        # Check fx component: 2*500/640
        assert np.isclose(proj[0, 0, 0], 2 * 500 / 640)
        # Check fy component: 2*500/480
        assert np.isclose(proj[0, 1, 1], 2 * 500 / 480)

    def test_batch(self):
        K = np.tile(
            np.array([[[500, 0, 320], [0, 500, 240], [0, 0, 1]]]), (3, 1, 1)
        ).astype(np.float64)
        proj = projection_from_intrinsics(K, image_size=(480, 640))
        assert proj.shape == (3, 4, 4)

    def test_flip_y(self):
        K = np.array([[[500, 0, 320], [0, 500, 240], [0, 0, 1]]], dtype=np.float64)
        proj_normal = projection_from_intrinsics(K, image_size=(480, 640), flip_y=False)
        proj_flipped = projection_from_intrinsics(K, image_size=(480, 640), flip_y=True)
        assert np.isclose(proj_flipped[0, 1, 1], -proj_normal[0, 1, 1])

    def test_4d_intrinsics(self):
        K = np.array([[500, 500, 320, 240]], dtype=np.float64)
        proj = projection_from_intrinsics(K, image_size=(480, 640))
        assert proj.shape == (1, 4, 4)


class TestOrbitCamera:
    def test_init_opengl(self):
        cam = OrbitCamera(640, 480, r=2, fovy=60, convention="opengl")
        assert cam.image_width == 640
        assert cam.image_height == 480
        assert cam.radius == 2
        assert cam.z_sign == -1
        assert cam.y_sign == -1

    def test_init_opencv(self):
        cam = OrbitCamera(640, 480, r=2, fovy=60, convention="opencv")
        assert cam.z_sign == 1
        assert cam.y_sign == 1

    def test_init_invalid_convention(self):
        with pytest.raises(ValueError, match="Unknown convention"):
            OrbitCamera(640, 480, convention="invalid")

    def test_pose_shape(self):
        cam = OrbitCamera(640, 480)
        pose = cam.pose
        assert pose.shape == (4, 4)

    def test_intrinsics(self):
        cam = OrbitCamera(640, 480, fovy=60)
        intr = cam.intrinsics
        assert len(intr) == 4  # fx, fy, cx, cy
        # cx should be w//2, cy should be h//2
        assert intr[2] == 320
        assert intr[3] == 240

    def test_fovx(self):
        cam = OrbitCamera(640, 480, fovy=60)
        assert cam.fovx == 60 / 480 * 640

    def test_projection_matrix(self):
        cam = OrbitCamera(640, 480)
        proj = cam.projection_matrix
        assert proj.shape == (4, 4)

    def test_world_view_transform(self):
        cam = OrbitCamera(640, 480)
        wvt = cam.world_view_transform
        assert wvt.shape == (4, 4)

    def test_full_proj_transform(self):
        cam = OrbitCamera(640, 480)
        fpt = cam.full_proj_transform
        assert fpt.shape == (4, 4)

    def test_scale(self):
        cam = OrbitCamera(640, 480, r=2.0)
        original_radius = cam.radius
        cam.scale(1)  # zoom in
        assert cam.radius < original_radius
        cam.scale(-1)  # zoom out
        # After zooming in then out, should be close to original
        assert abs(cam.radius - original_radius) < 0.01

    def test_reset(self):
        cam = OrbitCamera(640, 480, r=2.0)
        cam.scale(5)
        cam.orbit(30, 45)
        cam.reset()
        assert cam.radius == 2.0
        assert np.allclose(cam.look_at, [0, 0, 0])
