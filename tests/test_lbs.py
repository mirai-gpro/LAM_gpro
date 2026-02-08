"""Tests for lam/models/rendering/flame_model/lbs.py - Linear Blend Skinning."""

import importlib.util
import os
import pytest
import torch
import torch.nn.functional as F

# Load lbs.py directly from file to avoid lam.models.__init__ (requires accelerate)
_lbs_path = os.path.join(os.path.dirname(__file__), "..", "lam", "models", "rendering", "flame_model", "lbs.py")
_spec = importlib.util.spec_from_file_location("lbs", _lbs_path)
_lbs_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lbs_mod)

batch_rodrigues = _lbs_mod.batch_rodrigues
vertices2joints = _lbs_mod.vertices2joints
blend_shapes = _lbs_mod.blend_shapes
transform_mat = _lbs_mod.transform_mat
batch_rigid_transform = _lbs_mod.batch_rigid_transform


class TestBatchRodrigues:
    def test_identity_rotation(self):
        # Near-zero rotation should give ~identity
        rot_vecs = torch.zeros(1, 3)
        R = batch_rodrigues(rot_vecs)
        assert R.shape == (1, 3, 3)
        # Due to epsilon in the implementation, not exactly identity
        # but should be close
        assert torch.allclose(R[0], torch.eye(3), atol=1e-3)

    def test_batch_shape(self):
        rot_vecs = torch.randn(5, 3)
        R = batch_rodrigues(rot_vecs)
        assert R.shape == (5, 3, 3)

    def test_rotation_determinant(self):
        # Rotation matrices should have determinant 1
        rot_vecs = torch.randn(10, 3)
        R = batch_rodrigues(rot_vecs)
        for i in range(10):
            det = torch.det(R[i])
            assert torch.allclose(det, torch.tensor(1.0), atol=1e-4)

    def test_orthogonality(self):
        # R^T @ R should be identity
        rot_vecs = torch.randn(5, 3)
        R = batch_rodrigues(rot_vecs)
        for i in range(5):
            product = R[i].T @ R[i]
            assert torch.allclose(product, torch.eye(3), atol=1e-4)

    def test_x_rotation(self):
        # 90 degrees about x-axis
        angle = torch.pi / 2
        rot_vecs = torch.tensor([[angle, 0.0, 0.0]])
        R = batch_rodrigues(rot_vecs)
        # Should map y -> z, z -> -y (approximately)
        y_axis = torch.tensor([0.0, 1.0, 0.0])
        rotated = R[0] @ y_axis
        expected = torch.tensor([0.0, 0.0, 1.0])
        assert torch.allclose(rotated, expected, atol=1e-3)


class TestVertices2Joints:
    def test_basic_regression(self):
        # Simple regressor that averages vertices
        J_regressor = torch.tensor([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]])  # (2, 3)
        vertices = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
        )  # (1, 3, 3)
        joints = vertices2joints(J_regressor, vertices)
        assert joints.shape == (1, 2, 3)
        # Joint 0 = 0.5*v0 + 0.5*v1 = [0.5, 0.5, 0.0]
        assert torch.allclose(joints[0, 0], torch.tensor([0.5, 0.5, 0.0]))

    def test_batch(self):
        J_regressor = torch.eye(3)[:2]  # (2, 3) - select first 2 vertices
        vertices = torch.randn(4, 3, 3)
        joints = vertices2joints(J_regressor, vertices)
        assert joints.shape == (4, 2, 3)


class TestBlendShapes:
    def test_basic(self):
        betas = torch.tensor([[1.0, 0.0]])  # (1, 2)
        # 3 vertices, 3D positions, 2 shape components
        shape_disps = torch.randn(3, 3, 2)
        result = blend_shapes(betas, shape_disps)
        assert result.shape == (1, 3, 3)
        # Only first shape component should contribute
        expected = shape_disps[:, :, 0].unsqueeze(0)
        assert torch.allclose(result, expected)

    def test_zero_betas(self):
        betas = torch.zeros(2, 5)
        shape_disps = torch.randn(10, 3, 5)
        result = blend_shapes(betas, shape_disps)
        assert result.shape == (2, 10, 3)
        assert torch.allclose(result, torch.zeros(2, 10, 3))

    def test_batch(self):
        betas = torch.randn(8, 10)
        shape_disps = torch.randn(100, 3, 10)
        result = blend_shapes(betas, shape_disps)
        assert result.shape == (8, 100, 3)


class TestTransformMat:
    def test_identity(self):
        R = torch.eye(3).unsqueeze(0)  # (1, 3, 3)
        t = torch.zeros(1, 3, 1)  # (1, 3, 1)
        T = transform_mat(R, t)
        assert T.shape == (1, 4, 4)
        assert torch.allclose(T[0], torch.eye(4))

    def test_with_translation(self):
        R = torch.eye(3).unsqueeze(0)
        t = torch.tensor([[[1.0], [2.0], [3.0]]])  # (1, 3, 1)
        T = transform_mat(R, t)
        assert T[0, 0, 3] == 1.0
        assert T[0, 1, 3] == 2.0
        assert T[0, 2, 3] == 3.0
        assert T[0, 3, 3] == 1.0

    def test_batch(self):
        R = torch.eye(3).unsqueeze(0).repeat(3, 1, 1)
        t = torch.randn(3, 3, 1)
        T = transform_mat(R, t)
        assert T.shape == (3, 4, 4)


class TestBatchRigidTransform:
    def test_single_joint_identity(self):
        rot_mats = torch.eye(3).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)
        joints = torch.tensor([[[0.0, 0.0, 0.0]]])  # (1, 1, 3)
        parents = torch.tensor([0])
        posed_joints, rel_transforms = batch_rigid_transform(
            rot_mats, joints, parents
        )
        assert posed_joints.shape == (1, 1, 3)
        assert rel_transforms.shape == (1, 1, 4, 4)

    def test_two_joints(self):
        rot_mats = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(1, 2, 1, 1)
        joints = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
        parents = torch.tensor([0, 0])
        posed_joints, rel_transforms = batch_rigid_transform(
            rot_mats, joints, parents
        )
        assert posed_joints.shape == (1, 2, 3)
        assert rel_transforms.shape == (1, 2, 4, 4)
        # With identity rotation, joints should stay the same
        assert torch.allclose(posed_joints[0, 0], torch.tensor([0.0, 0.0, 0.0]))
        assert torch.allclose(posed_joints[0, 1], torch.tensor([1.0, 0.0, 0.0]))
