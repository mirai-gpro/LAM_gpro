"""Tests for vhap/config/base.py - Configuration dataclasses."""

import pytest
from pathlib import Path

from vhap.config.base import (
    Config,
    DataConfig,
    ModelConfig,
    RenderConfig,
    LearningRateConfig,
    LossWeightConfig,
    LogConfig,
    ExperimentConfig,
    StageConfig,
    StageLmkInitRigidConfig,
    StageLmkInitAllConfig,
)


class TestConfig:
    def test_getitem(self):
        cfg = DataConfig()
        assert cfg["sequence"] == ""
        assert cfg["calibrated"] is False

    def test_getitem_missing_raises(self):
        cfg = DataConfig()
        with pytest.raises(AttributeError, match="no attribute"):
            cfg["nonexistent_field"]


class TestDataConfig:
    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.sequence == ""
        assert cfg.calibrated is False
        assert cfg.align_cameras_to_axes is True
        assert cfg.camera_convention_conversion == "opencv->opengl"
        assert cfg.target_extrinsic_type == "w2c"
        assert cfg.scale_factor == 1.0
        assert cfg.background_color == "white"
        assert cfg.use_landmark is True
        assert cfg.landmark_source == "star"

    def test_custom_values(self):
        cfg = DataConfig(sequence="test_seq", calibrated=True, scale_factor=0.5)
        assert cfg.sequence == "test_seq"
        assert cfg.calibrated is True
        assert cfg.scale_factor == 0.5


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert cfg.n_shape == 300
        assert cfg.n_expr == 100
        assert cfg.n_tex == 100
        assert cfg.use_static_offset is False
        assert cfg.use_dynamic_offset is False
        assert cfg.add_teeth is True
        assert cfg.tex_resolution == 2048
        assert cfg.tex_painted is True
        assert cfg.tex_extra is True
        assert cfg.residual_tex is True
        assert cfg.occluded == ()

    def test_tex_clusters(self):
        cfg = ModelConfig()
        assert "skin" in cfg.tex_clusters
        assert "hair" in cfg.tex_clusters


class TestRenderConfig:
    def test_defaults(self):
        cfg = RenderConfig()
        assert cfg.backend == "nvdiffrast"
        assert cfg.use_opengl is False
        assert cfg.background_train == "target"
        assert cfg.background_eval == "target"
        assert cfg.lighting_type == "SH"
        assert cfg.lighting_space == "world"
        assert cfg.disturb_rate_fg == 0.5
        assert cfg.disturb_rate_bg == 0.5


class TestLearningRateConfig:
    def test_defaults(self):
        cfg = LearningRateConfig()
        assert cfg.base == 5e-3
        assert cfg.translation == 1e-3
        assert cfg.expr == 5e-2
        assert cfg.static_offset == 5e-4
        assert cfg.dynamic_offset == 5e-4


class TestLossWeightConfig:
    def test_defaults(self):
        cfg = LossWeightConfig()
        assert cfg.landmark == 10.0
        assert cfg.photo == 30.0
        assert cfg.reg_shape == 3e-1
        assert cfg.reg_expr == 3e-2
        assert cfg.always_enable_jawline_landmarks is True

    def test_reg_tex_res_for_defaults(self):
        cfg = LossWeightConfig()
        assert "sclerae" in cfg.reg_tex_res_for
        assert "teeth" in cfg.reg_tex_res_for


class TestLogConfig:
    def test_defaults(self):
        cfg = LogConfig()
        assert cfg.interval_scalar == 100
        assert cfg.interval_media == 500
        assert cfg.image_format == "jpg"
        assert cfg.max_num_views == 3
        assert cfg.stack_views_in_rows is True


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.output_folder == Path("output/track")
        assert cfg.reuse_landmarks is True
        assert cfg.keyframes == ()
        assert cfg.photometric is False


class TestStageConfigs:
    def test_stage_lmk_init_rigid(self):
        cfg = StageLmkInitRigidConfig()
        assert cfg.num_steps == 300
        assert "cam" in cfg.optimizable_params
        assert "pose" in cfg.optimizable_params
        assert cfg.disable_jawline_landmarks is False

    def test_stage_lmk_init_all(self):
        cfg = StageLmkInitAllConfig()
        assert cfg.num_steps == 300
        assert "shape" in cfg.optimizable_params
        assert "expr" in cfg.optimizable_params
