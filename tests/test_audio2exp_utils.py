"""Tests for audio2exp-service utilities - registry, misc, path."""

import os
import pytest
import tempfile
import numpy as np
import torch
from pathlib import Path

from LAM_Audio2Expression.utils.misc import (
    AverageMeter,
    intersection_and_union,
    find_free_port,
    is_seq_of,
    is_str,
    import_modules_from_strings,
)
from LAM_Audio2Expression.utils.path import (
    is_filepath,
    check_file_exist,
    mkdir_or_exist,
    scandir,
    find_vcs_root,
)
from LAM_Audio2Expression.utils.registry import Registry, build_from_cfg


class TestAverageMeter:
    def test_init(self):
        meter = AverageMeter()
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0

    def test_single_update(self):
        meter = AverageMeter()
        meter.update(10)
        assert meter.val == 10
        assert meter.sum == 10
        assert meter.count == 1
        assert meter.avg == 10.0

    def test_multiple_updates(self):
        meter = AverageMeter()
        meter.update(10)
        meter.update(20)
        meter.update(30)
        assert meter.val == 30
        assert meter.sum == 60
        assert meter.count == 3
        assert meter.avg == 20.0

    def test_weighted_update(self):
        meter = AverageMeter()
        meter.update(10, n=3)  # 3 samples with value 10
        meter.update(20, n=2)  # 2 samples with value 20
        assert meter.sum == 70  # 30 + 40
        assert meter.count == 5
        assert meter.avg == 14.0

    def test_reset(self):
        meter = AverageMeter()
        meter.update(100)
        meter.reset()
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0


class TestIntersectionAndUnion:
    def test_perfect_match(self):
        output = np.array([0, 1, 2, 0, 1])
        target = np.array([0, 1, 2, 0, 1])
        inter, union, area_target = intersection_and_union(output, target, K=3)
        assert np.array_equal(inter, np.array([2, 2, 1]))
        assert np.array_equal(union, np.array([2, 2, 1]))

    def test_no_match(self):
        output = np.array([0, 0, 0])
        target = np.array([1, 1, 1])
        inter, union, area_target = intersection_and_union(output, target, K=2)
        assert inter[0] == 0
        assert inter[1] == 0

    def test_with_ignore_index(self):
        output = np.array([0, 1, 2, 0])
        target = np.array([0, 1, -1, 0])
        inter, union, area_target = intersection_and_union(
            output, target, K=3, ignore_index=-1
        )
        # The element at ignore_index position should be excluded
        assert inter[2] == 0  # no valid class-2 intersection


class TestFindFreePort:
    def test_returns_valid_port(self):
        port = find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535


class TestIsSeqOf:
    def test_list_of_ints(self):
        assert is_seq_of([1, 2, 3], int) is True

    def test_list_of_strings(self):
        assert is_seq_of(["a", "b"], str) is True

    def test_mixed_types(self):
        assert is_seq_of([1, "a"], int) is False

    def test_empty_sequence(self):
        assert is_seq_of([], int) is True

    def test_tuple(self):
        assert is_seq_of((1, 2, 3), int) is True

    def test_not_sequence(self):
        assert is_seq_of(42, int) is False

    def test_with_seq_type(self):
        assert is_seq_of([1, 2], int, seq_type=list) is True
        assert is_seq_of((1, 2), int, seq_type=list) is False


class TestIsStr:
    def test_string(self):
        assert is_str("hello") is True

    def test_not_string(self):
        assert is_str(42) is False
        assert is_str(None) is False
        assert is_str([]) is False


class TestImportModulesFromStrings:
    def test_single_import(self):
        result = import_modules_from_strings("os.path")
        import os.path
        assert result is os.path

    def test_list_import(self):
        result = import_modules_from_strings(["os.path", "sys"])
        assert len(result) == 2

    def test_none_input(self):
        result = import_modules_from_strings(None)
        assert result is None

    def test_empty_string(self):
        result = import_modules_from_strings("")
        assert result is None

    def test_failed_import_raises(self):
        with pytest.raises(ImportError):
            import_modules_from_strings("nonexistent_module_xyz")

    def test_failed_import_allowed(self):
        result = import_modules_from_strings(
            "nonexistent_module_xyz", allow_failed_imports=True
        )
        assert result is None

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            import_modules_from_strings(42)


class TestIsFilepath:
    def test_string_path(self):
        assert is_filepath("/tmp/test.txt") is True

    def test_path_object(self):
        assert is_filepath(Path("/tmp/test.txt")) is True

    def test_not_filepath(self):
        assert is_filepath(42) is False
        assert is_filepath(None) is False


class TestCheckFileExist:
    def test_existing_file(self):
        with tempfile.NamedTemporaryFile() as f:
            check_file_exist(f.name)  # Should not raise

    def test_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            check_file_exist("/nonexistent/file/path.txt")

    def test_custom_message(self):
        with pytest.raises(FileNotFoundError, match="custom"):
            check_file_exist("/nonexistent", msg_tmpl='custom error: "{}"')


class TestMkdirOrExist:
    def test_create_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            new_dir = os.path.join(tmpdir, "subdir")
            mkdir_or_exist(new_dir)
            assert os.path.isdir(new_dir)

    def test_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mkdir_or_exist(tmpdir)  # Should not raise

    def test_empty_string(self):
        mkdir_or_exist("")  # Should return without error


class TestScandir:
    def test_scan_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            for name in ["a.txt", "b.py", "c.txt"]:
                open(os.path.join(tmpdir, name), "w").close()
            files = list(scandir(tmpdir))
            assert len(files) == 3

    def test_scan_with_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.txt", "b.py", "c.txt"]:
                open(os.path.join(tmpdir, name), "w").close()
            files = list(scandir(tmpdir, suffix=".txt"))
            assert len(files) == 2

    def test_scan_recursive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            os.makedirs(os.path.join(tmpdir, "sub"))
            open(os.path.join(tmpdir, "a.txt"), "w").close()
            open(os.path.join(tmpdir, "sub", "b.txt"), "w").close()
            files = list(scandir(tmpdir, recursive=True))
            assert len(files) == 2

    def test_scan_case_insensitive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for name in ["a.TXT", "b.txt", "c.py"]:
                open(os.path.join(tmpdir, name), "w").close()
            files = list(scandir(tmpdir, suffix=".txt", case_sensitive=False))
            assert len(files) == 2

    def test_invalid_dir_path_type(self):
        with pytest.raises(TypeError):
            list(scandir(123))


class TestFindVcsRoot:
    def test_find_git_root(self):
        # The project root has a .git directory
        root = find_vcs_root("/home/user/LAM_gpro")
        assert root is not None
        assert os.path.isdir(os.path.join(root, ".git"))

    def test_not_found(self):
        root = find_vcs_root("/tmp", markers=(".nonexistent_marker",))
        assert root is None


class TestAdvancedRegistry:
    def test_init(self):
        reg = Registry("test_reg", scope="test")
        assert reg.name == "test_reg"
        assert reg.scope == "test"
        assert len(reg) == 0

    def test_register_module_decorator(self):
        reg = Registry("test", scope="test")

        @reg.register_module()
        class MyClass:
            pass

        assert "MyClass" in reg
        assert reg.get("MyClass") is MyClass

    def test_register_module_with_name(self):
        reg = Registry("test", scope="test")

        @reg.register_module(name="custom_name")
        class MyClass:
            pass

        assert "custom_name" in reg
        assert reg.get("custom_name") is MyClass

    def test_register_module_direct(self):
        reg = Registry("test", scope="test")

        class MyClass:
            pass

        reg.register_module(module=MyClass)
        assert "MyClass" in reg

    def test_register_duplicate_raises(self):
        reg = Registry("test", scope="test")

        @reg.register_module()
        class MyClass:
            pass

        with pytest.raises(KeyError, match="already registered"):
            reg.register_module(module=MyClass)

    def test_register_force(self):
        reg = Registry("test", scope="test")

        @reg.register_module()
        class MyClass:
            val = 1

        class MyClass2:
            val = 2

        reg.register_module(name="MyClass", module=MyClass2, force=True)
        assert reg.get("MyClass").val == 2

    def test_split_scope_key(self):
        scope, key = Registry.split_scope_key("mmdet.ResNet")
        assert scope == "mmdet"
        assert key == "ResNet"

    def test_split_scope_key_no_scope(self):
        scope, key = Registry.split_scope_key("ResNet")
        assert scope is None
        assert key == "ResNet"

    def test_contains_false(self):
        reg = Registry("test", scope="test")
        assert "nonexistent" not in reg

    def test_repr(self):
        reg = Registry("test_reg", scope="test")
        repr_str = repr(reg)
        assert "test_reg" in repr_str

    def test_register_non_class_raises(self):
        reg = Registry("test", scope="test")
        with pytest.raises(TypeError, match="module must be a class"):
            reg._register_module("not_a_class")

    def test_force_must_be_bool(self):
        reg = Registry("test", scope="test")
        with pytest.raises(TypeError, match="force must be a boolean"):
            reg.register_module(force="yes")


class TestBuildFromCfg:
    def test_basic_build(self):
        reg = Registry("test", scope="test")

        @reg.register_module()
        class SimpleClass:
            def __init__(self, value=0):
                self.value = value

        obj = build_from_cfg({"type": "SimpleClass", "value": 42}, reg)
        assert obj.value == 42

    def test_build_with_default_args(self):
        reg = Registry("test", scope="test")

        @reg.register_module()
        class MyClass:
            def __init__(self, a=0, b=0):
                self.a = a
                self.b = b

        obj = build_from_cfg({"type": "MyClass", "a": 1}, reg, default_args={"b": 2})
        assert obj.a == 1
        assert obj.b == 2

    def test_build_cfg_not_dict_raises(self):
        reg = Registry("test", scope="test")
        with pytest.raises(TypeError, match="cfg must be a dict"):
            build_from_cfg("not_a_dict", reg)

    def test_build_missing_type_raises(self):
        reg = Registry("test", scope="test")
        with pytest.raises(KeyError, match="type"):
            build_from_cfg({"value": 42}, reg)

    def test_build_unknown_type_raises(self):
        reg = Registry("test", scope="test")
        with pytest.raises(KeyError, match="not in the"):
            build_from_cfg({"type": "Unknown"}, reg)

    def test_build_invalid_registry_raises(self):
        with pytest.raises(TypeError, match="registry must be"):
            build_from_cfg({"type": "X"}, "not_registry")

    def test_build_with_class_type(self):
        reg = Registry("test", scope="test")

        class DirectClass:
            def __init__(self, x=0):
                self.x = x

        obj = build_from_cfg({"type": DirectClass, "x": 99}, reg)
        assert obj.x == 99
