"""Tests for lam/utils/registry.py - Registry pattern."""

import pytest
from lam.utils.registry import Registry


class TestRegistry:
    def test_init_empty(self):
        reg = Registry()
        assert len(reg._registry) == 0

    def test_register_class(self):
        reg = Registry()

        @reg.register("my_module")
        class MyModule:
            pass

        assert "my_module" in reg
        assert reg["my_module"] is MyModule

    def test_register_multiple(self):
        reg = Registry()

        @reg.register("module_a")
        class A:
            pass

        @reg.register("module_b")
        class B:
            pass

        assert "module_a" in reg
        assert "module_b" in reg
        assert reg["module_a"] is A
        assert reg["module_b"] is B

    def test_register_duplicate_raises(self):
        reg = Registry()

        @reg.register("dup")
        class First:
            pass

        with pytest.raises(AssertionError, match="already registered"):

            @reg.register("dup")
            class Second:
                pass

    def test_getitem_missing_raises(self):
        reg = Registry()
        with pytest.raises(KeyError):
            reg["nonexistent"]

    def test_contains_false(self):
        reg = Registry()
        assert "anything" not in reg

    def test_register_function(self):
        reg = Registry()

        @reg.register("my_func")
        def my_func():
            return 42

        assert "my_func" in reg
        assert reg["my_func"]() == 42

    def test_registered_class_is_unchanged(self):
        reg = Registry()

        @reg.register("cls")
        class Original:
            value = 10

        assert Original.value == 10
        assert reg["cls"].value == 10
