"""Tests for core.naming — unified naming module."""

import pytest
from core.naming import sanitize_name, to_snake, to_pascal, slugify


# =============================================================================
# sanitize_name
# =============================================================================

class TestSanitizeName:

    def test_strips_path_prefix(self):
        assert sanitize_name("/build") == "build"

    def test_strips_deep_path(self):
        assert sanitize_name("~/.motherlabs/inputHistory") == "inputHistory"

    def test_strips_special_chars(self):
        assert sanitize_name("mother>Prompt") == "mother Prompt"

    def test_collapses_whitespace(self):
        assert sanitize_name("  task   manager  ") == "task manager"

    def test_empty_fallback(self):
        assert sanitize_name("") == "component"
        assert sanitize_name("///") == "component"
        assert sanitize_name(">>>") == "component"

    def test_preserves_alphanumeric(self):
        assert sanitize_name("TaskManager") == "TaskManager"

    def test_preserves_underscores_and_hyphens(self):
        assert sanitize_name("task-manager_v2") == "task-manager_v2"


# =============================================================================
# to_snake
# =============================================================================

class TestToSnake:

    def test_pascal_case(self):
        assert to_snake("TaskManager") == "task_manager"

    def test_single_word(self):
        assert to_snake("Task") == "task"

    def test_already_snake(self):
        assert to_snake("task_manager") == "task_manager"

    def test_spaces(self):
        assert to_snake("Task Manager") == "task_manager"

    def test_hyphens(self):
        assert to_snake("task-manager") == "task_manager"

    def test_path_prefix(self):
        assert to_snake("/build") == "build"

    def test_special_chars(self):
        assert to_snake("mother>Prompt") == "mother_prompt"

    def test_consecutive_uppercase(self):
        assert to_snake("HTMLParser") == "htmlparser"

    def test_empty_fallback(self):
        assert to_snake("") == "component"

    def test_mixed_separators(self):
        assert to_snake("task_Manager-v2") == "task_manager_v2"


# =============================================================================
# to_pascal
# =============================================================================

class TestToPascal:

    def test_already_pascal(self):
        assert to_pascal("TaskManager") == "TaskManager"

    def test_from_spaces(self):
        assert to_pascal("task manager") == "TaskManager"

    def test_from_snake(self):
        assert to_pascal("task_manager") == "TaskManager"

    def test_single_word(self):
        assert to_pascal("task") == "Task"

    def test_path_prefix(self):
        assert to_pascal("/build") == "Build"

    def test_special_chars(self):
        assert to_pascal("mother>Prompt") == "MotherPrompt"

    def test_from_hyphens(self):
        assert to_pascal("task-manager") == "TaskManager"


# =============================================================================
# slugify
# =============================================================================

class TestSlugify:

    def test_basic(self):
        assert slugify("Task Management") == "task_management"

    def test_special_chars(self):
        assert slugify("E-Commerce (v2)") == "e_commerce_v2"

    def test_empty(self):
        assert slugify("") == "project"

    def test_starts_with_number(self):
        result = slugify("123app")
        assert result.startswith("p_")
        assert result[0].isalpha()

    def test_lowercase(self):
        assert slugify("myapp") == "myapp"

    def test_strips_trailing_underscores(self):
        assert slugify("  myapp  ") == "myapp"


# =============================================================================
# Consistency: same input → same output across all functions
# =============================================================================

class TestConsistency:
    """Verify naming functions produce consistent, composable results."""

    @pytest.mark.parametrize("name", [
        "TaskManager", "task_manager", "task-manager", "Task Manager",
        "/build", "mother>Prompt", "~/.motherlabs/inputHistory",
    ])
    def test_to_snake_idempotent_after_first(self, name):
        """to_snake(to_snake(x)) == to_snake(x)."""
        once = to_snake(name)
        twice = to_snake(once)
        assert once == twice

    @pytest.mark.parametrize("name", [
        "TaskManager", "task_manager", "task-manager", "Task Manager",
    ])
    def test_to_pascal_idempotent_after_first(self, name):
        """to_pascal(to_pascal(x)) == to_pascal(x)."""
        once = to_pascal(name)
        twice = to_pascal(once)
        assert once == twice
