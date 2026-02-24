# tests/test_run_test_plan.py
import importlib
import json
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch as mock_patch

import pytest


MOD = "cli.lib.core.vllm.lib"

# We import inside tests so the MOD override above applies everywhere
run_test_plan_import_path = f"{MOD}.run_test_plan"


def _get_cmd(c):
    # Support both kwargs and positional args
    return c.kwargs.get("cmd", c.args[0] if c.args else None)


def _get_check(c):
    if "check" in c.kwargs:
        return c.kwargs["check"]
    # If positional, assume second arg is 'check' when present; default False
    return c.args[1] if len(c.args) > 1 else False


@pytest.fixture
def patch_module(monkeypatch):
    """
    Patch helpers ('pip_install_packages', 'temp_environ', 'working_directory',
    'run_command', 'logger', '_load_disabled_vllm_tests') inside the target
    module and expose them.
    """
    module = importlib.import_module(MOD)

    # Create fakes/mocks
    pip_install_packages = MagicMock(name="pip_install_packages")
    run_command = MagicMock(name="run_command", return_value=0)
    disabled_mock = MagicMock(name="_load_disabled_vllm_tests", return_value=[])

    # temp_environ / working_directory: record calls but act as context managers
    temp_calls: list[dict] = []
    workdir_calls: list[str] = []

    def fake_working_directory(path: str):
        workdir_calls.append(path)
        return nullcontext()

    def fake_temp_env(map: dict[str, str]):
        temp_calls.append(map)
        return nullcontext()

    logger = SimpleNamespace(
        info=MagicMock(name="logger.info"),
        warning=MagicMock(name="logger.warning"),
        error=MagicMock(name="logger.error"),
    )

    # Apply patches (raise if attribute doesn't exist)
    monkeypatch.setattr(
        module, "pip_install_packages", pip_install_packages, raising=True
    )
    monkeypatch.setattr(module, "run_command", run_command, raising=True)
    monkeypatch.setattr(
        module, "working_directory", fake_working_directory, raising=True
    )
    monkeypatch.setattr(module, "temp_environ", fake_temp_env, raising=True)
    monkeypatch.setattr(module, "logger", logger, raising=True)
    monkeypatch.setattr(
        module, "_load_disabled_vllm_tests", disabled_mock, raising=True
    )

    return SimpleNamespace(
        module=module,
        run_test_plan=module.run_test_plan,  # expose to avoid getattr("constant") (Ruff B009)
        pip_install_packages=pip_install_packages,
        run_command=run_command,
        temp_calls=temp_calls,
        workdir_calls=workdir_calls,
        logger=logger,
        disabled_mock=disabled_mock,
    )


def test_success_runs_all_steps_and_uses_env_and_workdir(monkeypatch, patch_module):
    run_test_plan = patch_module.run_test_plan

    tests_map = {
        "basic": {
            "title": "Basic suite",
            "package_install": [],
            "working_directory": "tests",
            "env_vars": {"GLOBAL_FLAG": "1"},
            "steps": [
                "export A=x && pytest -q",
                "export B=y && pytest -q tests/unit",
            ],
        }
    }

    # One exit code per step (export + two pytest)
    patch_module.run_command.side_effect = [0, 0, 0]

    run_test_plan("basic", "cpu", tests_map)

    calls = patch_module.run_command.call_args_list
    checks = [_get_check(c) for c in calls]

    if not all(chk is False for chk in checks):
        raise AssertionError(f"Expected all checks to be False, got checks={checks}")

    if patch_module.workdir_calls != ["tests"]:
        raise AssertionError(
            f"Expected workdir_calls=['tests'], got {patch_module.workdir_calls}"
        )
    if patch_module.temp_calls != [{"GLOBAL_FLAG": "1"}]:
        raise AssertionError(
            f"Expected temp_calls=[{{'GLOBAL_FLAG': '1'}}], got {patch_module.temp_calls}"
        )


def test_installs_packages_when_present(monkeypatch, patch_module):
    run_test_plan = patch_module.module.run_test_plan

    tests_map = {
        "with_pkgs": {
            "title": "Needs deps",
            "package_install": ["timm==1.0.0", "flash-attn"],
            "steps": ["pytest -q"],
        }
    }

    patch_module.run_command.return_value = 0

    run_test_plan("with_pkgs", "gpu", tests_map)

    patch_module.pip_install_packages.assert_called_once_with(
        packages=["timm==1.0.0", "flash-attn"],
        prefer_uv=True,
    )


def test_raises_on_missing_plan(patch_module):
    run_test_plan = patch_module.module.run_test_plan
    with pytest.raises(RuntimeError) as ei:
        run_test_plan("nope", "cpu", tests_map={})

    if "test nope not found" not in str(ei.value):
        raise AssertionError(
            f"Expected 'test nope not found' in error, got: {ei.value}"
        )


def test_aggregates_failures_and_raises(monkeypatch, patch_module):
    run_test_plan = patch_module.module.run_test_plan

    tests_map = {
        "mix": {
            "title": "Some pass some fail",
            "steps": [
                "pytest test_a.py",  # 0 → pass
                "pytest test_b.py",  # 1 → fail
                "pytest test_c.py",  # 2 → fail
            ],
        }
    }

    # Simulate pass, fail, fail
    patch_module.run_command.side_effect = [0, 1, 2]

    with pytest.raises(RuntimeError) as ei:
        run_test_plan("mix", "cpu", tests_map)

    msg = str(ei.value)
    if "2 pytest runs failed" not in msg:
        raise AssertionError(f"Expected '2 pytest runs failed' in error, got: {msg}")
    # Ensure logger captured failed tests list
    patch_module.logger.error.assert_called_once()
    # And we attempted all three commands
    if patch_module.run_command.call_count != 3:
        raise AssertionError(
            f"Expected run_command.call_count=3, got {patch_module.run_command.call_count}"
        )


def test_custom_working_directory_used(patch_module):
    run_test_plan = patch_module.module.run_test_plan

    tests_map = {
        "customwd": {
            "title": "Custom wd",
            "working_directory": "examples/ci",
            "steps": ["pytest -q"],
        }
    }

    patch_module.run_command.return_value = 0
    run_test_plan("customwd", "cpu", tests_map)

    if patch_module.workdir_calls != ["examples/ci"]:
        raise AssertionError(
            f"Expected workdir_calls=['examples/ci'], got {patch_module.workdir_calls}"
        )


# -- Disabled vLLM test injection (integration tests) -------------------------

_SIMPLE_TESTS_MAP = {
    "plan_a": {
        "title": "Plan A",
        "steps": ["pytest -v -s test_foo.py", "pytest -v -s test_bar.py"],
    }
}


def _cmds(patch_module):
    return [_get_cmd(c) for c in patch_module.run_command.call_args_list]


def test_deselect_injected_for_test_level(patch_module):
    patch_module.disabled_mock.return_value = [
        {"test": "test_foo.py::test_x", "issue": "http://issue/1"}
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    cmds = _cmds(patch_module)
    if not any("--deselect=test_foo.py::test_x" in c for c in cmds):
        raise AssertionError(
            f"Expected --deselect=test_foo.py::test_x in commands, got {cmds}"
        )


def test_ignore_injected_for_file_level(patch_module):
    patch_module.disabled_mock.return_value = [
        {"test": "test_foo.py", "issue": "http://issue/1"}
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    cmds = _cmds(patch_module)
    if not any("--ignore=test_foo.py" in c for c in cmds):
        raise AssertionError(f"Expected --ignore=test_foo.py in commands, got {cmds}")


def test_mixed_disabled_flags(patch_module):
    patch_module.disabled_mock.return_value = [
        {"test": "test_foo.py", "issue": "http://issue/1"},
        {"test": "test_bar.py::test_y", "issue": "http://issue/2"},
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    cmds = _cmds(patch_module)
    if not any("--ignore=test_foo.py" in c for c in cmds):
        raise AssertionError(f"Expected --ignore=test_foo.py in commands, got {cmds}")
    if not any("--deselect=test_bar.py::test_y" in c for c in cmds):
        raise AssertionError(
            f"Expected --deselect=test_bar.py::test_y in commands, got {cmds}"
        )


def test_empty_disabled_list_no_modification(patch_module):
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    for cmd in _cmds(patch_module):
        if "--ignore" in cmd or "--deselect" in cmd:
            raise AssertionError(
                f"Expected no --ignore/--deselect flags, got {cmd}"
            )


def test_disabled_flags_applied_to_every_pytest_step(patch_module):
    patch_module.disabled_mock.return_value = [
        {"test": "skip.py", "issue": "http://issue/1"}
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    for cmd in _cmds(patch_module):
        if "--ignore=skip.py" not in cmd:
            raise AssertionError(
                f"Expected --ignore=skip.py in every command, got {cmd}"
            )


def test_non_pytest_steps_not_modified(patch_module):
    tests_map = {
        "mixed_steps": {
            "title": "Mixed",
            "steps": ["echo hello", "pytest -v test.py"],
        }
    }
    patch_module.disabled_mock.return_value = [
        {"test": "test.py::test_z", "issue": "http://issue/1"}
    ]
    patch_module.run_test_plan("mixed_steps", "cpu", tests_map)
    cmds = _cmds(patch_module)
    if "--deselect" in cmds[0]:
        raise AssertionError(
            f"Non-pytest step should not have --deselect, got {cmds[0]}"
        )
    if "--deselect=test.py::test_z" not in cmds[1]:
        raise AssertionError(
            f"Expected --deselect=test.py::test_z in pytest step, got {cmds[1]}"
        )


def test_configs_matching_included(patch_module):
    patch_module.disabled_mock.return_value = [
        {"test": "test_foo.py", "issue": "http://issue/1", "configs": ["plan_a"]}
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    cmds = _cmds(patch_module)
    if not any("--ignore=test_foo.py" in c for c in cmds):
        raise AssertionError(f"Expected --ignore=test_foo.py in commands, got {cmds}")


def test_configs_non_matching_excluded(patch_module):
    patch_module.disabled_mock.return_value = [
        {"test": "test_foo.py", "issue": "http://issue/1", "configs": ["other_plan"]}
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    for cmd in _cmds(patch_module):
        if "--ignore" in cmd:
            raise AssertionError(
                f"Expected no --ignore flags for non-matching config, got {cmd}"
            )


def test_configs_absent_means_all(patch_module):
    patch_module.disabled_mock.return_value = [
        {"test": "test_foo.py", "issue": "http://issue/1"}
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    cmds = _cmds(patch_module)
    if not any("--ignore=test_foo.py" in c for c in cmds):
        raise AssertionError(f"Expected --ignore=test_foo.py in commands, got {cmds}")


def test_disabled_and_rerun_flags_both_present(patch_module):
    patch_module.disabled_mock.return_value = [
        {"test": "skip.py::test_x", "issue": "http://issue/1"}
    ]
    patch_module.run_test_plan("plan_a", "cpu", _SIMPLE_TESTS_MAP)
    for cmd in _cmds(patch_module):
        if "--deselect=skip.py::test_x" not in cmd:
            raise AssertionError(f"Expected --deselect=skip.py::test_x, got {cmd}")
        if "--reruns" not in cmd:
            raise AssertionError(f"Expected --reruns in command, got {cmd}")
        if cmd.count("pytest") != 1:
            raise AssertionError(f"Expected exactly one 'pytest' token, got {cmd}")


# -- Disabled vLLM test helpers (unit tests) -----------------------------------


def test_build_flags_deselect():
    module = importlib.import_module(MOD)
    flags = module._build_disabled_test_flags(
        [{"test": "a.py::test_1", "issue": "url"}], "plan"
    )
    if flags != "--deselect=a.py::test_1":
        raise AssertionError(f"Expected '--deselect=a.py::test_1', got '{flags}'")


def test_build_flags_ignore():
    module = importlib.import_module(MOD)
    flags = module._build_disabled_test_flags(
        [{"test": "a.py", "issue": "url"}], "plan"
    )
    if flags != "--ignore=a.py":
        raise AssertionError(f"Expected '--ignore=a.py', got '{flags}'")


def test_build_flags_config_filter():
    module = importlib.import_module(MOD)
    entries = [
        {"test": "a.py", "issue": "url", "configs": ["plan_x"]},
        {"test": "b.py::test_1", "issue": "url"},
    ]
    flags = module._build_disabled_test_flags(entries, "plan_x")
    if "--ignore=a.py" not in flags:
        raise AssertionError(f"Expected '--ignore=a.py' in '{flags}'")
    if "--deselect=b.py::test_1" not in flags:
        raise AssertionError(f"Expected '--deselect=b.py::test_1' in '{flags}'")


def test_build_flags_config_excludes():
    module = importlib.import_module(MOD)
    entries = [{"test": "a.py", "issue": "url", "configs": ["other"]}]
    flags = module._build_disabled_test_flags(entries, "plan_x")
    if flags != "":
        raise AssertionError(f"Expected empty flags, got '{flags}'")


def test_parse_issue_body():
    module = importlib.import_module(MOD)
    body = (
        "## Disabled vLLM Tests\n"
        "```yaml\n"
        "disabled_tests:\n"
        "  - test: foo.py::test_bar\n"
        "  - test: baz.py\n"
        "    configs:\n"
        "      - plan_a\n"
        "```\n"
    )
    entries = module._parse_disabled_tests_from_issue_body(body)
    if len(entries) != 2:
        raise AssertionError(f"Expected 2 entries, got {len(entries)}")
    if entries[0]["test"] != "foo.py::test_bar":
        raise AssertionError(f"Expected 'foo.py::test_bar', got '{entries[0]['test']}'")
    if entries[1]["test"] != "baz.py":
        raise AssertionError(f"Expected 'baz.py', got '{entries[1]['test']}'")
    if entries[1]["configs"] != ["plan_a"]:
        raise AssertionError(f"Expected ['plan_a'], got {entries[1]['configs']}")


def test_parse_issue_body_no_yaml():
    module = importlib.import_module(MOD)
    entries = module._parse_disabled_tests_from_issue_body("no yaml here")
    if entries != []:
        raise AssertionError(f"Expected [], got {entries}")


def test_load_yaml_valid_entries():
    module = importlib.import_module(MOD)
    yaml_content = (
        "disabled_tests:\n"
        "  - test: a.py\n"
        "    issue: https://github.com/pytorch/pytorch/issues/1\n"
        "  - test: b.py::test_x\n"
        "    issue: https://github.com/pytorch/pytorch/issues/2\n"
        "    configs:\n"
        "      - plan_a\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(yaml_content)
        tmp_path = Path(f.name)
    try:
        with mock_patch.object(module, "_DISABLED_VLLM_TESTS_PATH", tmp_path):
            entries = module._load_disabled_vllm_tests_from_yaml()
        expected_first = {
            "test": "a.py",
            "issue": "https://github.com/pytorch/pytorch/issues/1",
        }
        if len(entries) != 2:
            raise AssertionError(f"Expected 2 entries, got {len(entries)}")
        if entries[0] != expected_first:
            raise AssertionError(f"Unexpected first entry: {entries[0]}")
        if entries[1]["test"] != "b.py::test_x":
            raise AssertionError(
                f"Expected 'b.py::test_x', got '{entries[1]['test']}'"
            )
        if entries[1]["configs"] != ["plan_a"]:
            raise AssertionError(f"Expected ['plan_a'], got {entries[1]['configs']}")
    finally:
        tmp_path.unlink()


def test_load_yaml_missing_keys():
    module = importlib.import_module(MOD)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("disabled_tests:\n  - test: foo.py\n")
        tmp_path = Path(f.name)
    try:
        with mock_patch.object(module, "_DISABLED_VLLM_TESTS_PATH", tmp_path):
            with pytest.raises(ValueError, match="must have 'test' and 'issue' keys"):
                module._load_disabled_vllm_tests_from_yaml()
    finally:
        tmp_path.unlink()


def test_load_yaml_missing_file():
    module = importlib.import_module(MOD)
    with mock_patch.object(
        module, "_DISABLED_VLLM_TESTS_PATH", Path("/nonexistent/nope.yaml")
    ):
        result = module._load_disabled_vllm_tests_from_yaml()
    if result != []:
        raise AssertionError(f"Expected [], got {result}")


def test_load_github_skipped_when_issue_unset():
    module = importlib.import_module(MOD)
    with mock_patch.object(module, "_DISABLED_VLLM_TESTS_ISSUE", 0):
        result = module._load_disabled_vllm_tests_from_github()
    if result != []:
        raise AssertionError(f"Expected [], got {result}")


def test_load_github_failure_returns_empty():
    module = importlib.import_module(MOD)

    def _raise(*args, **kwargs):
        raise OSError("network error")

    with (
        mock_patch.object(module, "_DISABLED_VLLM_TESTS_ISSUE", 99999),
        mock_patch.object(module.urllib.request, "urlopen", _raise),
    ):
        result = module._load_disabled_vllm_tests_from_github()
    if result != []:
        raise AssertionError(f"Expected [], got {result}")


def test_load_github_filters_malformed_entries():
    module = importlib.import_module(MOD)
    issue_body = (
        "```yaml\n"
        "disabled_tests:\n"
        "  - bad_key: oops\n"
        "  - test: good.py\n"
        "```"
    )
    response_bytes = json.dumps(
        {"body": issue_body, "html_url": "https://github.com/issue/99999"}
    ).encode()

    class FakeResponse:
        def read(self):
            return response_bytes

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    with (
        mock_patch.object(module, "_DISABLED_VLLM_TESTS_ISSUE", 99999),
        mock_patch.object(
            module.urllib.request,
            "urlopen",
            lambda *a, **kw: FakeResponse(),
        ),
    ):
        entries = module._load_disabled_vllm_tests_from_github()

    if len(entries) != 1:
        raise AssertionError(f"Expected 1 entry, got {len(entries)}")
    if entries[0]["test"] != "good.py":
        raise AssertionError(f"Expected 'good.py', got '{entries[0]['test']}'")
    if entries[0]["issue"] != "https://github.com/issue/99999":
        raise AssertionError(f"Expected issue URL, got '{entries[0]['issue']}'")


def test_deduplication_yaml_wins():
    module = importlib.import_module(MOD)
    yaml_entries = [{"test": "a.py", "issue": "yaml-url"}]
    github_entries = [
        {"test": "a.py", "issue": "github-url"},
        {"test": "b.py::test_1", "issue": "github-url"},
    ]
    with (
        mock_patch.object(
            module,
            "_load_disabled_vllm_tests_from_yaml",
            return_value=yaml_entries,
        ),
        mock_patch.object(
            module,
            "_load_disabled_vllm_tests_from_github",
            return_value=github_entries,
        ),
    ):
        merged = module._load_disabled_vllm_tests()

    if len(merged) != 2:
        raise AssertionError(f"Expected 2 merged entries, got {len(merged)}")
    if merged[0] != {"test": "a.py", "issue": "yaml-url"}:
        raise AssertionError(f"Expected yaml entry to win, got {merged[0]}")
    if merged[1] != {"test": "b.py::test_1", "issue": "github-url"}:
        raise AssertionError(f"Unexpected second entry: {merged[1]}")
