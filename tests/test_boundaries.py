"""AST and runtime subprocess enforcement of architectural import boundaries."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENGINE_DIR = ROOT / "castle_steward_engine"
CODING_DIR = ROOT / "castle_steward_coding"
STDLIB = sys.stdlib_module_names | {"__future__"}


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((0, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.append((node.level, node.module or ""))
    return found


def test_engine_never_imports_coding_or_workflows() -> None:
    for path in sorted(ENGINE_DIR.rglob("*.py")):
        for level, mod in _imports(path):
            assert level <= 1, f"{path} has escaping relative import level={level}"
            assert not mod.startswith("castle_steward_coding"), f"{path} imports {mod}"


def test_workflow_sdk_only_imports_stdlib_and_pydantic() -> None:
    wf_path = ENGINE_DIR / "workflow.py"
    forbidden = {"langgraph", "mcp", "sqlite3", "sqlalchemy", "sqlmodel", "alembic", "subprocess"}
    for level, mod in _imports(wf_path):
        assert level == 0, f"workflow.py must not use relative imports (level={level}, {mod!r})"
        top = mod.split(".")[0]
        assert top not in forbidden, f"workflow.py must not import {mod}"
        assert top in STDLIB or top == "pydantic", f"workflow.py unexpected import: {mod}"


def test_mcp_server_never_imports_sqlite_or_langgraph() -> None:
    for level, mod in _imports(ENGINE_DIR / "mcp_server.py"):
        assert level == 0, f"mcp_server.py must not use relative imports (level={level})"
        assert mod.split(".")[0] not in {
            "sqlite3",
            "sqlalchemy",
            "sqlmodel",
            "alembic",
            "langgraph",
        }, f"mcp_server.py imports {mod}"


def test_store_never_imports_sqlite3() -> None:
    for _level, mod in _imports(ENGINE_DIR / "store.py"):
        assert mod.split(".")[0] not in {"sqlite3", "threading"}, f"store.py imports {mod}"
        assert not mod.startswith("castle_steward_engine.migrations"), f"store.py imports {mod}"


def test_coding_only_imports_public_engine_sdk_stdlib_and_pydantic() -> None:
    coding_files = sorted(CODING_DIR.rglob("*.py"))
    assert CODING_DIR / "models.py" in coding_files
    assert CODING_DIR / "routing.py" in coding_files
    assert CODING_DIR / "tools.py" in coding_files
    assert CODING_DIR / "prompt_templates.py" in coding_files
    tpl_tree = ast.parse((CODING_DIR / "prompt_templates.py").read_text(encoding="utf-8"))
    for node in ast.walk(tpl_tree):
        assert not isinstance(
            node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ), f"prompt_templates.py must contain only pure text constants, found {type(node).__name__}"
    allowed = {"pydantic", "pydantic_settings", "castle_steward_coding"}
    for path in coding_files:
        for level, mod in _imports(path):
            assert level <= 1, f"{path} has escaping relative import level={level}"
            if level == 1:
                continue
            top = mod.split(".")[0]
            assert top not in {"langgraph", "sqlalchemy", "sqlmodel", "alembic"}, (
                f"{path} must not import {mod}"
            )
            if top == "castle_steward_engine":
                assert mod == "castle_steward_engine", (
                    f"{path} must import only top-level castle_steward_engine, not {mod}"
                )
            else:
                assert top in STDLIB or top in allowed, f"{path} unexpected import: {mod}"


def test_lazy_top_level_and_mcp_never_load_langgraph_sqlite_or_testing() -> None:
    code = (
        "import sys, castle_steward_engine, castle_steward_engine.mcp_server, "
        "castle_steward_engine.cli; "
        "assert castle_steward_engine.QuotaSnapshot().by_agent == {}; "
        "assert castle_steward_engine.QuotaReading().blocked is False; "
        "wf = castle_steward_engine.load_workflow('coding'); "
        "assert wf.name == 'coding'; "
        "bad = [m for m in ('langgraph', 'sqlite3', 'sqlalchemy', 'sqlmodel', 'alembic', "
        "'castle_steward_engine.testing') "
        "if m in sys.modules]; "
        "assert not bad, f'unexpectedly imported: {bad}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
