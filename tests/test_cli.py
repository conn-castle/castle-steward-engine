"""CLI, EngineSettings, and real stdio `stew mcp` smoke tests."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from castle_steward_engine import EngineSettings
from castle_steward_engine.cli import main
from castle_steward_engine.mcp_server import SessionDir


def test_cli_workflows_settings_start_inspect_and_stop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert main(["workflows"]) == 0
    wfs = json.loads(capsys.readouterr().out)
    assert any(item["name"] == "coding" and item["version"] == "2.0.0" for item in wfs)

    env_state = tmp_path / "env-state"
    cli_state = tmp_path / "cli-state"
    monkeypatch.setenv("STEW_ENGINE_STATE_DIR", str(env_state))
    monkeypatch.setenv("STEW_ENGINE_AGENT_LAYER_EXECUTABLE", "custom-al")
    # Verify STEW_ENGINE_CODING_* env vars do not trip EngineSettings(extra="forbid").
    monkeypatch.setenv("STEW_ENGINE_CODING_MAX_AUTO_REPAIRS", "0")
    settings = EngineSettings()
    assert settings.state_dir == env_state and settings.agent_layer_executable == "custom-al"

    inp = json.dumps({"task": "demo", "checkout": str(tmp_path)})
    # Uses STEW_ENGINE_STATE_DIR when --state-dir is omitted
    assert main(["start", "coding", "--input", inp]) == 0
    job_id = json.loads(capsys.readouterr().out)["job_id"]
    assert (env_state / "engine.db").is_file()

    assert main(["inspect", job_id]) == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "pending"

    assert main(["stop", job_id]) == 0
    capsys.readouterr()
    assert main(["inspect", job_id]) == 0
    assert json.loads(capsys.readouterr().out)["phase"] == "stopped"

    # Explicit --state-dir overrides STEW_ENGINE_STATE_DIR
    assert main(["--state-dir", str(cli_state), "start", "coding", "--input", inp]) == 0
    capsys.readouterr()
    assert (cli_state / "engine.db").is_file()

    for cmd, flag in ((["start", "coding"], "--input"), (["reply", job_id, "w"], "--value")):
        for extra in ([], [flag, "{}", f"{flag}-file", "f"]):
            with pytest.raises(SystemExit) as exc:
                main([*cmd, *extra])
            assert exc.value.code == 2


def test_real_stdio_mcp_server_smoke(tmp_path: Path) -> None:
    session = SessionDir(tmp_path / "session")
    session.init(workflow="coding", role="coder")
    session.prepare_turn(
        turn_key="t1:0",
        attempt=1,
        expects_tool="submit_plan",
        context={},
        prompt="write plan",
    )

    async def _exercise() -> None:
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "castle_steward_engine", "mcp", "--session-dir", str(session.path)],
        )
        async with (
            stdio_client(params) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as client,
        ):
            await client.initialize()
            tools = [t.name for t in (await client.list_tools()).tools]
            assert tools == ["submit_plan", "ask", "check_done"]
            before = await client.call_tool("check_done", {})
            assert "Incomplete" in before.content[0].text  # type: ignore[union-attr]
            ok = await client.call_tool("submit_plan", {"plan": "# Plan\n1. Do work\n"})
            assert not ok.is_error
            conflict = await client.call_tool("ask", {"question": "too late?"})
            assert conflict.is_error
            after = await client.call_tool("check_done", {})
            assert not after.is_error
            assert "Complete" in after.content[0].text  # type: ignore[union-attr]

            # Switching the session to a check-only turn rejects submit_plan cleanly
            session.prepare_turn(
                turn_key="t2:0",
                attempt=1,
                expects_check="commit",
                context={},
                prompt="commit work",
            )
            inactive = await client.call_tool("submit_plan", {"plan": "# Wrong turn\n"})
            assert inactive.is_error
            assert "check_done" in inactive.content[0].text  # type: ignore[union-attr]

    asyncio.run(_exercise())
