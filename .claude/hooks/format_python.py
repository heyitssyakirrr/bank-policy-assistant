"""PostToolUse hook: format and auto-fix a Python file right after Claude edits it.

Never blocks Claude: any problem (no ruff, not a .py file) is ignored.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_RUFF = [REPO_ROOT / ".venv" / "Scripts" / "ruff.exe", REPO_ROOT / ".venv" / "bin" / "ruff"]


def find_ruff() -> str | None:
    for candidate in VENV_RUFF:
        if candidate.exists():
            return str(candidate)
    return shutil.which("ruff")


def main() -> None:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return
    file_path = event.get("tool_input", {}).get("file_path", "")
    ruff = find_ruff()
    if not file_path.endswith(".py") or ruff is None:
        return
    subprocess.run([ruff, "check", "--fix", "--quiet", file_path], cwd=REPO_ROOT, check=False)
    subprocess.run([ruff, "format", "--quiet", file_path], cwd=REPO_ROOT, check=False)


if __name__ == "__main__":
    main()
