#!/usr/bin/env python3
"""Build a macOS Phoenix bootstrap zip from any platform.

This is the cross-platform fallback for instructors who only have Windows/Linux
but need to give Mac users a simple download. It does not bundle a prebuilt
macOS Python environment. Instead, the zip contains Phoenix source plus a
double-clickable macOS command script that downloads micromamba and creates the
environment on the user's Mac at first launch.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import stat
import textwrap
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ZIP_PATH = DIST / "PhoenixMacBootstrap.zip"
PACKAGE_ROOT = "PhoenixMac"

EXCLUDE_DIRS = {
    ".git",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    ".pytest_cache",
    ".mypy_cache",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo"}


RUN_COMMAND = r"""#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_DIR="$ROOT_DIR/app"
ENV_DIR="$ROOT_DIR/env"
MAMBA_ROOT="$ROOT_DIR/.micromamba"
MAMBA_BIN="$MAMBA_ROOT/bin/micromamba"
PORT="${PHOENIX_PORT:-8501}"

echo
echo "============================================================"
echo "Phoenix Battery Lab"
echo "============================================================"
echo

if [[ ! -f "$APP_DIR/phoenix/app.py" ]]; then
    echo "ERROR: Could not find Phoenix at $APP_DIR/phoenix/app.py"
    read -r -p "Press Enter to close..."
    exit 1
fi

install_micromamba() {
    mkdir -p "$MAMBA_ROOT"
    if [[ -x "$MAMBA_BIN" ]]; then
        return
    fi

    ARCH="$(uname -m)"
    case "$ARCH" in
        arm64)
            MAMBA_URL="https://micro.mamba.pm/api/micromamba/osx-arm64/latest"
            ;;
        x86_64)
            MAMBA_URL="https://micro.mamba.pm/api/micromamba/osx-64/latest"
            ;;
        *)
            echo "ERROR: unsupported macOS CPU architecture: $ARCH"
            read -r -p "Press Enter to close..."
            exit 1
            ;;
    esac

    echo "Downloading micromamba for macOS $ARCH..."
    curl -L "$MAMBA_URL" -o "$ROOT_DIR/micromamba.tar.bz2"
    tar -xjf "$ROOT_DIR/micromamba.tar.bz2" -C "$MAMBA_ROOT"
    rm -f "$ROOT_DIR/micromamba.tar.bz2"

    if [[ ! -x "$MAMBA_BIN" ]]; then
        echo "ERROR: micromamba installation did not produce $MAMBA_BIN"
        read -r -p "Press Enter to close..."
        exit 1
    fi
}

install_environment() {
    install_micromamba
    if [[ -x "$ENV_DIR/bin/python" ]]; then
        return
    fi

    echo
    echo "Creating the Phoenix Python environment."
    echo "This first run can take several minutes and needs internet access."
    echo

    "$MAMBA_BIN" create -y -p "$ENV_DIR" -f "$APP_DIR/cellbench/environment.yml"

    if [[ ! -x "$ENV_DIR/bin/python" ]]; then
        echo "ERROR: environment creation did not produce $ENV_DIR/bin/python"
        read -r -p "Press Enter to close..."
        exit 1
    fi
}

install_environment

export STREAMLIT_SERVER_ADDRESS="127.0.0.1"
export STREAMLIT_SERVER_PORT="$PORT"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"
export PYTHONNOUSERSITE="1"

echo
echo "Starting Phoenix locally at:"
echo "  http://127.0.0.1:$PORT"
echo
echo "Close this Terminal window or press Ctrl+C to stop Phoenix."
echo

( sleep 3; open "http://127.0.0.1:$PORT" ) &

"$ENV_DIR/bin/python" -m streamlit run "$APP_DIR/phoenix/app.py" \
    --server.address=127.0.0.1 \
    --server.port="$PORT" \
    --server.headless=true \
    --browser.gatherUsageStats=false
"""


RESET_COMMAND = r"""#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "This removes the local Phoenix Mac environment created on this computer."
echo "It does not delete the Phoenix app source."
echo
read -r -p "Remove env/ and .micromamba/? [y/N] " ANSWER
case "$ANSWER" in
    y|Y|yes|YES)
        rm -rf "$ROOT_DIR/env" "$ROOT_DIR/.micromamba"
        echo "Removed local Phoenix environment."
        ;;
    *)
        echo "Cancelled."
        ;;
esac
read -r -p "Press Enter to close..."
"""


README = """\
Phoenix Mac bootstrap package
=============================

This package is for Mac users when the instructor cannot build a native Mac app.

How to run:
1. Extract PhoenixMacBootstrap.zip.
2. Open the extracted PhoenixMac folder.
3. Double-click "Run Phoenix.command".
4. The first launch downloads micromamba and creates the Phoenix environment.
5. Your browser should open http://127.0.0.1:8501.

Requirements:
- macOS on Apple Silicon or Intel.
- Internet access on the first launch.
- No Git required.
- No existing Python required.
- No existing conda required.

If macOS blocks the command:
- Right-click "Run Phoenix.command" and choose Open.

How to stop:
- Close the Terminal window or press Ctrl+C in it.

How to reset:
- Double-click "Reset Phoenix Environment.command".

Notes:
- This is not an offline notarized macOS app.
- It is a seminar-friendly bootstrap package around Streamlit/PyBaMM.
- The first launch can take several minutes because scientific Python packages
  are installed for the user's Mac architecture.
"""


def should_include(path: Path) -> bool:
    """Return True if a repo path should be included in the bootstrap app copy."""

    relative = path.relative_to(ROOT)
    if any(part in EXCLUDE_DIRS for part in relative.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return True


def iter_files() -> list[Path]:
    """Return sorted source files to include."""

    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if path.is_file() and should_include(path):
            files.append(path)
    return sorted(files)


def write_text_member(
    archive: zipfile.ZipFile,
    name: str,
    content: str,
    *,
    executable: bool = False,
) -> None:
    """Write a text file to the zip with stable Unix permissions."""

    data = textwrap.dedent(content).encode("utf-8")
    info = zipfile.ZipInfo(name)
    mode = 0o755 if executable else 0o644
    info.external_attr = (stat.S_IFREG | mode) << 16
    archive.writestr(info, data)


def write_file_member(archive: zipfile.ZipFile, source: Path, name: str) -> None:
    """Write a repo file to the zip with regular file permissions."""

    info = zipfile.ZipInfo(name)
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    archive.writestr(info, source.read_bytes())


def main() -> None:
    """Create dist/PhoenixMacBootstrap.zip."""

    DIST.mkdir(exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        write_text_member(
            archive,
            f"{PACKAGE_ROOT}/Run Phoenix.command",
            RUN_COMMAND,
            executable=True,
        )
        write_text_member(
            archive,
            f"{PACKAGE_ROOT}/Reset Phoenix Environment.command",
            RESET_COMMAND,
            executable=True,
        )
        write_text_member(
            archive,
            f"{PACKAGE_ROOT}/README_FIRST.txt",
            README,
        )
        for source in iter_files():
            relative = source.relative_to(ROOT).as_posix()
            write_file_member(archive, source, f"{PACKAGE_ROOT}/app/{relative}")

    size_mb = ZIP_PATH.stat().st_size / 1024 / 1024
    print(f"Created {ZIP_PATH} ({size_mb:.1f} MB)")
    print("Give this zip to Mac users. They extract it and double-click Run Phoenix.command.")


if __name__ == "__main__":
    main()
