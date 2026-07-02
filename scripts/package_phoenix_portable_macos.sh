#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Build a portable Phoenix macOS folder with a simple Phoenix.app launcher.
#
# Run this on macOS from the Phoenix repo:
#   bash scripts/package_phoenix_portable_macos.sh
#
# Output:
#   dist/PhoenixPortableMac/
#   dist/PhoenixPortableMac.zip
#
# This is not a notarized App Store-style app. It is a local seminar/workshop
# bundle around a packed conda environment and a Streamlit launcher.
# ---------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT_DIR/build/portable_macos"
DIST_ROOT="$ROOT_DIR/dist"
DIST_DIR="$DIST_ROOT/PhoenixPortableMac"
BUILD_ENV="$BUILD_DIR/env"
ENV_ARCHIVE="$BUILD_DIR/phoenix-env-macos.tar.gz"
ZIP_FILE="$DIST_ROOT/PhoenixPortableMac.zip"
APP_BUNDLE="$DIST_DIR/Phoenix.app"
APP_MACOS="$APP_BUNDLE/Contents/MacOS"
APP_RESOURCES="$APP_BUNDLE/Contents/Resources"

echo
echo "============================================================"
echo "Building Phoenix portable macOS package"
echo "============================================================"
echo "Repo:   $ROOT_DIR"
echo "Output: $DIST_DIR"
echo

if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda was not found on PATH."
    echo "Install Miniconda/Mambaforge or run this from a conda-enabled shell."
    exit 1
fi

if [[ ! -f "$ROOT_DIR/cellbench/environment.yml" ]]; then
    echo "ERROR: missing $ROOT_DIR/cellbench/environment.yml"
    exit 1
fi

mkdir -p "$BUILD_DIR" "$DIST_ROOT"

if [[ -d "$BUILD_ENV/conda-meta" ]]; then
    echo "Updating build environment..."
    conda env update -p "$BUILD_ENV" -f "$ROOT_DIR/cellbench/environment.yml" --prune
else
    echo "Creating build environment..."
    conda env create -p "$BUILD_ENV" -f "$ROOT_DIR/cellbench/environment.yml"
fi

echo "Ensuring conda-pack is available..."
conda install -n base -c conda-forge conda-pack -y

echo "Packing environment..."
rm -f "$ENV_ARCHIVE"
conda run -n base conda-pack -p "$BUILD_ENV" -o "$ENV_ARCHIVE" --force

echo "Recreating portable folder..."
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/app" "$DIST_DIR/env" "$APP_MACOS" "$APP_RESOURCES"

echo "Extracting portable environment..."
tar -xzf "$ENV_ARCHIVE" -C "$DIST_DIR/env"

echo "Copying Phoenix source tree..."
if command -v rsync >/dev/null 2>&1; then
    rsync -a "$ROOT_DIR/" "$DIST_DIR/app/" \
        --exclude ".git" \
        --exclude ".venv" \
        --exclude "__pycache__" \
        --exclude "build" \
        --exclude "dist" \
        --exclude ".pytest_cache" \
        --exclude ".mypy_cache" \
        --exclude "*.pyc" \
        --exclude ".DS_Store"
else
    cp -R "$ROOT_DIR/." "$DIST_DIR/app/"
    rm -rf "$DIST_DIR/app/.git" "$DIST_DIR/app/build" "$DIST_DIR/app/dist"
fi

cat > "$APP_MACOS/Phoenix" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

PORT="${PHOENIX_PORT:-8501}"
ROOT_DIR="$(cd "$(dirname "$0")/../../.." && pwd)"
APP_DIR="$ROOT_DIR/app"
ENV_DIR="$ROOT_DIR/env"

if [[ ! -f "$APP_DIR/phoenix/app.py" ]]; then
    osascript -e 'display dialog "Could not find phoenix/app.py inside the portable Phoenix folder." buttons {"OK"} with icon stop' || true
    exit 1
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
    osascript -e 'display dialog "Could not find bundled Python inside the portable Phoenix folder." buttons {"OK"} with icon stop' || true
    exit 1
fi

if [[ -x "$ENV_DIR/bin/conda-unpack" && ! -f "$ENV_DIR/.phoenix_unpacked" ]]; then
    "$ENV_DIR/bin/conda-unpack"
    touch "$ENV_DIR/.phoenix_unpacked"
fi

export STREAMLIT_SERVER_ADDRESS="127.0.0.1"
export STREAMLIT_SERVER_PORT="$PORT"
export STREAMLIT_SERVER_HEADLESS="true"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS="false"

( sleep 3; open "http://127.0.0.1:$PORT" ) &

"$ENV_DIR/bin/python" -m streamlit run "$APP_DIR/phoenix/app.py" \
    --server.address=127.0.0.1 \
    --server.port="$PORT" \
    --server.headless=true \
    --browser.gatherUsageStats=false
EOF

chmod +x "$APP_MACOS/Phoenix"

cat > "$APP_BUNDLE/Contents/Info.plist" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>Phoenix</string>
  <key>CFBundleDisplayName</key>
  <string>Phoenix Battery Lab</string>
  <key>CFBundleIdentifier</key>
  <string>de.uni-giessen.ag-bielefeld.phoenix</string>
  <key>CFBundleVersion</key>
  <string>0.1.0</string>
  <key>CFBundleShortVersionString</key>
  <string>0.1.0</string>
  <key>CFBundleExecutable</key>
  <string>Phoenix</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
</dict>
</plist>
EOF

cat > "$DIST_DIR/README_FIRST.txt" <<'EOF'
Phoenix portable macOS package
==============================

How to run:
1. Double-click Phoenix.app.
2. If macOS blocks it because it is unsigned, right-click Phoenix.app and choose Open.
3. Your browser should open http://127.0.0.1:8501.

How to stop:
- Close the terminal/process window if one appears, or quit the app process.

Notes:
- This is a local app wrapper around Streamlit/PyBaMM.
- It does not expose Phoenix to the network.
- The first launch may take a bit longer while the packed environment is prepared.
EOF

echo "Creating zip archive..."
rm -f "$ZIP_FILE"
(
    cd "$DIST_ROOT"
    zip -qr "$ZIP_FILE" "PhoenixPortableMac"
)

echo
echo "Done."
echo
echo "Test locally:"
echo "  open \"$APP_BUNDLE\""
echo
echo "Distribution:"
echo "  Give people $ZIP_FILE, have them extract it, then open Phoenix.app."
echo
