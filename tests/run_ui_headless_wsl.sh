#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$ROOT/.wsl-ui-venv"
BROWSER_HOME="${DATASET_STUDIO_UI_BROWSER_HOME:-$HOME/eho-ui-browser}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p "$BROWSER_HOME"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

source "$VENV/bin/activate"
python -m pip install --upgrade pip setuptools wheel selenium >/dev/null

python - <<'PY' "$BROWSER_HOME"
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
import subprocess
import sys

browser_home = Path(sys.argv[1])
browser_home.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(
    "https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions-with-downloads.json",
    timeout=30,
) as response:
    stable = json.load(response)["channels"]["Stable"]

chrome_url = next(item["url"] for item in stable["downloads"]["chrome"] if item["platform"] == "linux64")
driver_url = next(item["url"] for item in stable["downloads"]["chromedriver"] if item["platform"] == "linux64")

version_file = browser_home / "version.txt"
current_version = version_file.read_text(encoding="utf-8").strip() if version_file.exists() else ""
target_version = stable["version"]

if current_version != target_version:
    for name, url in (("chrome-linux64.zip", chrome_url), ("chromedriver-linux64.zip", driver_url)):
        archive = browser_home / name
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as fh:
            shutil.copyfileobj(response, fh)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(browser_home)

    subprocess.run(["apt", "download", "libnss3", "libnspr4", "libasound2t64"], cwd=browser_home, check=True)
    local_libs = browser_home / "local-libs"
    if local_libs.exists():
        shutil.rmtree(local_libs)
    local_libs.mkdir(parents=True)
    for deb in browser_home.glob("*.deb"):
        subprocess.run(["dpkg-deb", "-x", str(deb), str(local_libs)], check=True)

    version_file.write_text(target_version, encoding="utf-8")
PY

export DATASET_STUDIO_UI_CHROME_BINARY="$BROWSER_HOME/chrome-linux64/chrome"
export DATASET_STUDIO_UI_CHROMEDRIVER="$BROWSER_HOME/chromedriver-linux64/chromedriver"
export DATASET_STUDIO_UI_LD_LIBRARY_PATH="$BROWSER_HOME/local-libs/usr/lib/x86_64-linux-gnu"

cd "$ROOT"
python -m unittest tests.test_dataset_studio_ui -v
