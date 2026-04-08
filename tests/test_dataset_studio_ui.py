import sys
import os
import subprocess
import tempfile
import threading
import unittest
import warnings
from pathlib import Path

import shutil

import dataset_studio as ds

from tests.test_dataset_studio import DatasetStudioFixture

warnings.filterwarnings("ignore", category=ResourceWarning)


REPO_ROOT = Path(__file__).resolve().parents[1]
SITE_CANDIDATES = [
    REPO_ROOT / ".wsl-ui-venv" / "lib" / "python3.12" / "site-packages",
    REPO_ROOT / "venv" / "lib" / "python3.12" / "site-packages",
    Path.home() / "eho" / "venv" / "lib" / "python3.12" / "site-packages",
]
for candidate in SITE_CANDIDATES:
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from selenium import webdriver
    from selenium.common.exceptions import WebDriverException
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

    SELENIUM_AVAILABLE = True
except Exception:
    SELENIUM_AVAILABLE = False


def _existing_path(value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    return path if path.exists() else None


def _find_wsl_headless_chrome() -> tuple[Path, Path, str] | None:
    env_binary = _existing_path(os.getenv("DATASET_STUDIO_UI_CHROME_BINARY"))
    env_driver = _existing_path(os.getenv("DATASET_STUDIO_UI_CHROMEDRIVER"))
    env_lib = os.getenv("DATASET_STUDIO_UI_LD_LIBRARY_PATH", "")
    if env_binary and env_driver:
        return env_binary, env_driver, env_lib

    browser_roots = [
        Path.home() / "eho-ui-browser",
        REPO_ROOT / ".wsl-browser",
    ]
    for root in browser_roots:
        binary = root / "chrome-linux64" / "chrome"
        driver = root / "chromedriver-linux64" / "chromedriver"
        libs = root / "local-libs" / "usr" / "lib" / "x86_64-linux-gnu"
        if binary.exists() and driver.exists():
            return binary, driver, str(libs) if libs.exists() else ""

    binary_names = ("google-chrome", "chromium", "chromium-browser")
    for name in binary_names:
        binary_path = shutil.which(name)
        if not binary_path:
            continue
        driver_path = shutil.which("chromedriver")
        if driver_path:
            return Path(binary_path), Path(driver_path), ""

    return None


class DatasetStudioUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not SELENIUM_AVAILABLE:
            raise unittest.SkipTest("selenium is not available in this environment")

        cls.tempdir = tempfile.TemporaryDirectory()
        root = Path(cls.tempdir.name)
        fixture = DatasetStudioFixture(root)
        fixture.build()
        cls.store = ds.DatasetStudioStore(fixture.output)
        cls.store.refresh()
        cls.server = ds.ThreadingHTTPServer(("127.0.0.1", 0), ds.DatasetStudioHandler)
        cls.server.store = cls.store  # type: ignore[attr-defined]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        host, port = cls.server.server_address
        cls.base = f"http://{host}:{port}"

        browser_setup = _find_wsl_headless_chrome()
        if browser_setup is None:
            cls.server.shutdown()
            cls.server.server_close()
            cls.thread.join(timeout=3)
            cls.tempdir.cleanup()
            raise unittest.SkipTest(
                "browser automation unavailable: no WSL headless Chrome/Chromedriver configured"
            )
        binary_path, driver_path, lib_path = browser_setup

        options = ChromeOptions()
        options.add_argument("--headless=new")
        options.add_argument("--disable-gpu")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1440,1200")
        options.binary_location = str(binary_path)
        options.add_argument(f"--user-data-dir={Path(cls.tempdir.name) / 'chrome-profile'}")
        if lib_path:
            cls._old_ld_library_path = os.environ.get("LD_LIBRARY_PATH")
            os.environ["LD_LIBRARY_PATH"] = (
                f"{lib_path}:{cls._old_ld_library_path}" if cls._old_ld_library_path else lib_path
            )

        try:
            service = ChromeService(executable_path=str(driver_path), log_output=subprocess.DEVNULL)
            cls.driver = webdriver.Chrome(service=service, options=options)
        except Exception as exc:
            cls.server.shutdown()
            cls.server.server_close()
            cls.thread.join(timeout=3)
            cls.tempdir.cleanup()
            raise unittest.SkipTest(f"browser automation unavailable: {exc}")
        cls.wait = WebDriverWait(cls.driver, 20)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "driver"):
            try:
                cls.driver.quit()
            except Exception:
                pass
        if hasattr(cls, "_old_ld_library_path"):
            if cls._old_ld_library_path is None:
                os.environ.pop("LD_LIBRARY_PATH", None)
            else:
                os.environ["LD_LIBRARY_PATH"] = cls._old_ld_library_path
        if hasattr(cls, "server"):
            cls.server.shutdown()
            cls.server.server_close()
            cls.thread.join(timeout=3)
        if hasattr(cls, "tempdir"):
            cls.tempdir.cleanup()

    def test_can_navigate_core_tabs_and_open_chunk(self):
        self.driver.get(self.base)
        self.wait.until(EC.presence_of_element_located((By.ID, "summaryGrid")))
        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-tab='chunks']")))

        self.driver.find_element(By.CSS_SELECTOR, "[data-tab='chunks']").click()
        self.wait.until(EC.presence_of_element_located((By.ID, "chunksList")))
        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#chunksList [data-chunk-item]")))
        self.driver.find_element(By.CSS_SELECTOR, "#chunksList [data-chunk-item]").click()
        self.wait.until(lambda d: "Дом у Моста" in d.find_element(By.ID, "chunkDetailText").get_attribute("value"))

        self.driver.find_element(By.CSS_SELECTOR, "[data-tab='pipeline']").click()
        self.wait.until(lambda d: "\"model\"" in d.find_element(By.ID, "pipelineMetadata").get_attribute("value"))

        self.driver.find_element(By.CSS_SELECTOR, "[data-tab='timeline']").click()
        self.wait.until(EC.presence_of_element_located((By.ID, "timelineSummaryGrid")))
        self.assertIn("Nodes", self.driver.page_source)

    def test_can_open_llm_trace_explorer(self):
        self.driver.get(self.base)
        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "[data-tab='llm']")))
        self.driver.find_element(By.CSS_SELECTOR, "[data-tab='llm']").click()
        self.wait.until(EC.presence_of_element_located((By.ID, "llmTraceList")))
        self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#llmTraceList [data-llm-trace]")))
        self.driver.find_element(By.CSS_SELECTOR, "#llmTraceList [data-llm-trace]").click()
        self.wait.until(lambda d: d.find_element(By.NAME, "model_override").get_attribute("value") != "")


if __name__ == "__main__":
    unittest.main()
