"""
Test the Country Guess App end-to-end.

Usage:
    python -m unittest tests/test_e2e.py
"""

import json
import os
import shutil
import subprocess
import time
import unittest
from pathlib import Path

import requests


def wait_for_service(url, timeout=10):
    for _ in range(timeout):
        try:
            if requests.get(url, timeout=1).status_code == 200:
                return
        except requests.exceptions.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError(f"Service at {url} is unhealthy")


class TestEndToEnd(unittest.TestCase):
    """Test the Country Guess App end-to-end."""

    MLSERVER_URL = "http://localhost:5001"
    WEBAPP_URL = "http://localhost:5002"
    DRAWING_DIR = Path("tests/data/drawings_tmp")

    @classmethod
    def setUpClass(cls):
        # Start servers with custom environment
        env = os.environ.copy()
        env["DRAWING_DIR"] = str(cls.DRAWING_DIR)

        cls.mlserver_process = subprocess.Popen(
            ["make", "run-mlserver", "DEBUG=0"],
        )
        cls.webapp_process = subprocess.Popen(
            ["make", "run-webapp", "DEBUG=0"],
            env=env,
        )

        # Wait for services
        wait_for_service(f"{cls.MLSERVER_URL}/health", timeout=20)
        wait_for_service(f"{cls.WEBAPP_URL}/health", timeout=20)

        # Discover test files
        cls.test_files = sorted(Path("tests/data/lines").glob("*.json"))

    @classmethod
    def tearDownClass(cls):
        # Stop servers
        cls.mlserver_process.terminate()
        cls.mlserver_process.wait()
        cls.webapp_process.terminate()
        cls.webapp_process.wait()

        # Remove the entire drawings directory
        if cls.DRAWING_DIR.exists():
            shutil.rmtree(cls.DRAWING_DIR)

    def _run_country_guess_test(self, country_name, test_drawing):
        """Walk through the country guess flow on the webapp"""
        # Step 1: Send drawing and get prediction
        response = requests.post(
            f"{self.WEBAPP_URL}/guess", json=test_drawing, timeout=10
        )
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("drawing_id", result)
        self.assertIn("ranking", result)

        # Assert country is top 3 prediction
        ranking = result["ranking"]
        top_countries = [country for country, score in ranking[:3]]
        self.assertIn(country_name, top_countries)

        # Step 2: Submit feedback
        feedback_data = {"country": country_name, "drawing_id": result["drawing_id"]}
        response = requests.post(
            f"{self.WEBAPP_URL}/feedback", json=feedback_data, timeout=10
        )
        self.assertEqual(response.status_code, 200)

        # Step 3: Verify drawing was saved
        country_file = country_name.lower().replace(" ", "_")
        files = list(self.DRAWING_DIR.glob(f"{country_file}_*.geojson"))
        self.assertTrue(files, f"Drawing of {country_name} has not been saved")

    def test_country_guess_app(self):
        """Test the Country Guess App end-to-end for all countries."""
        for test_file in self.test_files:
            # Load test drawing
            with open(test_file, encoding="utf-8") as f:
                test_drawing = json.load(f)
            country_name = test_drawing["name"]

            with self.subTest(country_name=country_name):
                print(f"\nTesting country: {country_name}")
                self._run_country_guess_test(country_name, test_drawing)


if __name__ == "__main__":
    unittest.main()
