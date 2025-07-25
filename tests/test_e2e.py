"""
Test the Country Guess App end-to-end.

Usage:
    python -m tests.evaluation --model_name <MODEL_NAME>
"""

import json
import subprocess
import time
import unittest
from pathlib import Path

import requests


class TestEndToEnd(unittest.TestCase):
    """Test the Country Guess App end-to-end."""

    @classmethod
    def setUpClass(cls):
        """Start the ML server and webapp before running tests."""
        cls.mlserver_process = subprocess.Popen(
            ["make", "run-mlserver", "DEBUG=0"],
        )
        cls.webapp_process = subprocess.Popen(
            ["make", "run-webapp", "DEBUG=0", "DRAWING_DIR=tests/data/drawings"],
        )

        # Wait for ML server and webapp to be healthy
        cls._wait_for_service("http://localhost:5001/health", timeout=20)
        cls._wait_for_service("http://localhost:5002/health", timeout=20)

    @classmethod
    def tearDownClass(cls):
        """Stop the ML server and webapp after tests are done."""
        cls.mlserver_process.terminate()
        cls.mlserver_process.wait()
        cls.webapp_process.terminate()
        cls.webapp_process.wait()

    @staticmethod
    def _wait_for_service(url, timeout=10):
        """Wait for a service to become healthy."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if requests.get(url, timeout=1).status_code == 200:
                    return  # Service is healthy
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        raise RuntimeError(f"Service at {url} is unhealthy")

    def setUp(self):
        self.webapp_url = "http://localhost:5002"
        self.drawing_dir = Path("tests/data/drawings")
        self.country_name = "Mali"
        self.file_name = self.country_name.lower()
        self.start_time = time.time()

        # Load test drawing
        test_data_path = Path(__file__).parent / "data" / f"{self.file_name}.json"
        with open(test_data_path, encoding="utf-8") as f:
            self.test_drawing = json.load(f)

    def test_country_guess_app(self):
        """Test the Country Guess App end-to-end."""

        # Step 1: Send drawing and get prediction
        response = requests.post(
            f"{self.webapp_url}/guess", json=self.test_drawing, timeout=10
        )
        self.assertEqual(response.status_code, 200)

        result = response.json()
        self.assertIn("drawing_id", result)
        self.assertIn("ranking", result)

        # Assert country is top 3 prediction
        ranking = result["ranking"]
        self.assertIn("countries", ranking)
        self.assertIn("scores", ranking)
        self.assertIn(self.country_name, ranking["countries"][:3])

        drawing_id = result["drawing_id"]

        # Step 2: Submit feedback
        feedback_data = {"country": self.country_name, "drawing_id": drawing_id}
        response = requests.post(
            f"{self.webapp_url}/feedback", json=feedback_data, timeout=10
        )
        self.assertEqual(response.status_code, 200)

        # Step 3: Verify drawing was saved
        files = list(self.drawing_dir.glob(f"{self.file_name}_*.geojson"))
        self.assertTrue(files, "Could not find drawing file")

        latest_file = max(files, key=lambda p: p.stat().st_ctime)

        # Verify file was created during test
        file_ctime = latest_file.stat().st_ctime
        self.assertGreater(
            file_ctime, self.start_time, "Latest file was created before test started"
        )

        # Register cleanup for this specific file
        self.addCleanup(latest_file.unlink)


if __name__ == "__main__":
    unittest.main()
