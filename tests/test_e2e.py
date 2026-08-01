"""
Test the Country Guess App end-to-end.

Usage:
    uv run --only-group app python -m unittest tests/test_e2e.py
"""

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

import requests

from tests.e2e_services import (
    start_drawingstore,
    stop_drawingstore,
    stop_process,
    wait_for_service,
)


class TestEndToEnd(unittest.TestCase):
    """Test the Country Guess App end-to-end."""

    MLSERVER_URL = "http://localhost:5001"
    WEBAPP_URL = "http://localhost:5002"
    DRAWING_STORE_URL = "http://localhost:8080"

    @classmethod
    def setUpClass(cls):
        (
            cls.database_name,
            cls.drawingstore_process,
            cls.DRAWING_STORE_URL,
        ) = start_drawingstore()

        # Start servers with custom environment
        mlserver_env = os.environ.copy()
        mlserver_env.update({"DEBUG": "0", "MODEL_NAME": "triplet_model"})
        webapp_env = os.environ.copy()
        webapp_env.update(
            {
                "DEBUG": "0",
                "MLSERVER_URL": cls.MLSERVER_URL,
                "DRAWING_STORE_URL": cls.DRAWING_STORE_URL,
            }
        )

        cls.mlserver_process = subprocess.Popen(
            [sys.executable, "-m", "mlserver.serve"],
            env=mlserver_env,
            start_new_session=True,
        )
        cls.webapp_process = subprocess.Popen(
            [sys.executable, "-m", "webapp.app"],
            env=webapp_env,
            start_new_session=True,
        )

        # Wait for services
        wait_for_service(f"{cls.MLSERVER_URL}/health", timeout=20)
        wait_for_service(f"{cls.WEBAPP_URL}/health", timeout=20)

        # Discover test files
        cls.test_files = sorted(Path("tests/data/lines").glob("*.json"))

    @classmethod
    def tearDownClass(cls):
        # Stop servers
        stop_process(cls.mlserver_process)
        stop_process(cls.webapp_process)

        stop_drawingstore(cls.database_name, cls.drawingstore_process)

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

        # Step 3: Verify drawing was saved in PostgreSQL
        response = requests.get(
            f"{self.DRAWING_STORE_URL}/drawings/{result['drawing_id']}", timeout=5
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(country_name, response.json()["country"])

        # Step 4: TODO: Test leaderboard feature

    def test_country_guess_app(self):
        """Test the Country Guess App end-to-end for all test drawings."""
        empty_response = requests.get(
            f"{self.WEBAPP_URL}/drawing?rank=0", timeout=5
        )
        self.assertEqual(404, empty_response.status_code)
        self.assertEqual(
            "No drawing found for rank 0", empty_response.json()["message"]
        )

        for test_file in self.test_files:
            # Load test drawing
            with open(test_file, encoding="utf-8") as f:
                test_drawing = json.load(f)
            country_name = test_drawing["name"]

            with self.subTest(country_name=country_name):
                print(f"\nTesting country: {country_name}")
                self._run_country_guess_test(country_name, test_drawing)

        leaderboard_response = requests.get(
            f"{self.WEBAPP_URL}/drawing?rank=0", timeout=5
        )
        self.assertEqual(200, leaderboard_response.status_code)
        self.assertIn("id", leaderboard_response.json())


if __name__ == "__main__":
    unittest.main()
