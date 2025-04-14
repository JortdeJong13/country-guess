"""Test the Country Guess App end-to-end."""

import json
import unittest
from pathlib import Path
import time
import requests


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.webapp_url = "http://localhost:5002"
        self.drawing_dir = Path("data/drawings")
        self.country_name = "Mali"
        self.file_name = self.country_name.lower()
        self.start_time = time.time()

        # Load test drawing
        test_data_path = Path(__file__).parent / "data" / f"{self.file_name}.json"
        with open(test_data_path) as f:
            self.test_drawing = json.load(f)

    def test_country_guess_app(self):
        """Test the Country Guess App end-to-end."""

        # Step 1: Send drawing and get prediction
        response = requests.post(f"{self.webapp_url}/guess", json=self.test_drawing)
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

        response = requests.post(f"{self.webapp_url}/feedback", json=feedback_data)
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
