import os
import subprocess
import sys
import unittest

import requests

from tests.e2e_services import (
    _free_local_port,
    start_drawingstore,
    stop_drawingstore,
    stop_process,
    wait_for_service,
)


class TestAdminEndToEnd(unittest.TestCase):
    """Test validation and deletion through the admin app and drawingstore."""

    ADMIN_URL = "http://localhost:5003"
    DRAWING_STORE_URL = "http://localhost:8080"

    @classmethod
    def setUpClass(cls):
        (
            cls.database_name,
            cls.drawingstore_process,
            cls.DRAWING_STORE_URL,
        ) = start_drawingstore()
        cls.ADMIN_URL = f"http://127.0.0.1:{_free_local_port()}"

        admin_env = os.environ.copy()
        admin_env.update(
            {
                "DEBUG": "0",
                "ADMIN_PORT": cls.ADMIN_URL.rsplit(":", 1)[1],
                "DRAWING_STORE_URL": cls.DRAWING_STORE_URL,
            }
        )
        cls.admin_process = subprocess.Popen(
            [sys.executable, "-m", "webapp.admin"],
            env=admin_env,
            start_new_session=True,
        )
        wait_for_service(f"{cls.ADMIN_URL}/")

        cls.france_id = cls._create_drawing(
            "France", "France", 0.9, "admin-test-author"
        )
        cls.germany_id = cls._create_drawing(
            "Germany", "Germany", 0.8, "admin-test-author"
        )

        response = requests.patch(
            f"{cls.DRAWING_STORE_URL}/drawings/{cls.germany_id}",
            json={"validated": True},
            timeout=5,
        )
        if response.status_code != 200:
            raise AssertionError(response.text)

        response = requests.patch(
            f"{cls.DRAWING_STORE_URL}/drawings/{cls.germany_id}",
            json={"report": True},
            timeout=5,
        )
        if response.status_code != 200:
            raise AssertionError(response.text)

    @classmethod
    def tearDownClass(cls):
        stop_process(cls.admin_process)
        stop_drawingstore(
            cls.database_name, cls.drawingstore_process
        )

    @classmethod
    def _create_drawing(cls, country, guess, score, author_id):
        response = requests.post(
            f"{cls.DRAWING_STORE_URL}/drawings",
            json={
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [[[0, 0], [1, 1]]],
                },
                "ranking": [
                    {"country": guess, "score": score},
                    {"country": "Other", "score": 1 - score},
                ],
                "author_id": author_id,
            },
            timeout=5,
        )
        if response.status_code != 201:
            raise AssertionError(response.text)
        drawing_id = response.json()["id"]
        response = requests.patch(
            f"{cls.DRAWING_STORE_URL}/drawings/{drawing_id}",
            json={"country": country},
            timeout=5,
        )
        if response.status_code != 200:
            raise AssertionError(response.text)
        return drawing_id

    def test_admin_page(self):
        response = requests.get(f"{self.ADMIN_URL}/unvalidated_drawing", timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual("Drawing loaded successfully", data["message"])
        self.assertEqual(2, data["validation_count"])
        self.assertEqual("admin-test-author", data["author_id"])
        self.assertEqual(1, data["validated_author_count"])
        self.assertEqual(1, data["report_count"])
        first_id = data["id"]
        self.assertEqual(self.germany_id, first_id)

        response = requests.patch(
            f"{self.ADMIN_URL}/drawing/{first_id}",
            json={"validated": True},
            timeout=5,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("updated successfully.", response.json()["message"])

        response = requests.get(
            f"{self.DRAWING_STORE_URL}/drawings/{first_id}", timeout=5
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(0, response.json()["report_count"])

        response = requests.get(f"{self.ADMIN_URL}/unvalidated_drawing", timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual("Drawing loaded successfully", data["message"])
        self.assertEqual(1, data["validation_count"])
        self.assertEqual(1, data["validated_author_count"])
        second_id = data["id"]
        self.assertEqual(self.france_id, second_id)

        response = requests.delete(
            f"{self.ADMIN_URL}/drawing/{second_id}", timeout=5
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted successfully.", response.json()["message"])

        response = requests.get(f"{self.ADMIN_URL}/unvalidated_drawing", timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual("No drawings found for validation", data["message"])
        self.assertEqual(0, data["validation_count"])


if __name__ == "__main__":
    unittest.main()
