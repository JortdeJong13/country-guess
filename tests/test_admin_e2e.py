import os
import shutil
import subprocess
import unittest
from pathlib import Path

import requests

from tests.test_e2e import wait_for_service


class TestAdminEndToEnd(unittest.TestCase):
    """Test the Country Guess Admin App end-to-end."""

    ADMIN_URL = "http://localhost:5003"
    TEST_DRAWINGS_DIR = Path("tests/data/drawings")
    DRAWING_DIR = Path("tests/data/drawings_admin")

    @classmethod
    def setUpClass(cls):
        # Copy test drawings to the drawing directory
        if cls.DRAWING_DIR.exists():
            shutil.rmtree(cls.DRAWING_DIR)
        shutil.copytree(cls.TEST_DRAWINGS_DIR, cls.DRAWING_DIR)

        # Start admin server with custom environment
        env = os.environ.copy()
        env["DRAWING_DIR"] = str(cls.DRAWING_DIR)

        cls.admin_process = subprocess.Popen(
            ["make", "run-admin", "DEBUG=0"],
            env=env,
        )

        # Wait for the admin service to start
        wait_for_service(f"{cls.ADMIN_URL}/", timeout=20)

    @classmethod
    def tearDownClass(cls):
        # Stop admin server
        cls.admin_process.terminate()
        cls.admin_process.wait()

        # Delete the temporary drawings directory
        if cls.DRAWING_DIR.exists():
            shutil.rmtree(cls.DRAWING_DIR)

    def test_admin_page(self):
        """Simulate the full admin workflow."""

        # Step 1: Call GET /unvalidated_drawing
        print("Get first unvalidated drawing")
        response = requests.get(f"{self.ADMIN_URL}/unvalidated_drawing", timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual("Drawing loaded successfully", data["message"])

        # Step 2: Update the drawing status to validated
        print("Set drawing status to validated")
        file_name = data["filename"]
        response = requests.put(
            f"{self.ADMIN_URL}/drawing/{file_name}",
            json={"validated": True},
            timeout=5,
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("updated successfully.", response.json()["message"])

        # Step 3: Call GET /unvalidated_drawing again
        print("Get second unvalidated drawing")
        response = requests.get(f"{self.ADMIN_URL}/unvalidated_drawing", timeout=5)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual("Drawing loaded successfully", data["message"])

        # Step 4: Delete the drawing
        print("Delete drawing")
        file_name = data["filename"]
        response = requests.delete(f"{self.ADMIN_URL}/drawing/{file_name}", timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertIn("deleted successfully.", response.json()["message"])

        # Step 5: Call GET /unvalidated_drawing again
        print("Ensure no unvalidated drawings left")
        response = requests.get(f"{self.ADMIN_URL}/unvalidated_drawing", timeout=5)
        self.assertEqual(response.status_code, 200)
        self.assertEqual("No unvalidated drawings found", response.json()["message"])


if __name__ == "__main__":
    unittest.main()
