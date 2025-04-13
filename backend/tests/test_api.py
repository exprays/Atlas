import unittest
from fastapi.testclient import TestClient
import os
import shutil
import tempfile

from app.main import app

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Create temporary directory for test uploads
        self.test_dir = tempfile.mkdtemp()
        
        # Set up paths for test data
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        self.before_image = os.path.join(self.test_data_dir, 'before.tif')
        self.after_image = os.path.join(self.test_data_dir, 'after.tif')
        
        # Skip tests if test data doesn't exist
        if not os.path.exists(self.before_image) or not os.path.exists(self.after_image):
            self.skipTest("Test data not found")
    
    def tearDown(self):
        # Clean up temporary files
        shutil.rmtree(self.test_dir)
    
    def test_health_check(self):
        """Test API health check endpoint"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})
    
    def test_upload_images(self):
        """Test image upload endpoint"""
        with open(self.before_image, "rb") as before_file, open(self.after_image, "rb") as after_file:
            response = self.client.post(
                "/api/v1/detection/upload-images/",
                files={
                    "before_image": ("before.tif", before_file),
                    "after_image": ("after.tif", after_file),
                }
            )
            
            self.assertEqual(response.status_code, 200)
            self.assertIn("job_id", response.json())
            self.assertEqual(response.json()["status"], "uploaded")
            
            # Store job_id for processing test
            self.job_id = response.json()["job_id"]
    
    def test_process_images(self):
        """Test image processing endpoint"""
        # First upload images to get a job_id
        self.test_upload_images()
        
        # Then process the images
        response = self.client.post(f"/api/v1/detection/process/{self.job_id}")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["job_id"], self.job_id)
        # Processing might take time, so either "processing" or "completed" is acceptable
        self.assertIn(response.json()["status"], ["processing", "completed"])

if __name__ == '__main__':
    unittest.main()