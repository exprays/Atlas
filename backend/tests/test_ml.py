import unittest
import os
import torch
import numpy as np
from app.ml.inferencer.predictor import ChangeDetectionPredictor
from app.ml.models.unet import initialize_model

class TestML(unittest.TestCase):
    def setUp(self):
        # Set up paths for test data
        self.test_data_dir = os.path.join(os.path.dirname(__file__), 'test_data')
        self.before_image = os.path.join(self.test_data_dir, 'before.tif')
        self.after_image = os.path.join(self.test_data_dir, 'after.tif')
        
        # Skip tests if test data doesn't exist
        if not os.path.exists(self.before_image) or not os.path.exists(self.after_image):
            self.skipTest("Test data not found")
    
    def test_model_initialization(self):
        """Test that the model can be initialized"""
        model = initialize_model(n_channels=6, n_classes=1)
        self.assertIsNotNone(model)
        self.assertEqual(model.n_channels, 6)
        self.assertEqual(model.n_classes, 1)
    
    def test_model_forward(self):
        """Test model forward pass"""
        model = initialize_model(n_channels=6, n_classes=1)
        model.eval()
        
        # Create random input tensor
        x = torch.randn(1, 6, 256, 256)
        
        # Run forward pass
        with torch.no_grad():
            output = model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (1, 1, 256, 256))
    
    def test_predictor_initialization(self):
        """Test that the predictor can be initialized"""
        predictor = ChangeDetectionPredictor()
        self.assertIsNotNone(predictor)
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_predictor_gpu_device(self):
        """Test that the predictor can use GPU"""
        predictor = ChangeDetectionPredictor(device=torch.device('cuda'))
        self.assertEqual(predictor.device.type, 'cuda')

if __name__ == '__main__':
    unittest.main()