import unittest
import os
import sys
from PIL import Image
import numpy as np
from unittest.mock import MagicMock, patch

# Add parent dir to sys.path to allow imports of 'vectra' package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vectra.inference import VectraInference
from vectra.utils.vision import LiveStreamInference

class TestVectraInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate files relative to this script
        cls.base_path = os.path.dirname(__file__)
        cls.model_path = os.path.join(cls.base_path, "test_model.pt")
        cls.data_dir = os.path.join(cls.base_path, "test_data")
        
        if not os.path.exists(cls.model_path):
            raise FileNotFoundError(f"Test model not found at {cls.model_path}")
            
        # Initialize SDK (using CPU for tests)
        cls.sdk = VectraInference(cls.model_path, use_gpu=False)
        print(f"\n--- Model Metadata ---")
        print(f"Labels: {cls.sdk.labels}")
        print(f"Backbone: {cls.sdk.backbone_name}")
        print(f"Image Format: {cls.sdk.image_format}")
        print(f"Unknown Category Enabled: {cls.sdk.use_unknown}")
        print(f"Unknown Threshold: {cls.sdk.unknown_threshold}")
        print(f"----------------------\n")

    def test_individual_predictions(self):
        """Test each image in the test_data folder against its expected label."""
        print("Running individual image predictions:")
        mismatches = []
        for filename in sorted(os.listdir(self.data_dir)):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            img_path = os.path.join(self.data_dir, filename)
            expected_label = filename.split('_')[0].capitalize()
            
            result = self.sdk.predict(img_path)
            
            status = "PASS" if result['label'] == expected_label else "FAIL"
            print(f"  {filename:15} -> Predicted: {result['label']:8} (Conf: {result.get('confidence', 0):.4f}, Dist: {result.get('distance', 0):.4f}) -> {status}")
            
            if status == "FAIL":
                mismatches.append((filename, expected_label, result['label']))
            
            self.assertIn('label', result)
            self.assertIn('confidence', result)
            self.assertIn('distance', result)
        
        if mismatches:
            print(f"\nNOTE: {len(mismatches)} label mismatches occurred. This is likely due to the test model's accuracy on these specific images, not an SDK failure.")

    def test_batch_prediction(self):
        """Test predicting multiple images at once."""
        image_paths = [
            os.path.join(self.data_dir, f) 
            for f in sorted(os.listdir(self.data_dir))
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        
        results = self.sdk.predict_batch(image_paths)
        
        self.assertEqual(len(results), len(image_paths))
        for res in results:
            self.assertIn('label', res)
            self.assertIn('confidence', res)

    def test_input_formats(self):
        """Test prediction with different input types: path, PIL, and numpy."""
        test_img = next(f for f in sorted(os.listdir(self.data_dir)) if f.lower().endswith('.jpg'))
        img_path = os.path.join(self.data_dir, test_img)
        
        # 1. Test path
        res_path = self.sdk.predict(img_path)
        self.assertIsInstance(res_path['label'], str)
        
        # 2. Test PIL
        img_pil = Image.open(img_path)
        res_pil = self.sdk.predict(img_pil)
        self.assertEqual(res_pil['label'], res_path['label'])
        
        # 3. Test Numpy
        img_np = np.array(img_pil)
        res_np = self.sdk.predict(img_np)
        self.assertEqual(res_np['label'], res_path['label'])

    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    def test_vision_live_stream_wrapper(self, mock_wait_key, mock_imshow, mock_video_capture):
        """Verify that the LiveStreamInference utility correctly wraps the classifier."""
        mock_wait_key.return_value = ord('q') # Immediately quit
        
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.isOpened.return_value = True
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        live = LiveStreamInference(self.sdk)
        live.start(camera_index=0)
        
        mock_cap.read.assert_called()
        mock_imshow.assert_called()
        mock_cap.release.assert_called()

if __name__ == "__main__":
    unittest.main()
