import unittest
import time
import cv2

from src.modules.image_reading_module import ImageReadingModule
from src.modules.image_resize_module import ImageResizeModule


class TestImageModules(unittest.TestCase):
    def setUp(self):
        # Check if webcam is available
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.skipTest("Webcam is not available")
        self.cap.release()

    def test_image_resize_module(self):
        # Create ImageReadingModule with webcam source
        reading_module = ImageReadingModule(source=0)
        # Create ImageResizeModule with reading module as source and target size (128, 128)
        resize_module = ImageResizeModule(source_module=reading_module, image_size=(128, 128))

        # Start both modules
        reading_module.start()
        resize_module.start()

        # Let the modules run for a few seconds to capture and resize frames
        time.sleep(5)

        # Stop both modules
        reading_module.stop()
        resize_module.stop()

        # Check that the resized frame has the correct dimensions
        resized_frame = resize_module.value
        self.assertIsNotNone(resized_frame, "Resized frame should not be None")
        self.assertEqual(resized_frame.shape[1], 128, "Resized frame width should be 128")
        self.assertEqual(resized_frame.shape[0], 128, "Resized frame height should be 128")


if __name__ == "__main__":
    unittest.main()
