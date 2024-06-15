import cv2
import time
import unittest

import numpy as np

from src.modules.image_reading_module import ImageReadingModule
from src.modules.image_display_module import ImageDisplayModule


class TestImageModules(unittest.TestCase):
    def setUp(self):
        # Check if webcam is available
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.skipTest("Webcam is not available")
        self.cap.release()

    def test_image_modules(self):
        # Create ImageReadingModule with webcam source
        reading_module = ImageReadingModule(source=0)
        display_module = ImageDisplayModule(source_module=reading_module, frames_per_second=30.0, escape_key='q')

        # Start both modules
        reading_module.start()
        display_module.start()

        # Run display for 5 seconds and close
        time.sleep(5)
        display_module.stop()

        # Stop reading module and save the last frame
        reading_module.stop()
        last_frame = reading_module.value

        # Wait 2 seconds and check that the value has not changed
        time.sleep(2)
        self.assertTrue(np.array_equal(reading_module.value, last_frame))

        # Stop display module
        display_module.stop()


if __name__ == "__main__":
    unittest.main()
