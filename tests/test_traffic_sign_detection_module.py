import random
import unittest
import time
from src.modules.base_module import BaseModule
from src.modules.image_display_module import ImageDisplayModule
from src.modules.image_reading_module import ImageReadingModule
from src.modules.traffic_sign_detection_module import TrafficSignDetectionModule
from src.pools.base_pool import BasePool

MODEL_WEIGHTS = "ml/models/yolo/yolov8n_ncnn_model"
MODEL_SOURCE = "assets/videos/video_1.mp4"
WAIT_TIME = 20


class TestBasePool(unittest.TestCase):
    def setUp(self):
        image_reading_module = ImageReadingModule(source=MODEL_SOURCE, delay_seconds=1 / 60.0)
        self.image_reading_module = image_reading_module
        self.tsd_module = TrafficSignDetectionModule(source_module=image_reading_module, model_weights=MODEL_WEIGHTS)
        self.image_display_module = ImageDisplayModule(source_module=self.tsd_module, frames_per_second=60.0)

    def test_standard_tsd_module_performance(self):
        self.image_reading_module.start()
        self.tsd_module.start()
        self.image_display_module.start()

        time.sleep(200)

        self.image_reading_module.stop()
        self.tsd_module.stop()
        self.image_display_module.stop()


if __name__ == "__main__":
    unittest.main()
