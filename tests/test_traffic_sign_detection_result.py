import os
import unittest
from ultralytics import YOLO
import cv2
from src.traffic_sign_detection.traffic_sign_base import TrafficSignBase
from src.traffic_sign_detection.traffic_sign_detection_result import TrafficSignDetectionResult


class TestTrafficSignDetectionResult(unittest.TestCase):
    MODEL_PATH = 'ml/models/yolo/tsd_dfg_midcut_512.pt'
    IMAGE_PATH = 'assets/images/image.jpg'
    CONFIG_PATH = 'assets/traffic_sign_config.csv'
    IMAGE_SIZE = 512

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(cls.MODEL_PATH):
            cls.skipTest(cls, f"Model path {cls.MODEL_PATH} does not exist.")
        if not os.path.exists(cls.IMAGE_PATH):
            cls.skipTest(cls, f"Image path {cls.IMAGE_PATH} does not exist.")
        if not os.path.exists(cls.CONFIG_PATH):
            cls.skipTest(cls, f"Config path {cls.CONFIG_PATH} does not exist.")

        cls.model = YOLO(cls.MODEL_PATH)
        cls.image = cv2.imread(cls.IMAGE_PATH)
        cls.image = cv2.cvtColor(cls.image, cv2.COLOR_BGR2RGB)
        cls.traffic_sign_base = TrafficSignBase(cls.CONFIG_PATH)

    def test_traffic_sign_detection_result(self):
        results = self.model(self.image, imgsz=self.IMAGE_SIZE)
        yolo_result = results[0]

        detection_result = TrafficSignDetectionResult(yolo_result, self.traffic_sign_base)

        # Check if the detection result contains detected signs
        self.assertTrue(len(detection_result.detected_signs) > 0, "No traffic signs detected.")

        for detected_sign in detection_result.detected_signs:
            # Check if each detected sign has a valid class_id
            self.assertIsNotNone(detected_sign.class_id, "Detected sign class_id is None.")
            # Check if confidence is within the valid range [0, 1]
            self.assertGreaterEqual(detected_sign.confidence, 0.0, "Confidence score is less than 0.0.")
            self.assertLessEqual(detected_sign.confidence, 1.0, "Confidence score is greater than 1.0.")
            # Check if the bounding box coordinates are valid
            self.assertTrue(all(0.0 <= coord <= 1.0 for coord in detected_sign.xyxyn),
                            "Bounding box coordinates are out of range.")
            # Check if the sign image is not None
            self.assertIsNotNone(detected_sign.image, "Detected sign image is None.")
            # Check if the traffic sign information is correctly fetched
            self.assertIsNotNone(detected_sign.traffic_sign, "Traffic sign information could not be fetched.")

    @classmethod
    def tearDownClass(cls):
        del cls.model
        del cls.image
        del cls.traffic_sign_base


if __name__ == '__main__':
    unittest.main()
