import unittest
import os

from src.traffic_sign_detection.traffic_sign import TrafficSign
from src.traffic_sign_detection.traffic_sign_base import TrafficSignBase

DEFAULT_SIGN_CONFIG_PATH = "assets/traffic_sign_config.csv"
SAMPLE_IMAGE_PATH = "assets/traffic_sings/I-1.png"


class TestTrafficSignBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(DEFAULT_SIGN_CONFIG_PATH) or not os.path.exists(SAMPLE_IMAGE_PATH):
            raise unittest.SkipTest("Required files not found. Skipping tests.")

    def setUp(self):
        self.traffic_sign_base = TrafficSignBase()

    def test_load_signs(self):
        self.assertEqual(len(self.traffic_sign_base.sign_map), 200)
        self.assertIn('I-1', self.traffic_sign_base.sign_map)
        self.assertIsInstance(self.traffic_sign_base.sign_map['I-1'], TrafficSign)
        self.assertEqual(self.traffic_sign_base.sign_map['I-1'].name, 'Bend to the left.')

    def test_load_image(self):
        image = self.traffic_sign_base._load_image('I-1')
        self.assertIsNotNone(image)

    def test_get_sign_by_code(self):
        sign = self.traffic_sign_base.get_sign_by_code('I-1')
        self.assertIsNotNone(sign)
        self.assertEqual(sign.name, 'Bend to the left.')

        sign = self.traffic_sign_base.get_sign_by_code('I-999', soft=True)
        self.assertIsNone(sign)

        with self.assertRaises(IndexError):
            self.traffic_sign_base.get_sign_by_code('I-999', soft=False)

    def test_search_sign_by_code(self):
        sign = self.traffic_sign_base.search_sign_by_code('I-1')
        self.assertIsNotNone(sign)
        self.assertEqual(sign.name, 'Bend to the left.')

        sign = self.traffic_sign_base.search_sign_by_code('I-999')
        self.assertIsNone(sign)

        sign = self.traffic_sign_base.search_sign_by_code('I-3', threshold=90)
        self.assertIsNotNone(sign)
        self.assertEqual(sign.name, 'Dangerous descent.')


if __name__ == '__main__':
    unittest.main()
