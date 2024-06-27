import os
import warnings

import cv2
import pandas as pd

from fuzzywuzzy import process
from src.traffic_sign_detection.traffic_sign import TrafficSign

DEFAULT_SIGN_CONFIG_PATH = "assets/traffic_sign_config.csv"


class TrafficSignBase:
    def __init__(self, config_csv_path: str = DEFAULT_SIGN_CONFIG_PATH, exclude_no_name: bool = False):
        self.config_csv_path = config_csv_path
        self.exclude_no_name = exclude_no_name
        self.sign_map = self._load_signs()

    def _load_signs(self):
        signs = {}
        df = pd.read_csv(self.config_csv_path)

        for _, row in df.iterrows():
            code = row['code']
            name = row.get('name', None)
            category = row.get('category', None)

            if self.exclude_no_name and (pd.isna(name) or pd.isna(category)):
                continue

            picture = self._load_image(code)
            sign = TrafficSign(
                code=code,
                name=name if pd.notna(name) else None,
                category=category if pd.notna(category) else None,
                picture=picture
            )
            signs[code] = sign

        return signs

    @staticmethod
    def _load_image(code: str):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            image_path = f"assets/traffic_sings/{code}.png"
            image = cv2.imread(image_path) if os.path.exists(image_path) else None

            if image is None:
                return None

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image

    def get_sign_by_code(self, code: str, soft: bool = True):
        if code in self.sign_map:
            return self.sign_map[code]
        elif soft:
            return None
        else:
            raise IndexError(f"Traffic sign with code {code} not found.")

    def search_sign_by_code(self, code: str, threshold: int = 91):
        all_codes = list(self.sign_map.keys())
        best_match, score = process.extractOne(code, all_codes)
        if score >= threshold:
            return self.sign_map[best_match]
        return None
