import cv2
import numpy as np
import torch
from PIL import Image

from src.modules.base_module import BaseModule
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEFAULT_IMAGE_PROCESSOR_PATH = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
DEFAULT_DEPTH_DETECTION_MODEL_PATH = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"


class DepthDetectionModule(BaseModule):
    def __init__(self, source_module: BaseModule,
                 image_processor_path: str = DEFAULT_IMAGE_PROCESSOR_PATH,
                 depth_detection_model_path: str = DEFAULT_DEPTH_DETECTION_MODEL_PATH):
        super().__init__()
        self.source_module = source_module
        self.image_processor = AutoImageProcessor.from_pretrained(image_processor_path)
        self.model = AutoModelForDepthEstimation.from_pretrained(depth_detection_model_path)

    def _predict_frame(self, frame: np.ndarray):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        inputs = self.image_processor(images=pil_image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth

        depth_map = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(frame.shape[0], frame.shape[1]),
            mode="bicubic",
            align_corners=False,
        ).squeeze().cpu().numpy()

        return depth_map

    def perform(self):
        try:
            while not self._stop_event.is_set():
                frame = self.source_module.value

                if frame is not None:
                    self.value = self._predict_frame(frame=frame)
        finally:
            cv2.destroyAllWindows()
