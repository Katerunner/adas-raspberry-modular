import cv2
from src.modules.base_module import BaseModule
from typing import Tuple


class ImageResizeModule(BaseModule):
    def __init__(self, source_module: BaseModule, image_size: Tuple[int, int] = (256, 256)):
        """Initialize the ImageResizeModule with a source module and target image size."""
        super().__init__()
        self.source_module = source_module
        self.image_size = image_size

    def perform(self):
        """Resize frames from the source module to the specified image size."""
        while not self._stop_event.is_set():
            frame = self.source_module.value
            if frame is not None:
                resized_frame = cv2.resize(frame, self.image_size)
                self.value = resized_frame
