import cv2
from typing import Union


from src.modules.base_module import BaseModule


class ImageReadingModule(BaseModule):
    def __init__(self, source: Union[int, str], delay_seconds: float = 0.0):
        """Initialize the ImageReadingModule with a video source and delay."""
        super().__init__()
        self.delay_seconds = delay_seconds

        self.video_capture = cv2.VideoCapture(source)
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def perform(self):
        """Read frames from the video source and update the value."""
        try:
            while not self._stop_event.is_set():
                ret, frame = self.video_capture.read()
                if not ret:
                    break
                self.value = frame
                if self.delay_seconds > 0:
                    cv2.waitKey(int(self.delay_seconds * 1000))
        finally:
            self.video_capture.release()
