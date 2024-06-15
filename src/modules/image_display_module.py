import cv2
from src.modules.base_module import BaseModule


class ImageDisplayModule(BaseModule):
    def __init__(self, source_module: BaseModule, frames_per_second: float = 60.0, escape_key: str = 'q'):
        """Initialize the ImageDisplayModule with a source module, frames per second, and escape key."""
        super().__init__()
        self.source_module = source_module
        self.frames_per_second = frames_per_second
        self.escape_key = escape_key

    def perform(self):
        """Display frames from the source module."""
        try:
            while not self._stop_event.is_set():
                frame = self.source_module.value
                if frame is not None:
                    cv2.imshow('frame', frame)

                if cv2.waitKey(int(1000 / self.frames_per_second)) & 0xFF == ord(self.escape_key):
                    break
        finally:
            cv2.destroyAllWindows()
