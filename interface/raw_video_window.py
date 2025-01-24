import numpy as np
from PyQt6.QtGui import QImage, QPixmap, QPainter
from interface.rounded_widget import RoundedWidget


class RawVideoWindow(RoundedWidget):
    def __init__(self, color="black", *args, **kwargs):
        super().__init__(color, *args, **kwargs)
        self.frame = None  # To store the current frame as a NumPy array

    def update_frame(self, frame: np.ndarray):
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Frame must be a valid NumPy array.")
        self.frame = frame
        self.update()  # Trigger a repaint

    def paintEvent(self, event):
        super().paintEvent(event)

        if self.frame is not None:
            height, width, channels = self.frame.shape
            if channels != 3:
                raise ValueError("Frame must have 3 channels (RGB).")

            image = QImage(self.frame.data, width, height, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)

            painter = QPainter(self)
            target_rect = self.rect()
            painter.drawPixmap(target_rect, pixmap)
