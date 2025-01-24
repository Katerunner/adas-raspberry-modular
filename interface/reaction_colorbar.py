from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QRadialGradient, QBrush
from PyQt6.QtCore import QTimer, QRectF
import sys


class ReactionColorBar(QWidget):
    def __init__(self, resolution=100, parent=None):
        super().__init__(parent)
        self.resolution = resolution
        self.normal_color = QColor("gray")
        self.flash_color = None
        self.flash_rect = None
        self.flash_timer = QTimer()
        self.flash_timer.setSingleShot(True)
        # noinspection PyUnresolvedReferences
        self.flash_timer.timeout.connect(self.clear_flash)
        self.setMinimumHeight(30)
        self.end_alpha = 1.0

    def highlight(self, position: float, width: float, duration: float = 3, color: str = "red", end_alpha: float = 1.0):
        if not (0.0 <= position <= 1.0):
            raise ValueError("Position must be between 0.0 and 1.0")
        if not (0.01 <= width <= 1.0):
            raise ValueError("Width must be between 0.01 and 1.0")
        if not (0.0 <= end_alpha <= 1.0):
            raise ValueError("End alpha must be between 0.0 and 1.0")

        self.flash_color = QColor(color)
        self.end_alpha = end_alpha
        bar_width = self.width()
        center_x = (1 - position) * bar_width
        radius = width * bar_width / 2

        self.flash_rect = (center_x, self.height() / 2, radius)
        self.flash_timer.start(int(duration * 1000))
        self.update()

    def clear_flash(self):
        self.flash_color = None
        self.flash_rect = None
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        bar_width = self.width()
        part_width = bar_width / self.resolution

        for i in range(self.resolution):
            start_x = i * part_width
            rect = QRectF(start_x, 0, part_width, self.height())
            painter.fillRect(rect, self.normal_color)

        if self.flash_color and self.flash_rect:
            center_x, center_y, radius = self.flash_rect
            gradient = QRadialGradient(center_x, center_y, radius)
            gradient.setColorAt(0.0, self.flash_color)
            transparent_color = QColor(self.flash_color)
            transparent_color.setAlphaF(self.end_alpha)
            gradient.setColorAt(1.0, transparent_color)
            brush = QBrush(gradient)
            painter.setBrush(brush)
            painter.setPen(QColor(0, 0, 0, 0))  # Transparent pen
            painter.drawEllipse(center_x - radius, center_y - radius, radius * 2, radius * 2)
