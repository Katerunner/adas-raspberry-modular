import sys
from PyQt6.QtCore import QSize, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QLabel
from PyQt6.QtGui import QImage, QPixmap
import cv2
from interface.rounded_widget import RoundedWidget
from src.modules.image_reading_module import ImageReadingModule


class MainVideoWidget(RoundedWidget):
    def __init__(self, *args, **kwargs):
        super().__init__("blue", *args, **kwargs)
        self.setFixedSize(self.get_aspect_ratio_size(800, 600, 4, 3))

    def get_aspect_ratio_size(self, width, height, ratio_width, ratio_height):
        if width / ratio_width < height / ratio_height:
            height = int(width * ratio_height / ratio_width)
        else:
            width = int(height * ratio_width / ratio_height)
        return QSize(width, height)


class NavPanelWidget(QWidget):
    def __init__(self, click_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.click_callback = click_callback
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        # Create buttons for the nav panel
        for i in range(5):
            icon = QPushButton(f"Icon {i+1}")
            icon.setFixedSize(70, 70)
            if i == 0:  # Connect first icon to video start/stop
                icon.clicked.connect(self.click_callback)
            layout.addWidget(icon)

        self.setLayout(layout)


class SubVideoWidget(RoundedWidget):
    def __init__(self, *args, **kwargs):
        super().__init__("red", *args, **kwargs)
        self.setFixedSize(self.get_aspect_ratio_size(300, 225, 4, 3))
        self.video_label = QLabel(self)
        self.video_label.setGeometry(0, 0, 300, 225)

    def get_aspect_ratio_size(self, width, height, ratio_width, ratio_height):
        if width / ratio_width < height / ratio_height:
            height = int(width * ratio_height / ratio_width)
        else:
            width = int(height * ratio_width / ratio_height)
        return QSize(width, height)

    def update_video_frame(self, frame):
        """Update the displayed frame in the sub-video widget."""
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = QImage(frame.data, frame.shape[1], frame.shape[0], frame.strides[0], QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(image)
            self.video_label.setPixmap(pixmap)


class RightColumnSquaresWidget(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        layout = QHBoxLayout()
        top_grid = QVBoxLayout()
        bottom_grid = QVBoxLayout()

        for _ in range(2):
            square = RoundedWidget("yellow")
            square.setFixedSize(100, 100)
            top_grid.addWidget(square)

        for _ in range(2):
            square = RoundedWidget("orange")
            square.setFixedSize(100, 100)
            bottom_grid.addWidget(square)

        layout.addLayout(top_grid)
        layout.addLayout(bottom_grid)

        self.setLayout(layout)


class BottomBarWidget(QFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setStyleSheet("background-color: black;")
        self.setFixedHeight(20)  # Thin bar across the bottom


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Driver Assistance Interface")
        self.setGeometry(100, 100, 1280, 960)
        self.setStyleSheet("background-color: gray;")

        self.is_video_playing = False

        # Video module for raw video
        self.video_module = ImageReadingModule(source="assets/videos/video_1.mp4")

        # Timer to update the video frame
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_sub_video)

        # Main layout
        main_layout = QVBoxLayout()

        # Create the top section with horizontal layout
        top_layout = QHBoxLayout()

        # Left column (nav panel)
        nav_panel = NavPanelWidget(self.toggle_video_playback)

        # Central frame (main video)
        main_video = MainVideoWidget()

        # Right column (sub-video and squares)
        right_column_layout = QVBoxLayout()
        self.sub_video = SubVideoWidget()

        # Center the sub-video widget horizontally
        right_column_layout.addWidget(self.sub_video, alignment=Qt.AlignmentFlag.AlignHCenter)

        # Add the grid of squares in the right column
        right_column_squares = RightColumnSquaresWidget()
        right_column_layout.addWidget(right_column_squares)

        # Add left, main video, and right layouts to top layout
        top_layout.addWidget(nav_panel)
        top_layout.addWidget(main_video)
        top_layout.addLayout(right_column_layout)

        # Add top layout to the main layout
        main_layout.addLayout(top_layout)

        # Bottom bar (thin across the screen)
        bottom_bar = BottomBarWidget()

        # Add bottom bar to main layout
        main_layout.addWidget(bottom_bar)

        # Central widget
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def toggle_video_playback(self):
        """Start/stop video playback when the first icon is clicked."""
        if self.is_video_playing:
            self.video_module.stop()
            self.timer.stop()
            self.is_video_playing = False
        else:
            self.video_module.start()
            self.timer.start(30)  # Update video every 30 ms (approximately 30 FPS)
            self.is_video_playing = True

    def update_sub_video(self):
        """Update the sub-video widget with the current frame."""
        frame = self.video_module.value
        if frame is not None:
            self.sub_video.update_video_frame(frame)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
