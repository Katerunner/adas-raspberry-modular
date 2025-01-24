from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QCheckBox, QSlider
from PyQt6.QtCore import Qt


class SettingsWindow(QWidget):
    def __init__(self, description="Random settings", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle("Settings Window")
        self.resize(400, 300)

        # Main layout
        layout = QVBoxLayout()
        self.setLayout(layout)

        # Description
        self.description_label = QLabel(description)
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.description_label)

        # Checkboxes
        self.checkboxes = []
        for i in range(3):
            checkbox = QCheckBox(f"Option {i + 1}")
            layout.addWidget(checkbox)
            self.checkboxes.append(checkbox)

        # Float sliders
        self.sliders = []
        for i in range(3):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(50)  # Default midpoint value
            slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            slider.setTickInterval(10)
            layout.addWidget(slider)
            self.sliders.append(slider)
