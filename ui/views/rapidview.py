# ui/views/rapidview.py
import tkinter as tk
from ui.base.updatable_frame import UpdatableFrame
from ui.utils.helpers import cv2_to_tk


class RapidView(UpdatableFrame):
    def __init__(self, master, ps, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps

        # Top row container
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        self.video1 = tk.Label(top_frame, text="Traffic Sign Left", bg="black", fg="white")
        self.video1.pack(side=tk.LEFT, padx=5, pady=5, expand=True)

        self.video2 = tk.Label(top_frame, text="Forward Distance", bg="black", fg="white")
        self.video2.pack(side=tk.LEFT, padx=5, pady=5, expand=True)

        self.video3 = tk.Label(top_frame, text="Traffic Sign Right", bg="black", fg="white")
        self.video3.pack(side=tk.LEFT, padx=5, pady=5, expand=True)

        # Bottom row for LED Strip
        self.video4 = tk.Label(self, text="LED Strip", bg="white", fg="black")
        self.video4.pack(side=tk.TOP, padx=5, pady=5, expand=True)

        self.add_after(100, self.update_traffic_signs)
        self.add_after(100, self.update_forward_distance)
        self.add_after(100, self.update_led_strip)

    def update_traffic_signs(self):
        if self.ps.traffic_sign_display is not None:
            left_img, right_img = self.ps.traffic_sign_display
            photo_left = cv2_to_tk(left_img)
            self.video1.config(image=photo_left)
            self.video1.image = photo_left
            photo_right = cv2_to_tk(right_img)
            self.video3.config(image=photo_right)
            self.video3.image = photo_right
        self.add_after(100, self.update_traffic_signs)

    def update_forward_distance(self):
        if hasattr(self.ps, "forward_distance_display") and self.ps.forward_distance_display is not None:
            photo = cv2_to_tk(self.ps.forward_distance_display)
            self.video2.config(image=photo)
            self.video2.image = photo
        self.add_after(100, self.update_forward_distance)

    def update_led_strip(self):
        if self.ps.led_strip_module.value is not None:
            photo = cv2_to_tk(self.ps.led_strip_module.value)
            self.video4.config(image=photo)
            self.video4.image = photo
        self.add_after(100, self.update_led_strip)
