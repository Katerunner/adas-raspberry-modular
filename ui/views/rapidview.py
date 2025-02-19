# ui/views/rapidview.py
import cv2
import ttkbootstrap as tk
from ui.base.updatable_frame import UpdatableFrame
from ui.utils.helpers import cv2_to_tk


class RapidView(UpdatableFrame):
    def __init__(self, master, ps, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps

        # Configure grid: two rows and three columns.
        for col in range(3):
            self.columnconfigure(col, weight=1)
        for row in range(2):
            self.rowconfigure(row, weight=1)

        # Top row: three video labels
        self.video1 = tk.Label(self, text="Traffic Sign Left", background="black", foreground="white", anchor="center")
        self.video1.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        self.video2 = tk.Label(self, text="Forward Distance", background="black", foreground="white", anchor="center")
        self.video2.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        self.video3 = tk.Label(self, text="Traffic Sign Right", background="black", foreground="white", anchor="center")
        self.video3.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        # Bottom row: one label spanning all columns
        self.video4 = tk.Label(self, text="LED Strip", background="black", foreground="black", anchor="center")
        self.video4.grid(row=1, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        self.add_after(100, self.update_traffic_signs)
        self.add_after(100, self.update_forward_distance)
        self.add_after(100, self.update_led_strip)

    def get_dim_weights(self, c=0.9):
        return int(self.winfo_width() / 10 * c), int(self.winfo_height() / 5 * c)

    def update_traffic_signs(self):

        if self.ps.traffic_sign_display is not None:
            width_weight, height_weight = self.get_dim_weights()
            left_img, right_img = self.ps.traffic_sign_display
            left_img = cv2.resize(left_img, (width_weight * 4, height_weight * 4))
            right_img = cv2.resize(right_img, (width_weight * 4, height_weight * 4))

            photo_left = cv2_to_tk(left_img)
            self.video1.config(image=photo_left)
            self.video1.image = photo_left

            photo_right = cv2_to_tk(right_img)
            self.video3.config(image=photo_right)
            self.video3.image = photo_right

        self.add_after(100, self.update_traffic_signs)

    def update_forward_distance(self):
        if hasattr(self.ps, "forward_distance_display") and (self.ps.forward_distance_display is not None):
            width_weight, height_weight = self.get_dim_weights()
            if not width_weight or not height_weight:
                self.add_after(100, self.update_forward_distance)
                return

            fwd_image = self.ps.forward_distance_display
            fwd_image = cv2.resize(fwd_image, (width_weight * 2, height_weight * 4))

            photo = cv2_to_tk(fwd_image)

            self.video2.config(image=photo)
            self.video2.image = photo

        self.add_after(100, self.update_forward_distance)

    def update_led_strip(self):
        if self.ps.led_strip_module.value is not None:
            width_weight, height_weight = self.get_dim_weights()
            lds_image = self.ps.led_strip_module.value
            lds_image = cv2.resize(lds_image, (width_weight * 10, height_weight * 1))

            photo = cv2_to_tk(lds_image)

            self.video4.config(image=photo)
            self.video4.image = photo

        self.add_after(100, self.update_led_strip)
