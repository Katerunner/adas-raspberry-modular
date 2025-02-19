# ui/views/fullview.py
import cv2
import ttkbootstrap as tk
from ui.base.updatable_frame import UpdatableFrame
from ui.utils.helpers import cv2_to_tk


class FullView(UpdatableFrame):
    def __init__(self, master, ps, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        # Create a label with anchor set to "center"
        self.video = tk.Label(self, text="Processed Detection",
                              background="black", foreground="white",
                              anchor="center")
        # Pack with expand=True and anchor center so the label is centered in the container.
        self.video.pack(expand=True, anchor="center")
        self.add_after(50, self.update_video)

    def update_video(self):
        image_height = int(self.winfo_height() * 0.9)
        image_width = int(self.winfo_width() * 0.9)
        image = self.ps.frame
        if image is not None:
            image = cv2.resize(image, (image_width, image_height))
            photo = cv2_to_tk(image)
            self.video.config(image=photo)
            self.video.image = photo
        self.add_after(50, self.update_video)
