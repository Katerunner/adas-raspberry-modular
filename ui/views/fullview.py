# ui/views/fullview.py
import tkinter as tk
from ui.base.updatable_frame import UpdatableFrame
from ui.utils.helpers import cv2_to_tk


class FullView(UpdatableFrame):
    def __init__(self, master, ps, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        self.video = tk.Label(self, text="Processed Detection", bg="black", fg="white")
        self.video.pack(fill=tk.BOTH, expand=True)
        self.add_after(50, self.update_video)

    def update_video(self):
        if self.ps.frame is not None:
            photo = cv2_to_tk(self.ps.frame)
            self.video.config(image=photo)
            self.video.image = photo
        self.add_after(50, self.update_video)
