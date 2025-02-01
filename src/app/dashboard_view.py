import tkinter as tk
from tkinter import ttk


class DashboardView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Video frames container remains the same, but without buttons.
        video_frame = ttk.Frame(self)
        video_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        video_frame.columnconfigure(0, weight=3)
        video_frame.columnconfigure(1, weight=1)

        # Main 16:9 video placeholder exactly 480x270 (no resizing)
        self.video16_frame = ttk.Frame(video_frame, relief=tk.SUNKEN, width=480, height=270)
        self.video16_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        self.video16_frame.grid_propagate(False)
        self.dashboard_video_label = tk.Label(self.video16_frame)
        self.dashboard_video_label.pack(expand=True, fill="both")

        # Vertical video placeholder exactly 120x270
        self.video_vert_frame = ttk.Frame(video_frame, relief=tk.SUNKEN, width=120, height=270)
        self.video_vert_frame.grid(row=0, column=1, sticky="nsew")
        self.video_vert_frame.grid_propagate(False)
        self.vertical_video_label = tk.Label(self.video_vert_frame)
        self.vertical_video_label.pack(expand=True, fill="both")

        # LED Bar horizontal video placeholder exactly 600x100 remains as is.
        self.led_bar_frame = ttk.Frame(self, relief=tk.SUNKEN, width=600, height=100)
        self.led_bar_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.led_bar_frame.grid_propagate(False)
        self.led_bar_label = tk.Label(self.led_bar_frame, text="LED Bar Placeholder", background="black",
                                      foreground="white")
        self.led_bar_label.pack(expand=True, fill="both")
