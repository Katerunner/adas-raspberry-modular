import tkinter as tk
from tkinter import ttk

class Sidebar(ttk.Frame):
    def __init__(self, parent, show_frame_callback, start_callback, stop_callback):
        super().__init__(parent, width=150, relief=tk.RIDGE, padding=10)
        self.grid_propagate(False)
        title_lbl = ttk.Label(self, text="Menu", font=("Arial", 14))
        title_lbl.pack(pady=(0, 20))
        dash_btn = ttk.Button(self, text="Dashboard", command=lambda: show_frame_callback("dashboard"))
        dash_btn.pack(pady=5, fill="x")
        settings_btn = ttk.Button(self, text="Settings", command=lambda: show_frame_callback("settings"))
        settings_btn.pack(pady=5, fill="x")
        calib_btn = ttk.Button(self, text="Calibration", command=lambda: show_frame_callback("calibration"))
        calib_btn.pack(pady=5, fill="x")
        # Place video start/stop buttons at the bottom of the sidebar
        self.start_btn = ttk.Button(self, text="Start Video", command=start_callback)
        self.start_btn.pack(side="bottom", pady=5, fill="x")
        self.stop_btn = ttk.Button(self, text="Stop Video", command=stop_callback)
        self.stop_btn.pack(side="bottom", pady=5, fill="x")
