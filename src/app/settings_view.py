import tkinter as tk
from tkinter import ttk


class SettingsView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        header_lbl = ttk.Label(self, text="Settings", font=("Arial", 16))
        header_lbl.pack(pady=(0, 20))

        brightness_lbl = ttk.Label(self, text="Brightness")
        brightness_lbl.pack(pady=(10, 0))
        self.brightness_slider = ttk.Scale(self, from_=0, to=100, orient="horizontal")
        self.brightness_slider.pack(padx=10, pady=5, fill="x")

        text_lbl = ttk.Label(self, text="Enter text:")
        text_lbl.pack(pady=(10, 0))
        self.text_entry = ttk.Entry(self)
        self.text_entry.pack(padx=10, pady=5, fill="x")

        contrast_lbl = ttk.Label(self, text="Contrast")
        contrast_lbl.pack(pady=(10, 0))
        self.contrast_slider = ttk.Scale(self, from_=0, to=100, orient="horizontal")
        self.contrast_slider.pack(padx=10, pady=5, fill="x")
