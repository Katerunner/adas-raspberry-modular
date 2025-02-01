import tkinter as tk
from tkinter import ttk


class CalibrationView(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)

        # Create two columns: Left for frame video and sliders/buttons, right for perspective video.
        main_container = ttk.Frame(self)
        main_container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        main_container.columnconfigure(0, weight=1)
        main_container.columnconfigure(1, weight=0)

        # Left column container
        left_container = ttk.Frame(main_container)
        left_container.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_container.columnconfigure(0, weight=1)

        # Raw frame video placeholder (size 960x540, no resizing)
        self.video_frame = ttk.Frame(left_container, relief=tk.SUNKEN, width=960, height=540)
        self.video_frame.grid(row=0, column=0, sticky="nsew")
        self.video_frame.grid_propagate(False)
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack(expand=True, fill="both")

        # Sliders container below the video
        sliders_container = ttk.Frame(left_container)
        sliders_container.grid(row=1, column=0, sticky="ew", pady=(10, 10))
        sliders_container.columnconfigure(1, weight=1)

        lbl_ll = ttk.Label(sliders_container, text="Lower Left Corner")
        lbl_ll.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.slider_ll = ttk.Scale(sliders_container, from_=0, to=1, orient="horizontal")
        self.slider_ll.set(0.0)
        self.slider_ll.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        lbl_ul = ttk.Label(sliders_container, text="Upper Left Corner")
        lbl_ul.grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.slider_ul = ttk.Scale(sliders_container, from_=0, to=1, orient="horizontal")
        self.slider_ul.set(0.0)
        self.slider_ul.grid(row=1, column=1, sticky="ew", padx=5, pady=5)

        lbl_ur = ttk.Label(sliders_container, text="Upper Right Corner")
        lbl_ur.grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.slider_ur = ttk.Scale(sliders_container, from_=0, to=1, orient="horizontal")
        self.slider_ur.set(0.0)
        self.slider_ur.grid(row=2, column=1, sticky="ew", padx=5, pady=5)

        lbl_lr = ttk.Label(sliders_container, text="Lower Right Corner")
        lbl_lr.grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.slider_lr = ttk.Scale(sliders_container, from_=0, to=1, orient="horizontal")
        self.slider_lr.set(0.0)
        self.slider_lr.grid(row=3, column=1, sticky="ew", padx=5, pady=5)

        # Apply and Cancel buttons container below the sliders
        btn_container = ttk.Frame(left_container)
        btn_container.grid(row=2, column=0, sticky="ew", pady=(10, 10))
        btn_container.columnconfigure((0, 1), weight=1)
        self.apply_btn = ttk.Button(btn_container, text="Apply")
        self.apply_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")
        self.cancel_btn = ttk.Button(btn_container, text="Cancel")
        self.cancel_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

        # Right column: perspective video placeholder (size 135x540, no resizing)
        right_container = ttk.Frame(main_container)
        right_container.grid(row=0, column=1, sticky="nsew")
        self.perspective_frame = ttk.Frame(right_container, relief=tk.SUNKEN, width=135, height=540)
        self.perspective_frame.grid(row=0, column=0, sticky="nsew")
        self.perspective_frame.grid_propagate(False)
        self.perspective_label = tk.Label(self.perspective_frame)
        self.perspective_label.pack(expand=True, fill="both")
