# ui/views/calibration.py
import tkinter as tk
import cv2
import numpy as np
from ui.base.updatable_frame import UpdatableFrame
from ui.utils.helpers import cv2_to_tk, parse_matrix


class DraggablePoint:
    def __init__(self, canvas, x, y, visual_radius, update_callback, canvas_width, canvas_height):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.visual_radius = visual_radius
        self.hit_radius = visual_radius + 15
        self.update_callback = update_callback
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.debounce_after_id = None  # For debouncing update calls

        self.hit_area = canvas.create_oval(
            x - self.hit_radius, y - self.hit_radius,
            x + self.hit_radius, y + self.hit_radius,
            fill="", outline=""
        )
        self.circle = canvas.create_oval(
            x - self.visual_radius, y - self.visual_radius,
            x + self.visual_radius, y + self.visual_radius,
            fill='red', outline='black', width=2
        )
        self.text = canvas.create_text(
            x + self.visual_radius + 5, y - self.visual_radius - 5,
            text=self.get_text(), anchor='nw',
            fill='white', font=('Arial', 10, 'bold')
        )
        for tag in (self.hit_area, self.circle, self.text):
            canvas.tag_bind(tag, "<ButtonPress-1>", self.on_press)
            canvas.tag_bind(tag, "<B1-Motion>", self.on_motion)
            canvas.tag_bind(tag, "<ButtonRelease-1>", self.on_release)

    def get_text(self):
        norm_x = self.x / self.canvas_width
        norm_y = self.y / self.canvas_height
        return f"({norm_x:.2f}, {norm_y:.2f})"

    def on_press(self, event):
        self.on_motion(event)

    def on_motion(self, event):
        new_x = max(0, min(event.x, self.canvas_width))
        new_y = max(0, min(event.y, self.canvas_height))
        self.x = new_x
        self.y = new_y
        self.canvas.coords(self.hit_area,
                           self.x - self.hit_radius, self.y - self.hit_radius,
                           self.x + self.hit_radius, self.y + self.hit_radius)
        self.canvas.coords(self.circle,
                           self.x - self.visual_radius, self.y - self.visual_radius,
                           self.x + self.visual_radius, self.y + self.visual_radius)
        self.canvas.coords(self.text,
                           self.x + self.visual_radius + 5, self.y - self.visual_radius - 5)
        self.canvas.itemconfigure(self.text, text=self.get_text())

        if self.debounce_after_id is not None:
            self.canvas.after_cancel(self.debounce_after_id)
        self.debounce_after_id = self.canvas.after(100, self.update_callback)

    def on_release(self, event):
        if self.debounce_after_id is not None:
            self.canvas.after_cancel(self.debounce_after_id)
            self.debounce_after_id = None
        self.update_callback()


class CalibrationEditor(UpdatableFrame):
    def __init__(self, master, ps, settings, apply_callback, cancel_callback, *args, **kwargs):
        """
        ps: An object whose image_reading_module.value is a cv2 image.
        settings: A dictionary (from MainApplicationFrame) that holds persistent settings.
        apply_callback: Called with new calibration parameters when Apply is pressed.
        cancel_callback: Called when Cancel is pressed.
        """
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        self.settings = settings
        self.apply_callback = apply_callback
        self.cancel_callback = cancel_callback
        self.default_points = None

        # Get frame dimensions (default if not available)
        active_frame = self.ps.image_reading_module.value
        if active_frame is not None:
            self.main_height, self.main_width = active_frame.shape[:2]
        else:
            self.main_width, self.main_height = 640, 480

        self.persp_width = 128
        self.persp_height = self.main_height

        # Create canvases for main and perspective view.
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.canvas_main = tk.Canvas(self.canvas_frame, width=self.main_width, height=self.main_height, bg="black")
        self.canvas_main.grid(row=0, column=0, sticky="nsew")
        self.canvas_persp = tk.Canvas(self.canvas_frame, width=self.persp_width, height=self.persp_height, bg="black")
        self.canvas_persp.grid(row=0, column=1, padx=5, sticky="nsew")
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(1, weight=0)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        self.canvas_main_image = self.canvas_main.create_image(0, 0, anchor='nw')
        self.canvas_persp_image = self.canvas_persp.create_image(0, 0, anchor='nw')

        # Load persistent calibration from settings if available; otherwise use factory defaults.
        # Persistent order: BL, BR, TL, TR.
        if "src_weights" in self.settings and self.settings["src_weights"]:
            matrix = parse_matrix(self.settings["src_weights"], 4, 2)
            persistent_points = [(float(r[0]), float(r[1])) for r in matrix]
        else:
            persistent_points = [
                (0.0, 1.0),  # Bottom Left
                (1.0, 1.0),  # Bottom Right
                (0.47, 0.47),  # Top Left
                (0.53, 0.47)  # Top Right
            ]
            self.settings["src_weights"] = "0.0,1.0;1.0,1.0;0.47,0.47;0.53,0.47"

        # Save a copy for cancellation.
        self.default_points = persistent_points.copy() if self.default_points is None else self.default_points

        # Create draggable points in persistent order.
        self.points = []
        self.visual_radius = 5
        for norm in persistent_points:
            x = norm[0] * self.main_width
            y = norm[1] * self.main_height
            dp = DraggablePoint(self.canvas_main, x, y, self.visual_radius, self.update_all, self.main_width,
                                self.main_height)
            self.points.append(dp)

        # Create polygon overlay.
        ordered = [self.points[0], self.points[2], self.points[3], self.points[1]]
        coords = []
        for dp in ordered:
            coords.extend([dp.x, dp.y])
        self.polygon = self.canvas_main.create_polygon(coords, outline='yellow', fill='', width=2)

        # Add Apply and Cancel buttons.
        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.apply_btn = tk.Button(self.button_frame, text="Apply Calibration", command=self.on_apply)
        self.apply_btn.pack(side='left', padx=10)
        self.cancel_btn = tk.Button(self.button_frame, text="Cancel", command=self.on_cancel)
        self.cancel_btn.pack(side='left', padx=10)

        self.add_after(500, self.update_all)

    def get_normalized_coordinates(self):
        # Return coordinates in persistent order (BL, BR, TL, TR).
        return [(dp.x / self.main_width, dp.y / self.main_height) for dp in self.points]

    def on_apply(self):
        coords = self.get_normalized_coordinates()
        # Build new persistent string in order BL, BR, TL, TR.
        new_src_weights = ";".join([f"{x:.2f},{y:.2f}" for x, y in coords])
        print("Applied Calibration Coordinates (persistent order):", coords)
        self.settings["src_weights"] = new_src_weights
        self.apply_callback({"src_weights": new_src_weights})

    def on_cancel(self):
        # Revert draggable points to the persistent calibration from settings.
        if "src_weights" in self.settings and self.settings["src_weights"]:
            matrix = parse_matrix(self.settings["src_weights"], 4, 2)
            persistent = [(float(r[0]), float(r[1])) for r in matrix]
        else:
            persistent = self.default_points
        for i, (nx, ny) in enumerate(persistent):
            self.points[i].x = nx * self.main_width
            self.points[i].y = ny * self.main_height
        self.update_all()
        # Reset settings["src_weights"] to factory default.
        self.settings["src_weights"] = "0.0,1.0;1.0,1.0;0.47,0.47;0.53,0.47"
        self.cancel_callback()

    def update_all(self):
        if self.ps.image_reading_module.value is None:
            self.canvas_main.delete("all")
            self.canvas_main.create_text(
                self.main_width // 2, self.main_height // 2,
                text="Start Video", fill="white", font=("Arial", 20)
            )
            self.canvas_persp.delete("all")
            self.canvas_persp.create_text(
                self.persp_width // 2, self.persp_height // 2,
                text="Start Video", fill="white", font=("Arial", 12)
            )
            self.add_after(500, self.update_all)
            return

        self.main_cv2_img = self.ps.image_reading_module.value
        resized_main = cv2.resize(self.main_cv2_img, (self.main_width, self.main_height))
        photo_main = cv2_to_tk(resized_main)
        self.canvas_main.itemconfig(self.canvas_main_image, image=photo_main)
        self.photo_main = photo_main

        # For visualization, reorder persistent points [BL, BR, TL, TR] to [BL, TL, TR, BR].
        ordered = [self.points[0], self.points[2], self.points[3], self.points[1]]
        coords = []
        for p in ordered:
            coords.extend([p.x, p.y])
        self.canvas_main.coords(self.polygon, *coords)

        src_pts = np.array([[p.x, p.y] for p in ordered], dtype=np.float32)
        dst_pts = np.array([
            [0, self.persp_height],
            [0, 0],
            [self.persp_width, 0],
            [self.persp_width, self.persp_height]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        persp_img = cv2.warpPerspective(self.main_cv2_img, M, (self.persp_width, self.persp_height))
        photo_persp = cv2_to_tk(persp_img)
        self.canvas_persp.itemconfig(self.canvas_persp_image, image=photo_persp)
        self.photo_persp = photo_persp

        self.add_after(500, self.update_all)
