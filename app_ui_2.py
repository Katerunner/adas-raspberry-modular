import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import base64
import time

# -------------------------------
# Import Perception System Modules
# -------------------------------
from src.lane_detection.lane_curve_estimator import LaneCurveEstimator
from src.lane_detection.lane_processor_corrector import LaneProcessorCorrector
from src.modules.collision_warning_module import CollisionWarningModule
from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.image_reading_module import ImageReadingModule
from src.modules.lane_detection_module import LaneDetectionModule
from src.modules.led_strip_module import LEDStripModule
from src.modules.perspective_transformation_module import PerspectiveTransformationModule
from src.modules.pothole_detection_module import PotholeDetectionModule
from src.modules.speed_detection_module import SpeedDetectionModule
from src.modules.traffic_sign_detection_module import TrafficSignDetectionModule
from src.modules.traffic_sign_display_module import TrafficSignDisplayModule
from src.modules.forward_distance_module import ForwardDistanceModule
from src.system.perception_system import PerceptionSystem


#########################
# Base Class for Frames With After Callbacks
#########################
class UpdatableFrame(tk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.after_ids = []

    def add_after(self, delay, func):
        aid = self.after(delay, func)
        self.after_ids.append(aid)
        return aid

    def cancel_updates(self):
        for aid in self.after_ids:
            try:
                self.after_cancel(aid)
            except Exception as e:
                print("Error cancelling after id:", aid, e)
        self.after_ids = []

    def destroy(self):
        self.cancel_updates()
        super().destroy()


#########################
# Helper Functions
#########################
def parse_matrix(matrix_str, rows, cols):
    try:
        row_strs = matrix_str.split(";")
        matrix = []
        for r in row_strs:
            values = [float(x.strip()) for x in r.split(",")]
            matrix.append(values)
        arr = np.array(matrix, dtype=np.float32)
        if arr.shape != (rows, cols):
            raise ValueError("Matrix shape mismatch; expected ({}, {})".format(rows, cols))
        return arr
    except Exception as e:
        print("Error parsing matrix:", e)
        return None


def cv2_to_tk(cv_img):
    rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    return ImageTk.PhotoImage(im)


#########################
# DraggablePoint Class
#########################
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
        self.canvas.coords(
            self.hit_area,
            self.x - self.hit_radius, self.y - self.hit_radius,
            self.x + self.hit_radius, self.y + self.hit_radius
        )
        self.canvas.coords(
            self.circle,
            self.x - self.visual_radius, self.y - self.visual_radius,
            self.x + self.visual_radius, self.y + self.visual_radius
        )
        self.canvas.coords(
            self.text,
            self.x + self.visual_radius + 5, self.y - self.visual_radius - 5
        )
        self.canvas.itemconfigure(self.text, text=self.get_text())
        self.update_callback()

    def on_release(self, event):
        pass


#########################
# RapidView Class
#########################
class RapidView(UpdatableFrame):
    def __init__(self, master, ps, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        top_frame = tk.Frame(self)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.video1 = tk.Label(top_frame, text="Traffic Sign Left", bg="black", fg="white")
        self.video1.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        self.video2 = tk.Label(top_frame, text="Forward Distance", bg="black", fg="white")
        self.video2.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        self.video3 = tk.Label(top_frame, text="Traffic Sign Right", bg="black", fg="white")
        self.video3.pack(side=tk.LEFT, padx=5, pady=5, expand=True)
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


#########################
# FullView Class
#########################
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


#########################
# DashboardView Class
#########################
class DashboardView(UpdatableFrame):
    def __init__(self, master, ps, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.btn_rapid = tk.Button(btn_frame, text="Rapid View", command=self.show_rapid)
        self.btn_rapid.pack(side=tk.LEFT, padx=5)
        self.btn_full = tk.Button(btn_frame, text="Full View", command=self.show_full)
        self.btn_full.pack(side=tk.LEFT, padx=5)
        self.content_frame = tk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        self.current_subview = None
        self.show_rapid()

    def show_rapid(self):
        if self.current_subview:
            self.current_subview.destroy()
        self.current_subview = RapidView(self.content_frame, self.ps)
        self.current_subview.pack(fill=tk.BOTH, expand=True)

    def show_full(self):
        if self.current_subview:
            self.current_subview.destroy()
        self.current_subview = FullView(self.content_frame, self.ps)
        self.current_subview.pack(fill=tk.BOTH, expand=True)


#########################
# CalibrationEditor Class
#########################
class CalibrationEditor(UpdatableFrame):
    def __init__(self, master, ps, apply_callback, cancel_callback, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        self.apply_callback = apply_callback
        self.cancel_callback = cancel_callback

        self.main_width = 512
        self.main_height = 512
        self.persp_width = 128
        self.persp_height = 512

        self.input_images = []
        self.max_images = 6

        self.main_frame = tk.Frame(self)
        self.main_frame.grid(row=0, column=0, padx=10, pady=10)
        self.canvas_main = tk.Canvas(self.main_frame, width=self.main_width, height=self.main_height)
        self.canvas_main.grid(row=0, column=0)
        self.canvas_persp = tk.Canvas(self.main_frame, width=self.persp_width, height=self.persp_height)
        self.canvas_persp.grid(row=0, column=1, padx=5)

        if self.input_images:
            self.main_cv2_img = self.input_images[0]
        else:
            self.main_cv2_img = np.random.randint(0, 256, (self.main_height, self.main_width, 3), dtype=np.uint8)
        self.photo_main = cv2_to_tk(cv2.resize(self.main_cv2_img, (self.main_width, self.main_height)))
        self.canvas_main_image = self.canvas_main.create_image(0, 0, anchor='nw', image=self.photo_main)
        self.photo_persp = cv2_to_tk(cv2.resize(self.main_cv2_img, (self.persp_width, self.persp_height)))
        self.canvas_persp_image = self.canvas_persp.create_image(0, 0, anchor='nw', image=self.photo_persp)

        default_points = [
            (0.0, 1.0),
            (0.4, 0.5),
            (0.6, 0.5),
            (1.0, 1.0)
        ]
        self.points = []
        self.visual_radius = 5
        for norm in default_points:
            x = norm[0] * self.main_width
            y = norm[1] * self.main_height
            dp = DraggablePoint(self.canvas_main, x, y, self.visual_radius,
                                self.update_all, self.main_width, self.main_height)
            self.points.append(dp)
        coords = []
        for dp in self.points:
            coords.extend([dp.x, dp.y])
        self.polygon = self.canvas_main.create_polygon(coords, outline='yellow', fill='', width=2)

        self.coord_frame = tk.Frame(self)
        self.coord_frame.grid(row=1, column=0, columnspan=2, pady=5)
        self.coord_labels = []
        for dp in self.points:
            label = tk.Label(self.coord_frame, text=self.get_norm_text(dp), font=('Arial', 10))
            label.pack(side='left', padx=10)
            self.coord_labels.append(label)

        self.param_frame = tk.Frame(self)
        self.param_frame.grid(row=2, column=0, columnspan=2, pady=5)
        tk.Label(self.param_frame, text="Danger Zone Coefficients (row;row;...):").grid(row=0, column=0, sticky="w")
        self.entry_danger = tk.Entry(self.param_frame, width=50)
        self.entry_danger.grid(row=0, column=1, padx=5, pady=2)
        tk.Label(self.param_frame, text="Src Weights (row;row;...):").grid(row=1, column=0, sticky="w")
        self.entry_src = tk.Entry(self.param_frame, width=50)
        self.entry_src.grid(row=1, column=1, padx=5, pady=2)
        tk.Label(self.param_frame, text="Dst Weights (row;row;...):").grid(row=2, column=0, sticky="w")
        self.entry_dst = tk.Entry(self.param_frame, width=50)
        self.entry_dst.grid(row=2, column=1, padx=5, pady=2)
        settings = self.master.master.settings
        self.entry_danger.insert(0, settings.get("danger_zone_coefficients", "0.2,1.0;0.8,1.0;0.4,0.75;0.6,0.75"))
        self.entry_src.insert(0, settings.get("src_weights", "0.0,1.0;1.0,1.0;0.47,0.47;0.53,0.47"))
        self.entry_dst.insert(0, settings.get("dst_weights", "0.0,1.0;1.0,1.0;0.0,0.0;1.0,0.0"))

        # Create thumbnail label only once
        self.thumbnail_label = tk.Label(self, text="Select Calibration Image:")
        self.thumbnail_label.grid(row=3, column=0, columnspan=2, pady=5)

        self.thumbnail_frame = tk.Frame(self)
        self.thumbnail_frame.grid(row=4, column=0, columnspan=2, pady=5)
        self.thumbnail_buttons = []
        self.thumbnail_images = []

        btn_frame = tk.Frame(self)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=10)
        self.apply_btn = tk.Button(btn_frame, text="Apply Calibration", command=self.on_apply)
        self.apply_btn.pack(side='left', padx=10)
        self.cancel_btn = tk.Button(btn_frame, text="Cancel", command=self.on_cancel)
        self.cancel_btn.pack(side='left', padx=10)

        self.add_after(200, self.update_all)
        self.add_after(15000, self.update_registry)
        self.add_after(5000, self.update_thumbnails)

    def get_norm_text(self, dp):
        norm_x = dp.x / self.main_width
        norm_y = dp.y / self.main_height
        idx = self.points.index(dp) + 1
        return f"P{idx}: ({norm_x:.2f}, {norm_y:.2f})"

    def update_all(self):
        coords = []
        for dp in self.points:
            coords.extend([dp.x, dp.y])
        self.canvas_main.coords(self.polygon, *coords)
        for i, dp in enumerate(self.points):
            self.coord_labels[i].configure(text=self.get_norm_text(dp))
        src_pts = np.array([[dp.x, dp.y] for dp in self.points], dtype=np.float32)
        dst_pts = np.array([
            [0, self.persp_height],
            [0, 0],
            [self.persp_width, 0],
            [self.persp_width, self.persp_height]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        persp_img = cv2.warpPerspective(self.main_cv2_img, M, (self.persp_width, self.persp_height))
        self.photo_persp = cv2_to_tk(persp_img)
        self.canvas_persp.itemconfig(self.canvas_persp_image, image=self.photo_persp)
        self.add_after(200, self.update_all)

    def update_registry(self):
        current_img = self.ps.image_reading_module.value
        if current_img is not None:
            if len(self.input_images) == 0 or current_img.shape != self.input_images[-1].shape:
                self.input_images.append(current_img.copy())
            if len(self.input_images) > self.max_images:
                self.input_images.pop(0)
        self.add_after(15000, self.update_registry)

    def update_thumbnails(self):
        for widget in self.thumbnail_frame.winfo_children():
            if isinstance(widget, tk.Button):
                widget.destroy()
        self.thumbnail_buttons = []
        self.thumbnail_images = []
        for i, img in enumerate(self.input_images):
            thumb = cv2.resize(img, (80, 80))
            thumb_photo = cv2_to_tk(thumb)
            self.thumbnail_images.append(thumb_photo)
            btn = tk.Button(self.thumbnail_frame, image=thumb_photo, command=lambda i=i: self.change_image(i))
            btn.pack(side='left', padx=5)
            self.thumbnail_buttons.append(btn)
        self.add_after(5000, self.update_thumbnails)

    def change_image(self, index):
        if 0 <= index < len(self.input_images):
            self.main_cv2_img = self.input_images[index]
            self.photo_main = cv2_to_tk(self.main_cv2_img)
            self.canvas_main.itemconfig(self.canvas_main_image, image=self.photo_main)
            self.update_all()

    def get_coordinates(self):
        return [(dp.x / self.main_width, dp.y / self.main_height) for dp in self.points]

    def on_apply(self):
        coords = self.get_coordinates()
        print("Applied Calibration Coordinates:", coords)
        new_params = {
            "danger_zone_coefficients": self.entry_danger.get(),
            "src_weights": self.entry_src.get(),
            "dst_weights": self.entry_dst.get()
        }
        self.apply_callback({"coordinates": coords, **new_params})

    def on_cancel(self):
        default_points = [
            (0.0, 1.0),
            (0.4, 0.5),
            (0.6, 0.5),
            (1.0, 1.0)
        ]
        for i, norm in enumerate(default_points):
            self.points[i].x = norm[0] * self.main_width
            self.points[i].y = norm[1] * self.main_height
        self.update_all()
        self.cancel_callback()


#########################
# SettingsView Class
#########################
class SettingsView(UpdatableFrame):
    def __init__(self, master, ps, apply_callback, default_settings, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        self.apply_callback = apply_callback
        self.factory_settings = default_settings.copy()
        lbl = tk.Label(self, text="Settings", font=("Arial", 16))
        lbl.pack(pady=10)
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        self.params = {}
        self.create_general_tab()
        self.create_sign_tab()
        self.create_lane_tab()
        self.create_pothole_tab()
        self.create_collision_tab()
        self.create_speed_tab()
        self.create_sign_disp_tab()
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=10)
        self.apply_button = tk.Button(btn_frame, text="Apply", command=self.on_apply)
        self.apply_button.pack(side="left", padx=5)
        self.reset_button = tk.Button(btn_frame, text="Reset", command=self.on_reset)
        self.reset_button.pack(side="left", padx=5)

    def create_general_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="General Obj Det")
        row = 0

        def add_slider(label_text, var, frm, row, from_val, to_val, resolution, is_int=False):
            tk.Label(frm, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            scale = tk.Scale(frm, variable=var, from_=from_val, to=to_val, resolution=resolution, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            frm.grid_columnconfigure(1, weight=1)
            return row + 1

        var_det = tk.DoubleVar(value=self.factory_settings.get("general_detection_threshold", 0.3))
        var_conf = tk.DoubleVar(value=self.factory_settings.get("general_tracker_confidence_threshold", 0.7))
        var_feat = tk.DoubleVar(value=self.factory_settings.get("general_tracker_feature_threshold", 0.95))
        var_pos = tk.DoubleVar(value=self.factory_settings.get("general_tracker_position_threshold", 0.98))
        var_life = tk.IntVar(value=self.factory_settings.get("general_tracker_lifespan", 3))
        row = add_slider("Detection Threshold", var_det, tab, 0, 0, 1, 0.01)
        row = add_slider("Tracker Confidence Threshold", var_conf, tab, row, 0, 1, 0.01)
        row = add_slider("Tracker Feature Threshold", var_feat, tab, row, 0, 1, 0.01)
        row = add_slider("Tracker Position Threshold", var_pos, tab, row, 0, 1, 0.01)
        row = add_slider("Tracker Lifespan", var_life, tab, row, 1, 10, 1, is_int=True)
        self.params["general_detection_threshold"] = var_det
        self.params["general_tracker_confidence_threshold"] = var_conf
        self.params["general_tracker_feature_threshold"] = var_feat
        self.params["general_tracker_position_threshold"] = var_pos
        self.params["general_tracker_lifespan"] = var_life

    def create_sign_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Traffic Sign Det")
        row = 0

        def add_slider(label_text, var, frm, row, from_val, to_val, resolution, is_int=False):
            tk.Label(frm, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            scale = tk.Scale(frm, variable=var, from_=from_val, to=to_val, resolution=resolution, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            frm.grid_columnconfigure(1, weight=1)
            return row + 1

        var_det_sign = tk.DoubleVar(value=self.factory_settings.get("sign_detection_threshold", 0.7))
        var_reg_min = tk.IntVar(value=self.factory_settings.get("sign_registry_min_occurrences", 5))
        var_reg_life = tk.IntVar(value=self.factory_settings.get("sign_registry_lifetime", 5))
        var_cast_life = tk.IntVar(value=self.factory_settings.get("sign_casting_lifetime", 3))
        var_feat_sign = tk.DoubleVar(value=self.factory_settings.get("sign_tracker_feature_threshold", 0.95))
        var_pos_sign = tk.DoubleVar(value=self.factory_settings.get("sign_tracker_position_threshold", 0.95))
        row = add_slider("Detection Threshold", var_det_sign, tab, row, 0, 1, 0.01)
        row = add_slider("Registry Min Occurrences", var_reg_min, tab, row, 1, 20, 1, is_int=True)
        row = add_slider("Registry Lifetime", var_reg_life, tab, row, 1, 10, 1, is_int=True)
        row = add_slider("Casting Lifetime", var_cast_life, tab, row, 1, 10, 1, is_int=True)
        row = add_slider("Tracker Feature Threshold", var_feat_sign, tab, row, 0, 1, 0.01)
        row = add_slider("Tracker Position Threshold", var_pos_sign, tab, row, 0, 1, 0.01)
        self.params["sign_detection_threshold"] = var_det_sign
        self.params["sign_registry_min_occurrences"] = var_reg_min
        self.params["sign_registry_lifetime"] = var_reg_life
        self.params["sign_casting_lifetime"] = var_cast_life
        self.params["sign_tracker_feature_threshold"] = var_feat_sign
        self.params["sign_tracker_position_threshold"] = var_pos_sign

    def create_lane_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Lane Detection")
        row = 0

        def add_slider(label_text, var, frm, row, from_val, to_val, resolution, is_int=False):
            tk.Label(frm, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            scale = tk.Scale(frm, variable=var, from_=from_val, to=to_val, resolution=resolution, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            frm.grid_columnconfigure(1, weight=1)
            return row + 1

        var_conf_lane = tk.DoubleVar(value=self.factory_settings.get("lane_confidence_threshold", 0.2))
        row = add_slider("Confidence Threshold", var_conf_lane, tab, row, 0, 1, 0.01)
        self.params["lane_confidence_threshold"] = var_conf_lane

    def create_pothole_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Pothole Detection")
        row = 0

        def add_slider(label_text, var, frm, row, from_val, to_val, resolution, is_int=False):
            tk.Label(frm, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            scale = tk.Scale(frm, variable=var, from_=from_val, to=to_val, resolution=resolution, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            frm.grid_columnconfigure(1, weight=1)
            return row + 1

        var_det_pothole = tk.DoubleVar(value=self.factory_settings.get("pothole_detection_threshold", 0.5))
        row = add_slider("Detection Threshold", var_det_pothole, tab, row, 0, 1, 0.01)
        self.params["pothole_detection_threshold"] = var_det_pothole

    def create_collision_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Collision Warning")
        row = 0

        def add_slider(label_text, var, frm, row, from_val, to_val, resolution, is_int=False):
            tk.Label(frm, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            scale = tk.Scale(frm, variable=var, from_=from_val, to=to_val, resolution=resolution, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            frm.grid_columnconfigure(1, weight=1)
            return row + 1

        var_ttc = tk.IntVar(value=self.factory_settings.get("collision_ttc", 2))
        var_conf_tries = tk.IntVar(value=self.factory_settings.get("collision_confidence_tries", 2))
        var_trig_life = tk.IntVar(value=self.factory_settings.get("collision_triggered_lifetime", 3))
        row = add_slider("Time To Collision (TTC)", var_ttc, tab, row, 1, 10, 1, is_int=True)
        row = add_slider("Confidence Tries", var_conf_tries, tab, row, 1, 5, 1, is_int=True)
        row = add_slider("Triggered Lifetime", var_trig_life, tab, row, 1, 10, 1, is_int=True)
        self.params["collision_ttc"] = var_ttc
        self.params["collision_confidence_tries"] = var_conf_tries
        self.params["collision_triggered_lifetime"] = var_trig_life

    def create_speed_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Speed Detection")
        row = 0

        def add_slider(label_text, var, frm, row, from_val, to_val, resolution):
            tk.Label(frm, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            scale = tk.Scale(frm, variable=var, from_=from_val, to=to_val, resolution=resolution, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            frm.grid_columnconfigure(1, weight=1)
            return row + 1

        var_proc_noise = tk.DoubleVar(value=self.factory_settings.get("speed_kalman_process_noise", 1e-3))
        var_meas_noise = tk.DoubleVar(value=self.factory_settings.get("speed_kalman_measurement_noise", 1e-3))
        var_inner = tk.DoubleVar(value=self.factory_settings.get("speed_inner_weight", 0.75))
        var_middle = tk.DoubleVar(value=self.factory_settings.get("speed_middle_weight", 1.0))
        var_outer = tk.DoubleVar(value=self.factory_settings.get("speed_outer_weight", 1.25))
        var_left = tk.DoubleVar(value=self.factory_settings.get("speed_left_weight", 1.0))
        var_right = tk.DoubleVar(value=self.factory_settings.get("speed_right_weight", 1.0))
        row = add_slider("Kalman Process Noise", var_proc_noise, tab, row, 0.0001, 0.01, 0.0001)
        row = add_slider("Kalman Measurement Noise", var_meas_noise, tab, row, 0.0001, 0.01, 0.0001)
        row = add_slider("Inner Weight", var_inner, tab, row, 0.5, 2.0, 0.05)
        row = add_slider("Middle Weight", var_middle, tab, row, 0.5, 2.0, 0.05)
        row = add_slider("Outer Weight", var_outer, tab, row, 0.5, 2.0, 0.05)
        row = add_slider("Left Weight", var_left, tab, row, 0.5, 2.0, 0.05)
        row = add_slider("Right Weight", var_right, tab, row, 0.5, 2.0, 0.05)
        self.params["speed_kalman_process_noise"] = var_proc_noise
        self.params["speed_kalman_measurement_noise"] = var_meas_noise
        self.params["speed_inner_weight"] = var_inner
        self.params["speed_middle_weight"] = var_middle
        self.params["speed_outer_weight"] = var_outer
        self.params["speed_left_weight"] = var_left
        self.params["speed_right_weight"] = var_right

    def create_sign_disp_tab(self):
        tab = tk.Frame(self.notebook)
        self.notebook.add(tab, text="Traffic Sign Disp")
        row = 0

        def add_slider(label_text, var, frm, row, from_val, to_val, resolution, is_int=False):
            tk.Label(frm, text=label_text).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            scale = tk.Scale(frm, variable=var, from_=from_val, to=to_val, resolution=resolution, orient=tk.HORIZONTAL)
            scale.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
            frm.grid_columnconfigure(1, weight=1)
            return row + 1

        var_grid_rows = tk.IntVar(value=self.factory_settings.get("traffic_sign_grid_rows", 4))
        var_grid_cols = tk.IntVar(value=self.factory_settings.get("traffic_sign_grid_cols", 8))
        var_cell_width = tk.IntVar(value=self.factory_settings.get("traffic_sign_cell_width", 120))
        var_cell_height = tk.IntVar(value=self.factory_settings.get("traffic_sign_cell_height", 120))
        var_full_life = tk.DoubleVar(value=self.factory_settings.get("traffic_sign_full_lifetime", 2.0))
        var_fade_time = tk.DoubleVar(value=self.factory_settings.get("traffic_sign_fade_time", 1.0))
        row = add_slider("Grid Rows", var_grid_rows, tab, row, 1, 10, 1, is_int=True)
        row = add_slider("Grid Columns", var_grid_cols, tab, row, 1, 16, 1, is_int=True)
        row = add_slider("Cell Width", var_cell_width, tab, row, 50, 300, 1, is_int=True)
        row = add_slider("Cell Height", var_cell_height, tab, row, 50, 300, 1, is_int=True)
        row = add_slider("Full Lifetime", var_full_life, tab, row, 0.5, 5.0, 0.1)
        row = add_slider("Fade Time", var_fade_time, tab, row, 0.1, 5.0, 0.1)
        self.params["traffic_sign_grid_rows"] = var_grid_rows
        self.params["traffic_sign_grid_cols"] = var_grid_cols
        self.params["traffic_sign_cell_width"] = var_cell_width
        self.params["traffic_sign_cell_height"] = var_cell_height
        self.params["traffic_sign_full_lifetime"] = var_full_life
        self.params["traffic_sign_fade_time"] = var_fade_time

    def on_apply(self):
        new_settings = {key: var.get() for key, var in self.params.items()}
        self.apply_callback(new_settings)
        messagebox.showinfo("Settings", "Settings applied and system restarted.")

    def on_reset(self):
        for key, var in self.params.items():
            var.set(self.factory_settings.get(key))
        messagebox.showinfo("Settings", "Settings reset to factory defaults.")


#########################
# FullView (Duplicate for completeness)
#########################
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


#########################
# DashboardView Class
#########################
class DashboardView(UpdatableFrame):
    def __init__(self, master, ps, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        self.btn_rapid = tk.Button(btn_frame, text="Rapid View", command=self.show_rapid)
        self.btn_rapid.pack(side=tk.LEFT, padx=5)
        self.btn_full = tk.Button(btn_frame, text="Full View", command=self.show_full)
        self.btn_full.pack(side=tk.LEFT, padx=5)
        self.content_frame = tk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True)
        self.current_subview = None
        self.show_rapid()

    def show_rapid(self):
        if self.current_subview:
            self.current_subview.destroy()
        self.current_subview = RapidView(self.content_frame, self.ps)
        self.current_subview.pack(fill=tk.BOTH, expand=True)

    def show_full(self):
        if self.current_subview:
            self.current_subview.destroy()
        self.current_subview = FullView(self.content_frame, self.ps)
        self.current_subview.pack(fill=tk.BOTH, expand=True)


#########################
# MainApplicationFrame Class
#########################
class MainApplicationFrame(UpdatableFrame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master = master
        self.default_settings = {
            "general_detection_threshold": 0.3,
            "general_tracker_confidence_threshold": 0.7,
            "general_tracker_feature_threshold": 0.95,
            "general_tracker_position_threshold": 0.98,
            "general_tracker_lifespan": 3,
            "sign_detection_threshold": 0.7,
            "sign_registry_min_occurrences": 5,
            "sign_registry_lifetime": 5,
            "sign_casting_lifetime": 3,
            "sign_tracker_feature_threshold": 0.95,
            "sign_tracker_position_threshold": 0.95,
            "lane_confidence_threshold": 0.2,
            "pothole_detection_threshold": 0.5,
            "collision_ttc": 2,
            "collision_confidence_tries": 2,
            "collision_triggered_lifetime": 3,
            "speed_kalman_process_noise": 1e-3,
            "speed_kalman_measurement_noise": 1e-3,
            "speed_inner_weight": 0.75,
            "speed_middle_weight": 1.0,
            "speed_outer_weight": 1.25,
            "speed_left_weight": 1.0,
            "speed_right_weight": 1.0,
            "traffic_sign_grid_rows": 4,
            "traffic_sign_grid_cols": 8,
            "traffic_sign_cell_width": 120,
            "traffic_sign_cell_height": 120,
            "traffic_sign_full_lifetime": 2.0,
            "traffic_sign_fade_time": 1.0,
            "danger_zone_coefficients": "0.2,1.0;0.8,1.0;0.4,0.75;0.6,0.75",
            "src_weights": "0.0,1.0;1.0,1.0;0.47,0.47;0.53,0.47",
            "dst_weights": "0.0,1.0;1.0,1.0;0.0,0.0;1.0,0.0"
        }
        self.settings = self.default_settings.copy()
        self.sidebar = tk.Frame(self, width=200, bg="#DDDDDD")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.btn_dashboard = tk.Button(self.sidebar, text="Dashboard", command=self.show_dashboard)
        self.btn_dashboard.pack(fill=tk.X, padx=10, pady=10)
        self.btn_calibration = tk.Button(self.sidebar, text="Calibration", command=self.show_calibration)
        self.btn_calibration.pack(fill=tk.X, padx=10, pady=10)
        self.btn_settings = tk.Button(self.sidebar, text="Settings", command=self.show_settings)
        self.btn_settings.pack(fill=tk.X, padx=10, pady=10)
        filler = tk.Frame(self.sidebar, bg="#DDDDDD")
        filler.pack(fill=tk.BOTH, expand=True)
        self.status_label = tk.Label(self.sidebar, text="Status: Stopped", bg="#DDDDDD", fg="red")
        self.status_label.pack(padx=10, pady=5)
        self.btn_start = tk.Button(self.sidebar, text="Start Video", command=self.start_video)
        self.btn_start.pack(fill=tk.X, padx=10, pady=5)
        self.btn_stop = tk.Button(self.sidebar, text="Stop Video", command=self.stop_video)
        self.btn_stop.pack(fill=tk.X, padx=10, pady=5)
        self.main_area = tk.Frame(self, bg="#EEEEEE")
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.current_view = None
        self.initialize_system()
        self.show_dashboard()
        self.add_after(100, self.update_ui)

    def initialize_system(self):
        video_path = "assets/videos/video_2.mp4"
        object_model_path = "trained_models/moob-yolov8n.pt"
        lane_model_path = "trained_models/lane-yolov8n.pt"
        sign_model_path = "trained_models/sign-yolov8n.pt"
        hole_model_path = "trained_models/hole-yolov8n.pt"
        sncl_model_path = "trained_models/traffic-sign-class-yolov11n.pt"
        src_weights = parse_matrix(self.settings.get("src_weights"), 4, 2)
        dst_weights = parse_matrix(self.settings.get("dst_weights"), 4, 2)
        danger_zone_coefficients = parse_matrix(self.settings.get("danger_zone_coefficients"), 4, 2)
        self.image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1 / 30)
        self.object_detection_module = GeneralObjectDetectionModule(
            source_module=self.image_reading_module,
            model_weights=object_model_path,
            detection_threshold=self.settings["general_detection_threshold"],
            tracker_confidence_threshold=self.settings["general_tracker_confidence_threshold"],
            tracker_feature_threshold=self.settings["general_tracker_feature_threshold"],
            tracker_position_threshold=self.settings["general_tracker_position_threshold"],
            tracker_lifespan=self.settings["general_tracker_lifespan"]
        )
        lane_curve_estimator = LaneCurveEstimator(memory_size=25)
        lane_processor_corrector = LaneProcessorCorrector(lane_overlap=10, y_tolerance=5)
        self.lane_detection_module = LaneDetectionModule(
            source_module=self.image_reading_module,
            model_weights=lane_model_path,
            lane_curve_estimator=lane_curve_estimator,
            lane_processor_corrector=lane_processor_corrector,
            confidence_threshold=self.settings["lane_confidence_threshold"]
        )
        self.sign_detection_module = TrafficSignDetectionModule(
            source_module=self.image_reading_module,
            yolo_detect_weights=sign_model_path,
            yolo_class_weights=sncl_model_path,
            detection_threshold=self.settings["sign_detection_threshold"],
            registry_min_occurrences=self.settings["sign_registry_min_occurrences"],
            registry_lifetime=self.settings["sign_registry_lifetime"],
            casting_lifetime=self.settings["sign_casting_lifetime"],
            tracker_position_threshold=self.settings["sign_tracker_position_threshold"],
            tracker_feature_threshold=self.settings["sign_tracker_feature_threshold"]
        )
        self.perspective_transformation_module = PerspectiveTransformationModule(
            source_module=self.image_reading_module,
            src_weights=src_weights,
            dst_weights=dst_weights
        )
        self.pothole_detection_module = PotholeDetectionModule(
            source_module=self.image_reading_module,
            model_weights=hole_model_path,
            detection_threshold=self.settings["pothole_detection_threshold"]
        )
        self.collision_warning_module = CollisionWarningModule(
            object_detection_module=self.object_detection_module,
            frame_width=self.image_reading_module.frame_width,
            frame_height=self.image_reading_module.frame_height,
            danger_zone_coefficients=danger_zone_coefficients,
            ttc=self.settings["collision_ttc"],
            confidence_tries=self.settings["collision_confidence_tries"],
            triggered_lifetime=self.settings["collision_triggered_lifetime"]
        )
        self.led_strip_module = LEDStripModule(
            perspective_transformation_module=self.perspective_transformation_module,
            lane_detection_module=self.lane_detection_module,
            pothole_detection_module=self.pothole_detection_module,
            collision_warning_module=self.collision_warning_module,
            object_detection_module=self.object_detection_module,
            traffic_sign_detection_module=self.sign_detection_module,
            width=1200,
            height=200
        )
        from src.modules.speed_detection_module import SpeedDetectionModule
        self.speed_detection_module = SpeedDetectionModule(
            image_reading_module=self.image_reading_module,
            object_detection_module=self.object_detection_module,
            horiz_roi=(0.4, 0.6, 0.0, 1.0),
            radial_roi=(0.0, 1.0, 0.0, 1.0),
            kalman_process_noise=self.settings.get("speed_kalman_process_noise", 1e-3),
            kalman_measurement_noise=self.settings.get("speed_kalman_measurement_noise", 1e-3),
            inner_weight=self.settings.get("speed_inner_weight", 0.75),
            middle_weight=self.settings.get("speed_middle_weight", 1.0),
            outer_weight=self.settings.get("speed_outer_weight", 1.25),
            left_weight=self.settings.get("speed_left_weight", 1.0),
            right_weight=self.settings.get("speed_right_weight", 1.0)
        )
        from src.modules.traffic_sign_display_module import TrafficSignDisplayModule
        self.traffic_sign_display_module = TrafficSignDisplayModule(
            image_reading_module=self.image_reading_module,
            traffic_sign_detection_module=self.sign_detection_module,
            grid_rows=self.settings.get("traffic_sign_grid_rows", 4),
            grid_cols=self.settings.get("traffic_sign_grid_cols", 8),
            cell_width=self.settings.get("traffic_sign_cell_width", 120),
            cell_height=self.settings.get("traffic_sign_cell_height", 120),
            background_color=(255, 255, 255),
            full_lifetime=self.settings.get("traffic_sign_full_lifetime", 2.0),
            fade_time=self.settings.get("traffic_sign_fade_time", 1.0),
            include_positional_data=True,
            two_sided=True
        )
        from src.modules.forward_distance_module import ForwardDistanceModule
        self.forward_distance_module = ForwardDistanceModule(
            image_reading_module=self.image_reading_module,
            object_detection_module=self.object_detection_module,
            perspective_transformation_module=self.perspective_transformation_module,
            zone_fraction=0.3,
            resolution=(256, 512)
        )
        self.ps = PerceptionSystem(
            image_reading_module=self.image_reading_module,
            perspective_transformation_module=self.perspective_transformation_module,
            lane_detection_module=self.lane_detection_module,
            sign_detection_module=self.sign_detection_module,
            general_object_detection_module=self.object_detection_module,
            pothole_detection_module=self.pothole_detection_module,
            collision_warning_module=self.collision_warning_module,
            led_strip_module=self.led_strip_module,
            speed_detection_module=self.speed_detection_module,
            traffic_sign_display_module=self.traffic_sign_display_module,
            forward_distance_module=self.forward_distance_module
        )

    def show_dashboard(self):
        self.clear_main_area()
        self.current_view = DashboardView(self.main_area, self.ps)
        self.current_view.pack(fill=tk.BOTH, expand=True)

    def show_calibration(self):
        self.clear_main_area()
        self.current_view = CalibrationEditor(self.main_area, self.ps, self.calibration_applied,
                                              self.calibration_cancelled)
        self.current_view.pack(fill=tk.BOTH, expand=True)

    def show_settings(self):
        self.clear_main_area()
        self.current_view = SettingsView(self.main_area, self.ps, self.apply_settings, self.default_settings)
        self.current_view.pack(fill=tk.BOTH, expand=True)

    def clear_main_area(self):
        if self.current_view:
            self.current_view.destroy()
            self.current_view = None

    def start_video(self):
        self.ps.start()
        self.status_label.config(text="Status: Running", fg="green")
        print("Perception System Started")

    def stop_video(self):
        self.ps.stop()
        self.status_label.config(text="Status: Stopped", fg="red")
        print("Perception System Stopped")

    def apply_settings(self, new_settings):
        self.ps.stop()
        self.settings.update(new_settings)
        self.initialize_system()
        self.ps.start()
        print("Settings applied and system restarted")

    def calibration_applied(self, new_params):
        print("Calibration Applied. Parameters:", new_params)
        self.show_dashboard()

    def calibration_cancelled(self):
        print("Calibration Cancelled. Reverting to default calibration.")
        self.show_dashboard()

    def update_ui(self):
        self.add_after(100, self.update_ui)


#########################
# MainApplication Class
#########################
class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ADAS Dashboard")
        self.geometry("1024x768")
        self.app = MainApplicationFrame(self)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.app.stop_video()
        self.destroy()


if __name__ == '__main__':
    app = MainApplication()
    app.mainloop()
