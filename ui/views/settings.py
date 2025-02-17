# ui/views/settings.py
import tkinter as tk
from tkinter import ttk, messagebox
from ui.base.updatable_frame import UpdatableFrame


class SettingsView(UpdatableFrame):
    def __init__(self, master, ps, apply_callback, current_settings, default_settings, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.ps = ps
        self.apply_callback = apply_callback
        self.current_setting = current_settings.copy()
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

        var_det = tk.DoubleVar(value=self.current_setting.get("general_detection_threshold", 0.3))
        var_conf = tk.DoubleVar(value=self.current_setting.get("general_tracker_confidence_threshold", 0.7))
        var_feat = tk.DoubleVar(value=self.current_setting.get("general_tracker_feature_threshold", 0.95))
        var_pos = tk.DoubleVar(value=self.current_setting.get("general_tracker_position_threshold", 0.98))
        var_life = tk.IntVar(value=self.current_setting.get("general_tracker_lifespan", 3))
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

        var_det_sign = tk.DoubleVar(value=self.current_setting.get("sign_detection_threshold", 0.7))
        var_reg_min = tk.IntVar(value=self.current_setting.get("sign_registry_min_occurrences", 5))
        var_reg_life = tk.IntVar(value=self.current_setting.get("sign_registry_lifetime", 5))
        var_cast_life = tk.IntVar(value=self.current_setting.get("sign_casting_lifetime", 3))
        var_feat_sign = tk.DoubleVar(value=self.current_setting.get("sign_tracker_feature_threshold", 0.95))
        var_pos_sign = tk.DoubleVar(value=self.current_setting.get("sign_tracker_position_threshold", 0.95))
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

        var_conf_lane = tk.DoubleVar(value=self.current_setting.get("lane_confidence_threshold", 0.2))
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

        var_det_pothole = tk.DoubleVar(value=self.current_setting.get("pothole_detection_threshold", 0.5))
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

        var_ttc = tk.IntVar(value=self.current_setting.get("collision_ttc", 2))
        var_conf_tries = tk.IntVar(value=self.current_setting.get("collision_confidence_tries", 2))
        var_trig_life = tk.IntVar(value=self.current_setting.get("collision_triggered_lifetime", 3))
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

        var_proc_noise = tk.DoubleVar(value=self.current_setting.get("speed_kalman_process_noise", 1e-3))
        var_meas_noise = tk.DoubleVar(value=self.current_setting.get("speed_kalman_measurement_noise", 1e-3))
        var_inner = tk.DoubleVar(value=self.current_setting.get("speed_inner_weight", 0.75))
        var_middle = tk.DoubleVar(value=self.current_setting.get("speed_middle_weight", 1.0))
        var_outer = tk.DoubleVar(value=self.current_setting.get("speed_outer_weight", 1.25))
        var_left = tk.DoubleVar(value=self.current_setting.get("speed_left_weight", 1.0))
        var_right = tk.DoubleVar(value=self.current_setting.get("speed_right_weight", 1.0))
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

        var_grid_rows = tk.IntVar(value=self.current_setting.get("traffic_sign_grid_rows", 4))
        var_grid_cols = tk.IntVar(value=self.current_setting.get("traffic_sign_grid_cols", 8))
        var_cell_width = tk.IntVar(value=self.current_setting.get("traffic_sign_cell_width", 120))
        var_cell_height = tk.IntVar(value=self.current_setting.get("traffic_sign_cell_height", 120))
        var_full_life = tk.DoubleVar(value=self.current_setting.get("traffic_sign_full_lifetime", 2.0))
        var_fade_time = tk.DoubleVar(value=self.current_setting.get("traffic_sign_fade_time", 1.0))
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
