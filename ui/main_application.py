#!/usr/bin/env python3
from tkinter import ttk
import ttkbootstrap as tk
import numpy as np

from ui.base.updatable_frame import UpdatableFrame
from ui.utils.helpers import parse_matrix
from ui.views.dashboard import DashboardView
from ui.views.calibration import CalibrationEditor
from ui.views.settings import SettingsView

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
            "src_expand_weight": 0.1,
            # Persistent order: BL, BR, TL, TR.
            "src_weights": "0.0,1.0;1.0,1.0;0.47,0.47;0.53,0.47",
            "dst_weights": "0.0,1.0;1.0,1.0;0.0,0.0;1.0,0.0"
        }
        self.settings = self.default_settings.copy()

        self._build_sidebar()
        # Insert a vertical separator between the sidebar and main area
        self.separator = ttk.Separator(self, orient="vertical")
        self.separator.pack(side=tk.LEFT, fill=tk.Y, padx=2)
        self._build_main_area()
        self.current_view = None
        self.initialize_system()
        self.show_dashboard()
        self.add_after(100, self.update_ui)

    def _build_sidebar(self):
        self.sidebar = tk.Frame(self, width=200)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.btn_dashboard = tk.Button(
            self.sidebar, text="Dashboard", command=self.show_dashboard
        )
        self.btn_dashboard.pack(fill=tk.X, padx=10, pady=10)
        self.btn_calibration = tk.Button(
            self.sidebar, text="Calibration", command=self.show_calibration
        )
        self.btn_calibration.pack(fill=tk.X, padx=10, pady=10)
        self.btn_settings = tk.Button(
            self.sidebar, text="Settings", command=self.show_settings
        )
        self.btn_settings.pack(fill=tk.X, padx=10, pady=10)
        filler = tk.Frame(self.sidebar)
        filler.pack(fill=tk.BOTH, expand=True)
        self.status_label = tk.Label(
            self.sidebar, text="Status: Stopped", foreground="red"
        )
        self.status_label.pack(padx=10, pady=5)
        self.btn_start = tk.Button(
            self.sidebar, text="Start Video", command=self.start_video
        )
        self.btn_start.pack(fill=tk.X, padx=10, pady=5)
        self.btn_stop = tk.Button(
            self.sidebar, text="Stop Video", command=self.stop_video
        )
        self.btn_stop.pack(fill=tk.X, padx=10, pady=5)

    def _build_main_area(self):
        self.main_area = tk.Frame(self)
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def initialize_system(self):
        video_path = "assets/videos/video_1.mp4"
        object_model_path = "trained_models/moob-yolov8n.pt"
        lane_model_path = "trained_models/lane-yolov8n.pt"
        sign_model_path = "trained_models/sign-yolov8n.pt"
        hole_model_path = "trained_models/hole-yolov8n.pt"
        sncl_model_path = "trained_models/traffic-sign-class-yolov11n.pt"
        src_weights = parse_matrix(self.settings.get("src_weights"), 4, 2)
        dst_weights = parse_matrix(self.settings.get("dst_weights"), 4, 2)
        danger_zone_coefficients = parse_matrix(
            self.settings.get("danger_zone_coefficients"), 4, 2
        )

        src_expand_weight = self.settings["src_expand_weight"]
        src_width = np.abs(src_weights[2][0] - src_weights[3][0])
        src_weights[0][0] -= src_expand_weight
        src_weights[1][0] += src_expand_weight
        src_weights[2][0] -= src_expand_weight * src_width
        src_weights[3][0] += src_expand_weight * src_width

        self.image_reading_module = ImageReadingModule(
            source=video_path, delay_seconds=1 / 30
        )
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

        self.collision_warning_module = CollisionWarningModule(
            object_detection_module=self.object_detection_module,
            speed_detection_module=self.speed_detection_module,
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
        self.current_view = CalibrationEditor(
            self.main_area, self.ps, self.settings,
            self.calibration_applied, self.calibration_cancelled
        )
        self.current_view.pack(fill=tk.BOTH, expand=True)

    def calibration_applied(self, new_params):
        print("Calibration Applied. Parameters:", new_params)
        self.ps.stop()
        self.initialize_system()
        self.ps.start()
        self.show_dashboard()

    def calibration_cancelled(self):
        print("Calibration Cancelled. Reverting to default calibration.")
        self.settings["src_weights"] = self.default_settings["src_weights"]
        self.ps.stop()
        self.initialize_system()
        self.ps.start()
        self.show_dashboard()

    def show_settings(self):
        self.clear_main_area()
        self.current_view = SettingsView(
            self.main_area,
            self.ps,
            self.apply_settings,
            self.settings,
            self.default_settings
        )
        self.current_view.pack(fill=tk.BOTH, expand=True)

    def clear_main_area(self):
        if self.current_view:
            self.current_view.destroy()
            self.current_view = None

    def start_video(self):
        self.ps.start()
        self.status_label.config(text="Status: Running", foreground="green")
        print("Perception System Started")

    def stop_video(self):
        self.ps.stop()
        self.status_label.config(text="Status: Stopped", foreground="red")
        print("Perception System Stopped")

    def apply_settings(self, new_settings):
        self.ps.stop()
        self.settings.update(new_settings)
        self.initialize_system()
        self.ps.start()
        print("Settings applied and system restarted")

    def update_ui(self):
        self.add_after(100, self.update_ui)


class MainApplication(UpdatableFrame):
    # noinspection PyUnresolvedReferences
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.master.title("ADAS Dashboard")
        self.master.geometry("1024x768")
        self.app = MainApplicationFrame(self)
        self.app.pack(fill=tk.BOTH, expand=True)
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.app.stop_video()
        self.master.destroy()
