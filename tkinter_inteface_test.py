import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk

from src.lane_detection.lane_curve_estimator import LaneCurveEstimator
from src.lane_detection.lane_processor_corrector import LaneProcessorCorrector
from src.modules.collision_warning_module import CollisionWarningModule
from src.modules.general_object_detection_module import GeneralObjectDetectionModule
from src.modules.image_reading_module import ImageReadingModule
from src.modules.lane_detection_module import LaneDetectionModule
from src.modules.perspective_transformation_module import PerspectiveTransformationModule
from src.modules.pothole_detection_module import PotholeDetectionModule
from src.modules.traffic_sign_detection_module import TrafficSignDetectionModule
from src.system.perception_system import PerceptionSystem

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Instead of full screen, start maximized (expanded window)
        self.state("zoomed")
        self.title("Tkinter Video Interface")
        # The geometry here is a fallback; the window is maximized by default.
        self.geometry("900x600")

        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.create_sidebar()
        self.create_main_container()
        self.create_dashboard_view()
        self.create_settings_view()
        self.create_calibration_view()

        self.show_frame("dashboard")
        self.setup_perception_system()
        self.update_video_frame()

    def create_sidebar(self):
        sidebar = ttk.Frame(self, width=150, relief=tk.RIDGE, padding=10)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.grid_propagate(False)

        title_lbl = ttk.Label(sidebar, text="Menu", font=("Arial", 14))
        title_lbl.pack(pady=(0, 20))

        dash_btn = ttk.Button(sidebar, text="Dashboard", command=lambda: self.show_frame("dashboard"))
        dash_btn.pack(pady=5, fill="x")

        settings_btn = ttk.Button(sidebar, text="Settings", command=lambda: self.show_frame("settings"))
        settings_btn.pack(pady=5, fill="x")

        calib_btn = ttk.Button(sidebar, text="Calibration", command=lambda: self.show_frame("calibration"))
        calib_btn.pack(pady=5, fill="x")

    def create_main_container(self):
        self.main_container = ttk.Frame(self, padding=10)
        self.main_container.grid(row=0, column=1, sticky="nsew")
        self.main_container.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(0, weight=1)
        self.frames = {}

    def create_dashboard_view(self):
        dashboard = ttk.Frame(self.main_container)
        dashboard.grid(row=0, column=0, sticky="nsew")
        # Configure rows: first row for side-by-side videos, second row for LED Bar, third row for buttons.
        dashboard.columnconfigure(0, weight=1)
        dashboard.rowconfigure(0, weight=1)
        dashboard.rowconfigure(1, weight=0)
        dashboard.rowconfigure(2, weight=0)
        self.frames["dashboard"] = dashboard

        # Top frame: two videos side by side
        video_frame = ttk.Frame(dashboard)
        video_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        video_frame.columnconfigure(0, weight=3)
        video_frame.columnconfigure(1, weight=1)

        # Main video placeholder (16:9) exactly 480x270
        video16_frame = ttk.Frame(video_frame, relief=tk.SUNKEN, width=480, height=270)
        video16_frame.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        video16_frame.grid_propagate(False)
        self.dashboard_video_label = tk.Label(video16_frame)
        self.dashboard_video_label.pack(expand=True, fill="both")

        # Vertical video placeholder exactly 120x270 from ps.perspective_frame
        video_vert_frame = ttk.Frame(video_frame, relief=tk.SUNKEN, width=120, height=270)
        video_vert_frame.grid(row=0, column=1, sticky="nsew")
        video_vert_frame.grid_propagate(False)
        self.vertical_video_label = tk.Label(video_vert_frame)
        self.vertical_video_label.pack(expand=True, fill="both")

        # LED Bar horizontal video placeholder exactly 600x100 below the two videos.
        led_bar_frame = ttk.Frame(dashboard, relief=tk.SUNKEN, width=600, height=100)
        led_bar_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        led_bar_frame.grid_propagate(False)
        # For now, just a placeholder text; you can later update it as a video frame if needed.
        self.led_bar_label = tk.Label(led_bar_frame, text="LED Bar Placeholder", background="black", foreground="white")
        self.led_bar_label.pack(expand=True, fill="both")

        # Button frame
        btn_frame = ttk.Frame(dashboard)
        btn_frame.grid(row=2, column=0, sticky="ew")
        btn_frame.columnconfigure((0, 1), weight=1)

        start_btn = ttk.Button(btn_frame, text="Start", command=self.start_action)
        start_btn.grid(row=0, column=0, padx=5, pady=5, sticky="ew")

        stop_btn = ttk.Button(btn_frame, text="Stop", command=self.stop_action)
        stop_btn.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def create_settings_view(self):
        settings = ttk.Frame(self.main_container)
        settings.grid(row=0, column=0, sticky="nsew")
        settings.columnconfigure(0, weight=1)
        settings.rowconfigure(0, weight=1)
        self.frames["settings"] = settings

        header_lbl = ttk.Label(settings, text="Settings", font=("Arial", 16))
        header_lbl.pack(pady=(0, 20))

        brightness_lbl = ttk.Label(settings, text="Brightness")
        brightness_lbl.pack(pady=(10, 0))
        brightness_slider = ttk.Scale(settings, from_=0, to=100, orient="horizontal")
        brightness_slider.pack(padx=10, pady=5, fill="x")

        text_lbl = ttk.Label(settings, text="Enter text:")
        text_lbl.pack(pady=(10, 0))
        text_entry = ttk.Entry(settings)
        text_entry.pack(padx=10, pady=5, fill="x")

        contrast_lbl = ttk.Label(settings, text="Contrast")
        contrast_lbl.pack(pady=(10, 0))
        contrast_slider = ttk.Scale(settings, from_=0, to=100, orient="horizontal")
        contrast_slider.pack(padx=10, pady=5, fill="x")

    def create_calibration_view(self):
        calibration = ttk.Frame(self.main_container)
        calibration.grid(row=0, column=0, sticky="nsew")
        calibration.columnconfigure(0, weight=1)
        calibration.rowconfigure(1, weight=1)
        self.frames["calibration"] = calibration

        header_lbl = ttk.Label(calibration, text="Calibration", font=("Arial", 16))
        header_lbl.pack(pady=(0, 20))

        video16_frame = ttk.Frame(calibration, relief=tk.SUNKEN, width=480, height=270)
        video16_frame.pack(pady=10)
        video16_frame.grid_propagate(False)
        lbl16 = ttk.Label(video16_frame, text="16:9 Video Placeholder", background="black", foreground="white")
        lbl16.place(relx=0.5, rely=0.5, anchor="center")

        slider_frame = ttk.Frame(calibration)
        slider_frame.pack(pady=10, fill="x")

        slider1_lbl = ttk.Label(slider_frame, text="Slider 1")
        slider1_lbl.pack(pady=(10, 0))
        slider1 = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal")
        slider1.pack(padx=10, pady=5, fill="x")

        slider2_lbl = ttk.Label(slider_frame, text="Slider 2")
        slider2_lbl.pack(pady=(10, 0))
        slider2 = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal")
        slider2.pack(padx=10, pady=5, fill="x")

        slider3_lbl = ttk.Label(slider_frame, text="Slider 3")
        slider3_lbl.pack(pady=(10, 0))
        slider3 = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal")
        slider3.pack(padx=10, pady=5, fill="x")

        slider4_lbl = ttk.Label(slider_frame, text="Slider 4")
        slider4_lbl.pack(pady=(10, 0))
        slider4 = ttk.Scale(slider_frame, from_=0, to=100, orient="horizontal")
        slider4.pack(padx=10, pady=5, fill="x")

    def show_frame(self, frame_name):
        frame = self.frames.get(frame_name)
        if frame:
            frame.tkraise()
        else:
            print(f"Frame '{frame_name}' does not exist.")

    def setup_perception_system(self):
        video_path = "assets/videos/video_2.mp4"
        object_model_path = "trained_models/moob-yolov8n.pt"
        lane_model_path = "trained_models/lane-yolov8n.pt"
        sign_model_path = "trained_models/sign-yolov8n.pt"
        hole_model_path = "trained_models/hole-yolov8n.pt"

        image_reading_module = ImageReadingModule(source=video_path, delay_seconds=1/30)
        object_detection_module = GeneralObjectDetectionModule(
            source_module=image_reading_module,
            model_weights=object_model_path,
        )
        lane_curve_estimator = LaneCurveEstimator(memory_size=25)
        lane_processor_corrector = LaneProcessorCorrector(lane_overlap=10, y_tolerance=5)
        lane_detection_module = LaneDetectionModule(
            source_module=image_reading_module,
            model_weights=lane_model_path,
            lane_curve_estimator=lane_curve_estimator,
            lane_processor_corrector=lane_processor_corrector
        )
        sign_detection_module = TrafficSignDetectionModule(
            source_module=image_reading_module,
            model_weights=sign_model_path
        )
        perspective_transformation_module = PerspectiveTransformationModule(source_module=image_reading_module)
        pothole_detection_module = PotholeDetectionModule(
            source_module=image_reading_module,
            model_weights=hole_model_path
        )
        collision_warning_module = CollisionWarningModule(
            object_detection_module=object_detection_module,
            frame_width=image_reading_module.frame_width,
            frame_height=image_reading_module.frame_height
        )
        self.ps = PerceptionSystem(
            image_reading_module=image_reading_module,
            perspective_transformation_module=perspective_transformation_module,
            lane_detection_module=lane_detection_module,
            sign_detection_module=sign_detection_module,
            general_object_detection_module=object_detection_module,
            pothole_detection_module=pothole_detection_module,
            collision_warning_module=collision_warning_module
        )

    def update_video_frame(self):
        # Update main video frame from ps.frame, resized exactly to 480x270.
        frame = self.ps.frame
        if frame is not None:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(cv2image).resize((480, 270))
            imgtk = ImageTk.PhotoImage(image=im)
            self.dashboard_video_label.imgtk = imgtk
            self.dashboard_video_label.configure(image=imgtk)
        # Update vertical video frame from ps.perspective_frame, resized exactly to 120x270.
        perspective_frame = self.ps.perspective_frame
        if perspective_frame is not None:
            cv2image2 = cv2.cvtColor(perspective_frame, cv2.COLOR_BGR2RGB)
            im2 = Image.fromarray(cv2image2).resize((120, 270))
            imgtk2 = ImageTk.PhotoImage(image=im2)
            self.vertical_video_label.imgtk = imgtk2
            self.vertical_video_label.configure(image=imgtk2)
        self.after(30, self.update_video_frame)

    def start_action(self):
        print("Starting Perception System.")
        self.ps.start()

    def stop_action(self):
        print("Stopping Perception System.")
        self.ps.stop()

if __name__ == "__main__":
    app = App()
    app.mainloop()
