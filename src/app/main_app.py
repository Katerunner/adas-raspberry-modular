import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2

from src.app.sidebar import Sidebar
from src.app.dashboard_view import DashboardView
from src.app.settings_view import SettingsView
from src.app.calibration_view import CalibrationView
from src.app.perception_setup import setup_perception_system


class MainApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.state("zoomed")
        self.title("Tkinter Video Interface")
        self.geometry("900x600")
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self, self.show_frame, self.start_action, self.stop_action)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.main_container = ttk.Frame(self, padding=10)
        self.main_container.grid(row=0, column=1, sticky="nsew")
        self.main_container.rowconfigure(0, weight=1)
        self.main_container.columnconfigure(0, weight=1)

        self.frames = {}
        self.dashboard_view = DashboardView(self.main_container)
        self.dashboard_view.grid(row=0, column=0, sticky="nsew")
        self.frames["dashboard"] = self.dashboard_view

        self.settings_view = SettingsView(self.main_container)
        self.settings_view.grid(row=0, column=0, sticky="nsew")
        self.frames["settings"] = self.settings_view

        self.calibration_view = CalibrationView(self.main_container)
        self.calibration_view.grid(row=0, column=0, sticky="nsew")
        self.frames["calibration"] = self.calibration_view

        self.show_frame("dashboard")

        self.ps = setup_perception_system()
        self.update_video_frame()

    def show_frame(self, frame_name):
        frame = self.frames.get(frame_name)
        if frame:
            frame.tkraise()

    def update_video_frame(self):
        # Dashboard view: do not resize; use the raw frames
        frame = self.ps.frame
        if frame is not None:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            im = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=im)
            self.dashboard_view.dashboard_video_label.imgtk = imgtk
            self.dashboard_view.dashboard_video_label.configure(image=imgtk)
        perspective_frame = self.ps.perspective_frame
        if perspective_frame is not None:
            cv2image2 = cv2.cvtColor(perspective_frame, cv2.COLOR_BGR2RGB)
            im2 = Image.fromarray(cv2image2)
            imgtk2 = ImageTk.PhotoImage(image=im2)
            self.dashboard_view.vertical_video_label.imgtk = imgtk2
            self.dashboard_view.vertical_video_label.configure(image=imgtk2)

        # Calibration view: do not resize; show raw frames
        if "calibration" in self.frames:
            calib = self.frames["calibration"]
            if frame is not None:
                cv2image_calib = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_calib = Image.fromarray(cv2image_calib)
                imgtk_calib = ImageTk.PhotoImage(image=im_calib)
                calib.video_label.imgtk = imgtk_calib
                calib.video_label.configure(image=imgtk_calib)
            if perspective_frame is not None:
                cv2image_persp = cv2.cvtColor(perspective_frame, cv2.COLOR_BGR2RGB)
                im_persp = Image.fromarray(cv2image_persp)
                imgtk_persp = ImageTk.PhotoImage(image=im_persp)
                calib.perspective_label.imgtk = imgtk_persp
                calib.perspective_label.configure(image=imgtk_persp)
        self.after(30, self.update_video_frame)

    def start_action(self):
        print("Starting Perception System.")
        self.ps.start()

    def stop_action(self):
        print("Stopping Perception System.")
        self.ps.stop()
