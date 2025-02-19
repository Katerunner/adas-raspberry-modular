# ui/views/calibration.py
import ttkbootstrap as tk
import cv2
import numpy as np
from ui.base.updatable_frame import UpdatableFrame
from ui.utils.helpers import cv2_to_tk, parse_matrix


class DraggablePoint:
    def __init__(self, canvas, x, y, visual_radius, update_callback, canvas_width, canvas_height, role=None):
        """
        role: a string indicating the point's role.
            "TL" => top left, "TR" => top right, "BL" => bottom left, "BR" => bottom right.
        """
        self.canvas = canvas
        self.x = x
        self.y = y
        self.visual_radius = visual_radius
        self.hit_radius = visual_radius + 15
        self.update_callback = update_callback
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.role = role  # e.g. "TL", "TR", "BL", or "BR"
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
        # Use our helper method to get the proper label offset and anchor.
        dx, dy, anchor = self.get_label_offset_and_anchor()
        self.text = canvas.create_text(
            x + dx, y + dy,
            text=self.get_text(), anchor=anchor,
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

    def get_label_offset_and_anchor(self):
        """
        Returns a tuple (dx, dy, anchor) based on the point's role.
        The goal is:
          - Top left ("TL"): label to the left of the point.
          - Top right ("TR"): label to the right.
          - Bottom left ("BL"): label above the point (to the right).
          - Bottom right ("BR"): label above the point (to the left).
        """
        offset = self.visual_radius + 5
        if self.role == "TL":
            return -offset, -offset, "se"
        elif self.role == "TR":
            return offset, -offset, "sw"
        elif self.role == "BL":
            return offset, -offset, "nw"
        elif self.role == "BR":
            return -offset, -offset, "ne"
        else:
            return offset, -offset, "nw"

    def update_display(self):
        """Update the visual representation of the point on the canvas."""
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
        dx, dy, anchor = self.get_label_offset_and_anchor()
        self.canvas.coords(self.text, self.x + dx, self.y + dy)
        self.canvas.itemconfigure(self.text, text=self.get_text(), anchor=anchor)

    def on_press(self, event):
        self.on_motion(event)

    def on_motion(self, event):
        new_x = max(0, min(event.x, self.canvas_width))
        new_y = max(0, min(event.y, self.canvas_height))
        self.x = new_x
        self.y = new_y
        self.update_display()

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

        self.main_image_tk = None
        self.perspective_image_tk = None

        self.main_height, self.main_width = 512, 512
        self.persp_width = 128
        self.persp_height = 512

        # Create canvases for main and perspective view.
        self.canvas_frame = tk.Frame(self)
        self.canvas_frame.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.canvas_main = tk.Canvas(self.canvas_frame, width=self.main_width, height=self.main_height,
                                     background="black")
        self.canvas_main.grid(row=0, column=0, sticky="nsew")
        self.canvas_persp = tk.Canvas(self.canvas_frame, width=self.persp_width, height=self.persp_height,
                                      background="black")
        self.canvas_persp.grid(row=0, column=1, padx=5, sticky="nsew")
        self.canvas_frame.grid_columnconfigure(0, weight=1)
        self.canvas_frame.grid_columnconfigure(1, weight=0)
        self.canvas_frame.grid_rowconfigure(0, weight=1)

        # Create the main image.
        self.canvas_main_image = self.canvas_main.create_image(0, 0, anchor='nw')
        self.canvas_prsp_image = self.canvas_persp.create_image(0, 0, anchor='nw')

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

        self.src_expand_weight = self.settings.get('src_expand_weight', 0.1)

        # Save a copy for cancellation.
        self.default_points = persistent_points.copy() if self.default_points is None else self.default_points

        # Create draggable points in persistent order.
        # Order in settings: [BL, BR, TL, TR]
        roles = ["BL", "BR", "TL", "TR"]
        self.points = []
        self.visual_radius = 5
        for idx, norm in enumerate(persistent_points):
            x = norm[0] * self.main_width
            y = norm[1] * self.main_height
            dp = DraggablePoint(
                canvas=self.canvas_main,
                x=x, y=y,
                visual_radius=self.visual_radius,
                update_callback=self.update_all,
                canvas_width=self.main_width,
                canvas_height=self.main_height,
                role=roles[idx]
            )
            self.points.append(dp)

        # Create polygon overlays.
        polygon_coords = self.draggable_points_to_coords(expand_x_coefficient=0.0)
        self.polygon = self.canvas_main.create_polygon(polygon_coords, outline='blue', fill='', width=2)
        outer_polygon_coords = self.draggable_points_to_coords(expand_x_coefficient=self.src_expand_weight)
        self.outer_polygon = self.canvas_main.create_polygon(outer_polygon_coords, outline='dodger blue', fill='',
                                                             width=1)

        # Set the stacking order:
        #   1. Lower the main image to the bottom.
        #   2. Raise the polygon overlays above the image.
        #   3. Raise the draggable point items above the polygons.
        self.canvas_main.tag_lower(self.canvas_main_image)
        self.canvas_main.tag_raise(self.polygon, self.canvas_main_image)
        self.canvas_main.tag_raise(self.outer_polygon, self.canvas_main_image)
        for dp in self.points:
            self.canvas_main.tag_raise(dp.hit_area)
            self.canvas_main.tag_raise(dp.circle)
            self.canvas_main.tag_raise(dp.text)

        # Add Apply and Cancel buttons.
        self.button_frame = tk.Frame(self)
        self.button_frame.grid(row=1, column=0, columnspan=2, pady=10)
        self.apply_btn = tk.Button(self.button_frame, text="Apply Calibration", command=self.on_apply)
        self.apply_btn.pack(side='left', padx=10)
        self.cancel_btn = tk.Button(self.button_frame, text="Cancel", command=self.on_cancel)
        self.cancel_btn.pack(side='left', padx=10)

        self.add_after(500, self.update_all)

    def enforce_top_points_distance(self):
        """
        Enforce that the Euclidean distance (in normalized coordinates)
        between the top two points (indices 2 and 3) remains exactly 0.06.
        """
        top_left = self.points[2]
        top_right = self.points[3]
        mw, mh = self.main_width, self.main_height

        # Get normalized positions
        tl_norm = (top_left.x / mw, top_left.y / mh)
        tr_norm = (top_right.x / mw, top_right.y / mh)

        dx_norm = tr_norm[0] - tl_norm[0]
        dy_norm = tr_norm[1] - tl_norm[1]
        d_norm = (dx_norm ** 2 + dy_norm ** 2) ** 0.5
        if d_norm < 1e-6:
            return

        desired = 0.06  # desired Euclidean distance in normalized units
        factor = desired / d_norm

        new_dx_norm = dx_norm * factor
        new_dy_norm = dy_norm * factor

        # Compute the midpoint in normalized coordinates.
        mid_norm = ((tl_norm[0] + tr_norm[0]) / 2, (tl_norm[1] + tr_norm[1]) / 2)
        new_tl_norm = (mid_norm[0] - new_dx_norm / 2, mid_norm[1] - new_dy_norm / 2)
        new_tr_norm = (mid_norm[0] + new_dx_norm / 2, mid_norm[1] + new_dy_norm / 2)

        # Convert back to pixel coordinates.
        new_tl_x = new_tl_norm[0] * mw
        new_tl_y = new_tl_norm[1] * mh
        new_tr_x = new_tr_norm[0] * mw
        new_tr_y = new_tr_norm[1] * mh

        # Update the point positions.
        top_left.x = new_tl_x
        top_left.y = new_tl_y
        top_right.x = new_tr_x
        top_right.y = new_tr_y

        # Update their visual representation.
        top_left.update_display()
        top_right.update_display()

    def get_normalized_coordinates(self):
        # Return coordinates in persistent order (BL, BR, TL, TR).
        return [(dp.x / self.main_width, dp.y / self.main_height) for dp in self.points]

    def on_apply(self):
        coords = self.get_normalized_coordinates()
        new_src_weights = ";".join([f"{x:.2f},{y:.2f}" for x, y in coords])
        print("Applied Calibration Coordinates (persistent order):", coords)
        self.settings["src_weights"] = new_src_weights
        self.apply_callback({"src_weights": new_src_weights})

    def on_cancel(self):
        if "src_weights" in self.settings and self.settings["src_weights"]:
            matrix = parse_matrix(self.settings["src_weights"], 4, 2)
            persistent = [(float(r[0]), float(r[1])) for r in matrix]
        else:
            persistent = self.default_points

        for i, (nx, ny) in enumerate(persistent):
            self.points[i].x = nx * self.main_width
            self.points[i].y = ny * self.main_height
            self.points[i].update_display()

        self.update_all()
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

        input_image = self.ps.image_reading_module.value
        main_image_np = cv2.resize(input_image, (self.main_width, self.main_height))
        main_image_tk = cv2_to_tk(main_image_np)
        self.canvas_main.itemconfig(self.canvas_main_image, image=main_image_tk)

        self.enforce_top_points_distance()

        polygon_coords = self.draggable_points_to_coords(expand_x_coefficient=0.0)
        self.canvas_main.coords(self.polygon, *polygon_coords)
        outer_polygon_coords = self.draggable_points_to_coords(expand_x_coefficient=self.src_expand_weight)
        self.canvas_main.coords(self.outer_polygon, *outer_polygon_coords)

        src_pts = np.array([[p.x, p.y] for p in self.get_ordered_points()], dtype=np.float32)
        dst_pts = np.array([
            [0, self.persp_height],
            [0, 0],
            [self.persp_width, 0],
            [self.persp_width, self.persp_height]
        ], dtype=np.float32)

        perspective_image_np = self.apply_perspective_transformation(
            image=main_image_np,
            src_pts=src_pts,
            dst_pts=dst_pts
        )
        perspective_image_tk = cv2_to_tk(perspective_image_np)
        self.canvas_persp.itemconfig(self.canvas_prsp_image, image=perspective_image_tk)

        self.main_image_tk = main_image_tk
        self.perspective_image_tk = perspective_image_tk
        self.add_after(500, self.update_all)

    def get_ordered_points(self):
        # Returns points in order: BL, TL, TR, BR.
        return [self.points[0], self.points[2], self.points[3], self.points[1]]

    def draggable_points_to_coords(self, expand_x_coefficient=0.0):
        ordered = self.get_ordered_points()
        coords = []
        for p in ordered:
            coords.extend([p.x, p.y])
        top_width = np.abs(coords[2] - coords[4])
        coords[0] -= self.main_width * expand_x_coefficient
        coords[2] -= top_width * expand_x_coefficient
        coords[4] += top_width * expand_x_coefficient
        coords[6] += self.main_width * expand_x_coefficient
        return coords

    def apply_perspective_transformation(self, image: np.ndarray, src_pts: np.ndarray, dst_pts: np.ndarray):
        transformation_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return cv2.warpPerspective(image, transformation_matrix, (self.persp_width, self.persp_height))
