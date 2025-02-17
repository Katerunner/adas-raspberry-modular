import tkinter as tk
import cv2
import numpy as np
import base64


def cv2_to_photoimage(cv2_img):
    """
    Convert a cv2 BGR image into a Tkinter PhotoImage.
    """
    cv2_rgb = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    ret, buf = cv2.imencode('.png', cv2_rgb)
    b64 = base64.b64encode(buf).decode('utf-8')
    return tk.PhotoImage(data=b64)


class DraggablePoint:
    def __init__(self, canvas, x, y, visual_radius, update_callback, canvas_width, canvas_height):
        self.canvas = canvas
        self.x = x
        self.y = y
        self.visual_radius = visual_radius
        # Increase hit area for easier dragging:
        self.hit_radius = visual_radius + 15
        self.update_callback = update_callback
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height

        # Create an invisible oval (hit area) that’s larger than the visible dot.
        self.hit_area = canvas.create_oval(
            x - self.hit_radius, y - self.hit_radius,
            x + self.hit_radius, y + self.hit_radius,
            fill="", outline=""
        )
        # Create the visible dot.
        self.circle = canvas.create_oval(
            x - self.visual_radius, y - self.visual_radius,
            x + self.visual_radius, y + self.visual_radius,
            fill='red', outline='black', width=2
        )
        # Create on-canvas text that shows normalized coordinates.
        self.text = canvas.create_text(
            x + self.visual_radius + 5, y - self.visual_radius - 5,
            text=self.get_text(), anchor='nw',
            fill='white', font=('Arial', 10, 'bold')
        )
        # Bind mouse events to all three items.
        for tag in (self.hit_area, self.circle, self.text):
            canvas.tag_bind(tag, "<ButtonPress-1>", self.on_press)
            canvas.tag_bind(tag, "<B1-Motion>", self.on_motion)
            canvas.tag_bind(tag, "<ButtonRelease-1>", self.on_release)
        # (We no longer need an offset, since we want the dot to center on the pointer.)
        self._drag_data = {"x": 0, "y": 0}

    def get_text(self):
        # Return normalized coordinates as text.
        norm_x = self.x / self.canvas_width
        norm_y = self.y / self.canvas_height
        return f"({norm_x:.2f}, {norm_y:.2f})"

    def on_press(self, event):
        # Instead of preserving an offset, we center the dot under the pointer.
        # (This ensures the invisible hit area is effectively centered on the dot.)
        self._drag_data["x"] = 0
        self._drag_data["y"] = 0
        self.on_motion(event)  # Immediately update position

    def on_motion(self, event):
        # Directly update the dot’s center to the mouse pointer.
        new_x = event.x
        new_y = event.y
        # Clamp the new position within the canvas.
        new_x = max(0, min(new_x, self.canvas_width))
        new_y = max(0, min(new_y, self.canvas_height))
        self.x = new_x
        self.y = new_y

        # Update positions for hit area, visible dot, and text.
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
        # Notify the main app to update the polygon, coordinate labels, and perspective image.
        self.update_callback()

    def on_release(self, event):
        # (Optional) Add code here for when the mouse button is released.
        pass


class PolygonEditor:
    def __init__(self, root):
        # Dimensions for the main and perspective images.
        self.main_width = 512
        self.main_height = 512
        self.persp_width = 128
        self.persp_height = 512

        # Create a main frame with two columns.
        self.main_frame = tk.Frame(root)
        self.main_frame.grid(row=0, column=0, padx=5, pady=5)

        # Left canvas: displays the main 512x512 image.
        self.canvas_main = tk.Canvas(self.main_frame, width=self.main_width, height=self.main_height)
        self.canvas_main.grid(row=0, column=0)

        # Right canvas: displays the live perspective-transformed image (128x512).
        self.canvas_persp = tk.Canvas(self.main_frame, width=self.persp_width, height=self.persp_height)
        self.canvas_persp.grid(row=0, column=1, padx=5)

        # Generate a random main image using cv2.
        self.main_cv2_img = self.create_random_cv2_image(self.main_width, self.main_height)
        self.photo_main = cv2_to_photoimage(self.main_cv2_img)
        self.canvas_main.create_image(0, 0, anchor='nw', image=self.photo_main)

        # Create a placeholder perspective image (will be updated live).
        self.photo_persp = cv2_to_photoimage(self.main_cv2_img)  # temporary placeholder
        self.persp_img_item = self.canvas_persp.create_image(0, 0, anchor='nw', image=self.photo_persp)

        # Define default draggable points using normalized coordinates.
        # Order: bottom-left, top-left, top-right, bottom-right.
        default_points = [
            (0.0, 1.0),  # bottom-left
            (0.4, 0.5),  # top-left (will be mapped to (0,0) in perspective image)
            (0.6, 0.5),  # top-right (maps to (128,0))
            (1.0, 1.0)  # bottom-right
        ]
        self.points = []
        self.visual_radius = 5
        for norm in default_points:
            x = norm[0] * self.main_width
            y = norm[1] * self.main_height
            dp = DraggablePoint(self.canvas_main, x, y, self.visual_radius,
                                self.update_all, self.main_width, self.main_height)
            self.points.append(dp)

        # Draw the initial polygon connecting the draggable points.
        coords = []
        for dp in self.points:
            coords.extend([dp.x, dp.y])
        self.polygon = self.canvas_main.create_polygon(coords, outline='yellow', fill='', width=2)

        # Create a frame for coordinate labels that spans both columns.
        self.coord_frame = tk.Frame(root)
        self.coord_frame.grid(row=1, column=0, columnspan=2, pady=5)
        self.coord_labels = []
        for i in range(len(self.points)):
            label = tk.Label(self.coord_frame, text=self.get_norm_text(self.points[i]), font=('Arial', 10))
            label.pack(side='left', padx=10)
            self.coord_labels.append(label)

        # Update the perspective transform immediately.
        self.update_all()

    def create_random_cv2_image(self, width, height):
        """
        Generate a random color image (numpy array) using cv2.
        """
        return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)

    def get_norm_text(self, dp):
        norm_x = dp.x / self.main_width
        norm_y = dp.y / self.main_height
        idx = self.points.index(dp) + 1
        return f"P{idx}: ({norm_x:.2f}, {norm_y:.2f})"

    def update_all(self):
        # Update polygon on the main canvas.
        coords = []
        for dp in self.points:
            coords.extend([dp.x, dp.y])
        self.canvas_main.coords(self.polygon, *coords)
        # Update the external coordinate labels.
        for i, dp in enumerate(self.points):
            self.coord_labels[i].configure(text=self.get_norm_text(dp))
        # Compute the perspective transformation.
        # Source points: current positions of the draggable points.
        src_pts = np.array([[dp.x, dp.y] for dp in self.points], dtype=np.float32)
        # Destination points: map to the four corners of the perspective image.
        dst_pts = np.array([
            [0, self.persp_height],  # bottom-left
            [0, 0],  # top-left
            [self.persp_width, 0],  # top-right
            [self.persp_width, self.persp_height]  # bottom-right
        ], dtype=np.float32)
        # Compute the transformation matrix.
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        # Warp the main image using the computed matrix.
        persp_img = cv2.warpPerspective(self.main_cv2_img, M, (self.persp_width, self.persp_height))
        # Update the perspective canvas with the new image.
        self.photo_persp = cv2_to_photoimage(persp_img)
        self.canvas_persp.itemconfig(self.persp_img_item, image=self.photo_persp)


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Live Perspective Transformation Editor")
    app = PolygonEditor(root)
    root.mainloop()
