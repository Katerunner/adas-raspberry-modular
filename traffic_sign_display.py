import cv2
import numpy as np
import time
import random


class TrafficSignDisplay:
    def __init__(self,
                 grid_rows=4,
                 grid_cols=8,
                 cell_width=120,
                 cell_height=120,
                 background_color=(255, 255, 255),
                 full_lifetime=2.0,
                 fade_time=1.0,
                 include_positional_data=True):
        """
        Parameters:
          grid_rows, grid_cols: Number of grid cells vertically and horizontally.
          cell_width, cell_height: Dimensions of each grid cell (in pixels).
          background_color: BGR tuple for the background.
          full_lifetime: Time (in seconds) that a sign is fully visible.
          fade_time: Duration (in seconds) of the fade-out.
          include_positional_data: If True, new signs (with a provided relative position)
                                   are placed in the corresponding grid cell.
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.cell_width = cell_width
        self.cell_height = cell_height
        self.background_color = background_color
        self.full_lifetime = full_lifetime
        self.fade_time = fade_time
        self.total_lifetime = full_lifetime + fade_time
        self.include_positional_data = include_positional_data

        # Internal state: mapping from grid cell (row, col) to sign info.
        # Each sign info is a dict with keys: id, name, image, last_update.
        self.signs = {}
        # Also, a mapping from sign id to grid cell for quick updates.
        self.id_to_cell = {}

    def update(self, new_signs: list):
        """
        Update the display with a list of new sign information.
        Each new sign is a dict with at least the keys:
            "id": a unique identifier,
            "name": the sign name,
            "cropped": the cropped image (numpy array) of the sign,
        Optionally, a "position": (x, y) relative coordinate (both in [0,1]) may be provided.
        """
        now = time.time()
        for sign in new_signs:
            sign_id = sign.get("id")
            name = sign.get("name", "Unknown")
            sign_img = sign.get("cropped")
            position = sign.get("position")  # relative (x,y)
            if sign_id in self.id_to_cell:
                # Update the existing sign.
                cell = self.id_to_cell[sign_id]
                self.signs[cell]["image"] = sign_img
                self.signs[cell]["name"] = name
                self.signs[cell]["last_update"] = now
            else:
                desired_cell = None
                if self.include_positional_data and position is not None:
                    desired_col = min(self.grid_cols - 1, max(0, int(position[0] * self.grid_cols)))
                    desired_row = min(self.grid_rows - 1, max(0, int(position[1] * self.grid_rows)))
                    desired_cell = (desired_row, desired_col)
                cell = self._find_free_cell(desired_cell)
                if cell is None:
                    cell = self._find_cell_lowest_lifetime(now)
                self.signs[cell] = {"id": sign_id, "name": name, "image": sign_img, "last_update": now}
                self.id_to_cell[sign_id] = cell
        self._remove_expired_signs(now)

    def _find_free_cell(self, desired_cell):
        free_cells = [(r, c) for r in range(self.grid_rows)
                      for c in range(self.grid_cols) if (r, c) not in self.signs]
        if not free_cells:
            return None
        if desired_cell and desired_cell in free_cells:
            return desired_cell
        elif desired_cell:
            best = None
            dmin = None
            for cell in free_cells:
                dist = ((cell[0] - desired_cell[0]) ** 2 + (cell[1] - desired_cell[1]) ** 2) ** 0.5
                if dmin is None or dist < dmin:
                    dmin = dist
                    best = cell
            return best
        else:
            return free_cells[0]

    def _find_cell_lowest_lifetime(self, now):
        min_remaining = None
        best = None
        for cell, info in self.signs.items():
            elapsed = now - info["last_update"]
            remaining = self.total_lifetime - elapsed
            if min_remaining is None or remaining < min_remaining:
                min_remaining = remaining
                best = cell
        return best

    def _remove_expired_signs(self, now):
        expired = []
        for cell, info in self.signs.items():
            if now - info["last_update"] > self.total_lifetime:
                expired.append(cell)
        for cell in expired:
            sign_id = self.signs[cell]["id"]
            del self.signs[cell]
            if sign_id in self.id_to_cell:
                del self.id_to_cell[sign_id]

    # --- Private method for gradient border (shadow) ---
    def _draw_gradient_border(self, sign_img, border_thickness=3):
        """
        Create a new image that is the sign image with a 3px black gradient border.
        The border is fully opaque black at the sign edge and fades linearly to fully transparent at the outer edge.
        """
        h, w = sign_img.shape[:2]
        new_h = h + 2 * border_thickness
        new_w = w + 2 * border_thickness
        # Create new image filled with background color.
        bordered = np.full((new_h, new_w, 3), self.background_color, dtype=np.uint8)
        # Place the sign image in the center.
        bordered[border_thickness:border_thickness + h, border_thickness:border_thickness + w] = sign_img

        # Create a mask where the sign area is 0 and border area is 255.
        mask = np.full((new_h, new_w), 255, dtype=np.uint8)
        mask[border_thickness:border_thickness + h, border_thickness:border_thickness + w] = 0

        # Compute the distance transform on the mask.
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        # Clip the distances to the border thickness.
        dist = np.clip(dist, 0, border_thickness)
        # Compute alpha: at distance 0 (inner border) alpha=1 (fully black) and at border_thickness, alpha=0.
        alpha = 1.0 - (dist / border_thickness)
        # Only apply alpha in the border region (where mask is nonzero).
        alpha[mask == 0] = 0
        # Expand alpha to three channels.
        alpha = np.repeat(alpha[:, :, np.newaxis], 3, axis=2)

        # Blend each pixel in the border with black.
        # The formula: new_pixel = (1 - alpha) * original_pixel + alpha * (0,0,0)
        # Since black is (0,0,0), this simplifies to new_pixel = (1 - alpha) * original_pixel.
        bordered = ((1 - alpha) * bordered).astype(np.uint8)
        # Restore the original sign area.
        bordered[border_thickness:border_thickness + h, border_thickness:border_thickness + w] = sign_img
        return bordered

    def get_image(self, two_sided=False):
        """
        Returns an image (numpy array) representing the current state of the display.
        Signs are drawn (with gradient shadow borders and with text) in their grid cells.
        Also draws the 8x4 grid.

        If two_sided is True and grid_cols is even, returns a tuple (left_img, right_img)
        corresponding to a left 4x4 grid and a right 4x4 grid.
        """
        width = self.grid_cols * self.cell_width
        height = self.grid_rows * self.cell_height
        full_img = np.full((height, width, 3), self.background_color, dtype=np.uint8)

        # Draw grid lines (using light gray lines)
        for r in range(1, self.grid_rows):
            y = r * self.cell_height
            cv2.line(full_img, (0, y), (width, y), (200, 200, 200), 1)
        for c in range(1, self.grid_cols):
            x = c * self.cell_width
            cv2.line(full_img, (x, 0), (x, height), (200, 200, 200), 1)

        now = time.time()

        # Helper function to overlay an image onto the background.
        def overlay_image(background, overlay_img, pos, opacity):
            x, y = pos
            h_o, w_o = overlay_img.shape[:2]
            roi = background[y:y + h_o, x:x + w_o]
            blended = cv2.addWeighted(roi, 1 - opacity, overlay_img, opacity, 0)
            background[y:y + h_o, x:x + w_o] = blended
            return background

        # Draw each sign in its grid cell.
        for (r, c), info in self.signs.items():
            elapsed = now - info["last_update"]
            if elapsed < self.full_lifetime:
                opacity = 1.0
            elif elapsed < self.total_lifetime:
                opacity = 1.0 - ((elapsed - self.full_lifetime) / self.fade_time)
            else:
                opacity = 0.0
            if opacity <= 0:
                continue

            cell_x = c * self.cell_width
            cell_y = r * self.cell_height
            margin = 5
            # Reserve some vertical space (e.g., 30 px) for the sign name.
            target_width = self.cell_width - 2 * margin
            target_height = self.cell_height - 30 - 2 * margin

            sign_img = info["image"]
            # Resize the sign image to fit in the cell.
            sign_img_resized = cv2.resize(sign_img, (target_width, target_height))

            # Create a sign image with a gradient border.
            border_thickness = 3
            sign_with_border = self._draw_gradient_border(sign_img_resized, border_thickness)
            # Adjust position to account for the added border.
            pos = (cell_x + margin - border_thickness, cell_y + margin - border_thickness)

            full_img = overlay_image(full_img, sign_with_border, pos, opacity)

            # Draw the sign's name under the image (centered).
            text = info["name"]
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = cell_x + (self.cell_width - text_size[0]) // 2
            text_y = cell_y + margin + target_height + text_size[1] + 5
            cv2.putText(full_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if two_sided and self.grid_cols % 2 == 0:
            half = (self.grid_cols // 2) * self.cell_width
            left_img = full_img[:, :half].copy()
            right_img = full_img[:, half:].copy()
            return left_img, right_img
        else:
            return full_img


# ---------------- Simulation Code ----------------
if __name__ == '__main__':
    # For simulation, use a fixed random seed for reproducibility.
    random.seed(42)
    display = TrafficSignDisplay()

    cv2.namedWindow("Traffic Signs Full")
    cv2.namedWindow("Traffic Signs Left")
    cv2.namedWindow("Traffic Signs Right")

    last_sign_time = time.time()
    sign_interval = 1.0  # generate a new sign every 1 second

    while True:
        now = time.time()
        # Randomly generate a new sign every sign_interval seconds.
        if now - last_sign_time >= sign_interval:
            # Generate a random sign image (a 50x50 image with random colors)
            sign_img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
            # Pick a random sign id between 1 and 10 (so sometimes the same sign is updated)
            sign_id = random.randint(1, 10)
            sign_name = f"Sign {sign_id}"
            # Generate random relative position (x, y in [0,1])
            pos = (random.random(), random.random())
            sign_info = {
                "id": sign_id,
                "name": sign_name,
                "cropped": sign_img,
                "position": pos
            }
            display.update([sign_info])
            last_sign_time = now

        # Retrieve the current display images.
        full_img = display.get_image(two_sided=False)
        left_img, right_img = display.get_image(two_sided=True)

        cv2.imshow("Traffic Signs Full", full_img)
        cv2.imshow("Traffic Signs Left", left_img)
        cv2.imshow("Traffic Signs Right", right_img)
        key = cv2.waitKey(30)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
