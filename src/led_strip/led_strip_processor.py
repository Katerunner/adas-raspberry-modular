import time
import numpy as np
import cv2


class LEDStripProcessor:
    def __init__(self, width=640, height=80):
        self.width = width
        self.height = height
        self.events = []

    def add_event(self, relative_x, lifetime=2.0, width=0.3, intensity=1.0, color=(0, 0, 255)):
        # Default gradient event.
        x = int(relative_x * self.width)
        y = self.height // 2
        event = {
            'type': 'default',
            'x': x,
            'y': y,
            'start_time': time.time(),
            'lifetime': lifetime,
            'width': width,
            'intensity': intensity,
            'color': color
        }
        self.events.append(event)

    def add_traffic_light(self, lights):
        """
        Add traffic light events.
        lights: list of tuples (relative_x, color, lifetime)
          - relative_x: horizontal position (0.0-1.0) along the LED strip.
          - color: active color as a string: 'r', 'y', or 'g'.
          - lifetime: duration in seconds.
        """
        for rel_x, color_str, lifetime in lights:
            x = int(rel_x * self.width)
            y = self.height // 2
            event = {
                'type': 'traffic_light',
                'x': x,
                'y': y,
                'start_time': time.time(),
                'lifetime': lifetime,
                'main_color': color_str
            }
            self.events.append(event)

    def add_pedestrian_warning(self, lifetime, main_opacity=0.5, secondary_opacity=0.3):
        """
        Add a pedestrian warning event.
        The event will display a pedestrian crosswalk as a group of five wide white bars,
        evenly spaced across the LED strip.
        The opacity provided (main_opacity) will be applied uniformly.
        (The secondary_opacity parameter is ignored in this design.)
        """
        event = {
            'type': 'pedestrian_warning',
            'start_time': time.time(),
            'lifetime': lifetime,
            'main_opacity': main_opacity
        }
        self.events.append(event)

    def _draw_gradient_strip(self, image, center, half_width, color, intensity=1.0):
        # Default gradient event drawing remains unchanged.
        x0, y0 = center
        x1 = max(x0 - half_width, 0)
        x2 = min(x0 + half_width, self.width - 1)
        y1 = 0
        y2 = self.height - 1
        roi_width = x2 - x1 + 1
        roi_height = y2 - y1 + 1
        xs = np.linspace(x1 - x0, x2 - x0, roi_width)
        mask_1d = np.clip(1 - np.abs(xs) / half_width, 0, 1)
        mask = np.tile(mask_1d, (roi_height, 1))
        mask *= intensity
        roi = image[y1:y2 + 1, x1:x2 + 1].astype(np.float32)
        mask_3 = mask[..., np.newaxis]
        color_arr = np.array(color, dtype=np.float32)
        blended_roi = roi * (1 - mask_3) + color_arr * mask_3
        image[y1:y2 + 1, x1:x2 + 1] = blended_roi.astype(np.uint8)

    def _draw_traffic_light(self, image, center, fade, main_color):
        """
        Draws a traffic light event as three full-height vertical bars.
        (This method remains from the previous version.)
        """
        # Parameters for the traffic light group:
        bar_width = 20
        gap = 4
        group_width = 3 * bar_width + 2 * gap
        group_height = self.height
        glow_width = 4
        cx, _ = center
        group_x = cx - group_width // 2
        group_y = 0

        base_colors = {'r': (0, 0, 255), 'y': (0, 255, 255), 'g': (0, 255, 0)}
        group_region = np.zeros((group_height, group_width, 3), dtype=np.uint8)
        positions = []
        for i, led in enumerate(['r', 'y', 'g']):
            col_x = i * (bar_width + gap)
            positions.append((col_x, col_x + bar_width))
            intensity = fade if led == main_color else 0.3 * fade
            color = np.array(base_colors[led], dtype=np.float32) * intensity
            group_region[:, col_x:col_x + bar_width] = np.tile(color, (group_height, bar_width, 1)).astype(np.uint8)

        active_idx = {'r': 0, 'y': 1, 'g': 2}.get(main_color, 1)
        active_start, active_end = positions[active_idx]
        for side in ['left', 'right']:
            if side == 'left':
                for i in range(glow_width):
                    if active_start - i - 1 < 0: continue
                    alpha = (glow_width - i) / glow_width
                    original = group_region[:, active_start - i - 1].astype(np.float32)
                    overlay = np.array(base_colors[main_color], dtype=np.float32) * fade * alpha
                    group_region[:, active_start - i - 1] = np.clip(original + overlay, 0, 255).astype(np.uint8)
            else:
                for i in range(glow_width):
                    if active_end + i >= group_width: continue
                    alpha = (glow_width - i) / glow_width
                    original = group_region[:, active_end + i].astype(np.float32)
                    overlay = np.array(base_colors[main_color], dtype=np.float32) * fade * alpha
                    group_region[:, active_end + i] = np.clip(original + overlay, 0, 255).astype(np.uint8)

        shadow_width = 6
        for i in range(shadow_width):
            x_coord = group_x - shadow_width + i
            if x_coord < 0: continue
            alpha = (shadow_width - i) / shadow_width * 0.9
            image[:, x_coord] = np.clip(image[:, x_coord].astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)
        for i in range(shadow_width):
            x_coord = group_x + group_width + i
            if x_coord >= self.width: continue
            alpha = (i + 1) / shadow_width * 0.9
            image[:, x_coord] = np.clip(image[:, x_coord].astype(np.float32) * (1 - alpha), 0, 255).astype(np.uint8)

        x0 = max(group_x, 0)
        x1 = min(group_x + group_width, self.width)
        image[0:group_height, x0:x1] = group_region[:, (x0 - group_x):(x1 - group_x)]
        cv2.line(image, (group_x, 0), (group_x, group_height - 1), (128, 128, 128), 2)
        cv2.line(image, (group_x + group_width - 1, 0), (group_x + group_width - 1, group_height - 1), (128, 128, 128),
                 2)

    def _draw_pedestrian_warning(self, image, main_opacity, secondary_opacity):
        """
        Draws a pedestrian crosswalk on the LED strip as a group of five wide white bars,
        evenly spaced across the full width.
        All bars are drawn with the provided opacity (main_opacity).
        Each bar spans the full height.
        A thin gray border is drawn around each bar.
        """
        h, w = self.height, self.width
        num_bars = 5
        bar_width = 80  # wide white bars
        # Compute center positions for the bars; equally spaced across the full width.
        positions = [int((i + 1) * w / (num_bars + 1)) for i in range(num_bars)]
        for pos in positions:
            x0 = pos - bar_width // 2
            x1 = pos + bar_width // 2
            # Create the bar: white with the specified opacity.
            bar = np.full((h, bar_width, 3), 255, dtype=np.uint8)
            bar = (bar.astype(np.float32) * main_opacity).astype(np.uint8)
            # Place the bar into the image (crop if necessary).
            x0_clamped = max(x0, 0)
            x1_clamped = min(x1, w)
            image[:, x0_clamped:x1_clamped] = bar[:, (x0_clamped - x0):(x1_clamped - x0)]
            cv2.rectangle(image, (x0_clamped, 0), (x1_clamped - 1, h - 1), (128, 128, 128), 2)

    def update(self):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        current_time = time.time()
        remaining_events = []
        for event in self.events:
            elapsed = current_time - event['start_time']
            fade = 1.0 - elapsed / event['lifetime']
            if fade > 0:
                if event.get('type', 'default') == 'traffic_light':
                    self._draw_traffic_light(image, (event['x'], event['y']), fade, event['main_color'])
                elif event.get('type') == 'pedestrian_warning':
                    # Use the provided main_opacity for all bars.
                    self._draw_pedestrian_warning(image, event.get('main_opacity', 0.5) * fade,
                                                  event.get('secondary_opacity', 0.3) * fade)
                else:
                    half_width = int((event['width'] * self.height) / 2)
                    self._draw_gradient_strip(image, (event['x'], event['y']),
                                              half_width, event['color'],
                                              intensity=fade * event['intensity'])
                remaining_events.append(event)
        self.events = remaining_events
        return image
