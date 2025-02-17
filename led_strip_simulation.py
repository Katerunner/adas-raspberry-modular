import cv2
import numpy as np
import time
import random


class LEDStripProcessor:
    def __init__(self, width=640, height=80):
        self.width = width
        self.height = height
        self.events = []

    def add_event(self, relative_x, lifetime=2.0, width=0.3, intensity=1.0, color=(0, 0, 255)):
        x = int(relative_x * self.width)
        y = self.height // 2
        event = {
            'x': x,
            'y': y,
            'start_time': time.time(),
            'lifetime': lifetime,
            'width': width,
            'intensity': intensity,
            'color': color
        }
        self.events.append(event)

    def _draw_gradient_strip(self, image, center, half_width, color, intensity=1.0):
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

    def update(self):
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        current_time = time.time()
        remaining_events = []
        for event in self.events:
            elapsed = current_time - event['start_time']
            fade = 1.0 - elapsed / event['lifetime']
            if fade > 0:
                eff_intensity = fade * event['intensity']
                half_width = int((event['width'] * self.height) / 2)
                self._draw_gradient_strip(image, (event['x'], event['y']), half_width, event['color'],
                                          intensity=eff_intensity)
                remaining_events.append(event)
        self.events = remaining_events
        return image


if __name__ == '__main__':
    processor = LEDStripProcessor(640, 80)
    cv2.namedWindow("LED Strip", cv2.WINDOW_NORMAL)
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
    while True:
        if random.random() < 0.05:
            processor.add_event(
                random.random(),
                lifetime=random.uniform(1.0, 3.0),
                width=random.uniform(0.1, 0.3),
                intensity=random.uniform(0.5, 1.0),
                color=random.choice(colors)
            )
        image = processor.update()
        cv2.imshow("LED Strip", image)
        if cv2.waitKey(30) == ord('q'):
            break
    cv2.destroyAllWindows()
