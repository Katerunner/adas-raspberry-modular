import time

from src.traffic_sign_display.traffic_sign_display_processor import TrafficSignDisplayProcessor
from src.modules.base_module import BaseModule
from src.modules.image_reading_module import ImageReadingModule
from src.modules.traffic_sign_detection_module import TrafficSignDetectionModule


class TrafficSignDisplayModule(BaseModule):
    def __init__(
            self,
            image_reading_module: ImageReadingModule,
            traffic_sign_detection_module: TrafficSignDetectionModule,
            grid_rows=4,
            grid_cols=8,
            cell_width=120,
            cell_height=120,
            background_color=(255, 255, 255),
            full_lifetime=2.0,
            fade_time=1.0,
            include_positional_data=True,
            two_sided: bool = True
    ):
        super().__init__()

        self.traffic_sign_detection_module = traffic_sign_detection_module
        self.image_reading_module = image_reading_module
        self.two_sided = two_sided
        self.traffic_sign_display_processor = TrafficSignDisplayProcessor(
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            cell_width=cell_width,
            cell_height=cell_height,
            background_color=background_color,
            full_lifetime=full_lifetime,
            fade_time=fade_time,
            include_positional_data=include_positional_data
        )

    def perform(self):
        time.sleep(2)
        while not self._stop_event.is_set():
            if self.traffic_sign_detection_module.value is not None:
                registry = self.traffic_sign_detection_module.value.registry
                for sign in registry:
                    position = [
                        (sign.position[0] + sign.position[2]) / 2 / self.image_reading_module.frame_width,
                        (sign.position[1] + sign.position[3]) / 2 / self.image_reading_module.frame_height,
                    ]

                    sign_info = {
                        "id": sign.guid,
                        "name": sign.name.replace("--", ": ").replace("-", " ").title(),
                        "cropped": sign.image,
                        "position": position
                    }

                    self.traffic_sign_display_processor.update([sign_info])

                self.value = self.traffic_sign_display_processor.get_image(two_sided=self.two_sided)
