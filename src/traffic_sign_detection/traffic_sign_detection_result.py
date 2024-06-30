from src.traffic_sign_detection.detected_traffic_sign import DetectedTrafficSign
from src.traffic_sign_detection.traffic_sign_base import TrafficSignBase


class TrafficSignDetectionResult:
    def __init__(self, yolo_result, traffic_sign_base: TrafficSignBase):
        self.boxes = yolo_result.boxes
        self.names = yolo_result.names
        self.image = yolo_result.orig_img
        self.shape = yolo_result.orig_shape
        self.traffic_sign_base = traffic_sign_base
        self.detected_signs = self._get_detected_signs_from_result()

    def _get_detected_signs_from_result(self):
        detected_sings = []
        for i in range(len(self.boxes.cls)):
            class_id = int(self.boxes.cls[i].cpu().numpy())
            confidence = float(self.boxes.conf[i].cpu().numpy())
            xyxy = self.boxes.xyxy[i].cpu().numpy()
            xyxyn = self.boxes.xyxyn[i].cpu().numpy()
            sign_image = self.image[int(xyxy[0]):int(xyxy[2]), int(xyxy[1]):int(xyxy[3])]
            traffic_sign = self.traffic_sign_base.get_sign_by_code(self.names[class_id])

            detected_sings.append(
                DetectedTrafficSign(
                    class_id=class_id,
                    confidence=confidence,
                    xyxyn=xyxyn,
                    image=sign_image,
                    traffic_sign=traffic_sign
                )
            )
        return detected_sings
