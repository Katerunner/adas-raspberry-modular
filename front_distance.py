import cv2
import numpy as np
import random

class ForwardDistanceCalculator:
    def __init__(self, zone_fraction=0.1):
        """
        Initializes the ForwardDistanceCalculator.

        Parameters:
            zone_fraction (float): The fraction of the image width that defines the central vertical zone.
                                   For example, 0.1 means the zone is 10% of the image width, centered horizontally.
        """
        self.zone_fraction = zone_fraction

    def calculate_distance(self, image, centers):
        """
        Calculates the forward distance from the bottom-center of the image to the detected object
        (provided as a bottom-center coordinate) that is inside the central vertical zone and has the largest y coordinate.

        Parameters:
            image (np.ndarray): The input image (BGR).
            centers (list): A list of tuples (bc_x, bc_y) where each tuple represents the bottom-center
                            of a bounding box in pixel coordinates.

        Returns:
            distance (float): The Euclidean distance (in pixels) from the bottom-center of the image to the
                              selected object's center. If no object is found within the zone, returns None.
        """
        h, w = image.shape[:2]
        # Define the central narrow vertical zone.
        zone_half_width = (self.zone_fraction / 2) * w
        x_min = w / 2 - zone_half_width
        x_max = w / 2 + zone_half_width

        selected_center = None
        max_y = -1  # Will store the largest y coordinate among centers in the zone.

        # Loop over centers.
        for center in centers:
            bc_x, bc_y = center
            if x_min <= bc_x <= x_max:
                # Choose the center with the largest y (i.e. closest to the bottom)
                if bc_y > max_y:
                    max_y = bc_y
                    selected_center = (bc_x, bc_y)

        if selected_center is None:
            return None

        # Define the bottom-center of the image.
        bottom_center = (w / 2, h)
        # Compute the Euclidean distance.
        dx = bottom_center[0] - selected_center[0]
        dy = bottom_center[1] - selected_center[1]
        distance = np.sqrt(dx * dx + dy * dy)
        return distance


# ---------------- Simulation Code ----------------
if __name__ == '__main__':
    # Image dimensions
    width, height = 640, 480
    # Create an instance of the ForwardDistanceCalculator with a 10% central zone.
    calculator = ForwardDistanceCalculator(zone_fraction=0.1)

    while True:
        # Create a white background image.
        frame = np.full((height, width, 3), 255, dtype=np.uint8)

        # Simulate a list of detected objects by generating random bounding boxes.
        detected_objects = []
        centers = []  # This list will hold the (bc_x, bc_y) for each bounding box.
        num_objects = random.randint(3, 7)
        for i in range(num_objects):
            # Randomly generate a bounding box.
            # Ensure the box is at least 30x30 pixels.
            x1 = random.randint(0, width - 100)
            y1 = random.randint(0, height - 100)
            box_width = random.randint(30, 100)
            box_height = random.randint(30, 100)
            x2 = min(x1 + box_width, width - 1)
            y2 = min(y1 + box_height, height - 1)
            detected_objects.append({"xyxy": [x1, y1, x2, y2]})
            # Compute bottom-center for the bounding box.
            bc_x = (x1 + x2) // 2
            bc_y = y2
            centers.append((bc_x, bc_y))
            # Draw the bounding box for visualization.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw the center of the box.
            cv2.circle(frame, (bc_x, bc_y), 3, (0, 0, 255), -1)

        # Draw the central narrow vertical zone.
        zone_half_width = int((calculator.zone_fraction / 2) * width)
        x_zone_min = width // 2 - zone_half_width
        x_zone_max = width // 2 + zone_half_width
        cv2.line(frame, (x_zone_min, 0), (x_zone_min, height), (255, 0, 0), 2)
        cv2.line(frame, (x_zone_max, 0), (x_zone_max, height), (255, 0, 0), 2)

        # Calculate the forward distance using the list of (bc_x, bc_y) tuples.
        distance = calculator.calculate_distance(frame, centers)
        if distance is not None:
            text = f"Distance: {distance:.1f}px"
        else:
            text = "Distance: N/A"
        cv2.putText(frame, text, (10, height - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 2)

        # Display the result.
        cv2.imshow("Forward Distance", frame)
        key = cv2.waitKey(500)  # update every 500ms
        if key == ord('q'):
            break

    cv2.destroyAllWindows()
