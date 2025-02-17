import numpy as np


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
        if len(centers) == 0:
            return

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
