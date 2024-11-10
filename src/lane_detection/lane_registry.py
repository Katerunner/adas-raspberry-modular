from src.lane_detection.lane import Lane
from src.lane_detection.lane_processor import LaneProcessor


class LaneRegistry:
    lane_labels = {0: "LL", 1: "LC", 2: "RC", 3: "RR"}

    def __init__(self, lanes: list[Lane] = None):

        if lanes is None:
            lanes = [None, None, None, None]
        elif len(lanes) != 4:
            raise ValueError("LaneRegistry must be initialized with exactly 4 lanes.")

        self.lanes = lanes

    def update_from_lane_processor(self, lane_processor: LaneProcessor):
        if lane_processor:
            self.lanes = lane_processor.lanes
