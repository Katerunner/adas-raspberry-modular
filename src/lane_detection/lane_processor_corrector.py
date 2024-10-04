import numpy as np
from concurrent.futures import ThreadPoolExecutor


class LaneProcessorCorrector:
    def __init__(self, lane_overlap=10, y_tolerance=5):
        """
        Initialize the LaneProcessorCorrector.

        :param lane_overlap: Minimum allowed horizontal distance between lanes, defaults to 10.
        :param y_tolerance: Maximum vertical distance to group similar y-values, defaults to 5.
        """
        self.lane_overlap = lane_overlap
        self.y_tolerance = y_tolerance

    def correct(self, lane_processor):
        """
        Correct the lanes in the LaneProcessor to ensure they don't overlap.

        :param lane_processor: The LaneProcessor object containing lanes to be corrected.
        :return: The corrected LaneProcessor object.
        """
        lanes = lane_processor.lanes
        lane_points = {i: lanes[i].estimated_points if lanes[i] else None for i in range(4)}

        # Get all y-values from all lanes and group them by proximity using the tolerance
        all_y = sorted(set(y for pts in lane_points.values() if pts is not None for _, y in pts))

        # Group y-values that are within the tolerance
        common_y = self._group_common_y(all_y)

        if not common_y:
            return lane_processor  # No correction can be done if there are no common y-values

        # Filter out points that do not share similar y-values across all lanes
        lane_points = self._filter_points_by_common_y(lane_points, common_y)

        # Correct lane points based on overlap rules
        lane_points = self._parallel_correct_lanes(lane_points, common_y)

        # Update the lane processor with the corrected points
        for lane_id, pts in lane_points.items():
            if lanes[lane_id]:
                lanes[lane_id].estimated_points = np.array(pts)

        return lane_processor

    def _group_common_y(self, all_y):
        """Group y-values that are within the tolerance."""
        common_y = []
        temp_group = []
        for y in all_y:
            if not temp_group or abs(y - temp_group[-1]) <= self.y_tolerance:
                temp_group.append(y)
            else:
                common_y.append(int(np.mean(temp_group)))  # Take the average of the group
                temp_group = [y]

        if temp_group:
            common_y.append(int(np.mean(temp_group)))  # Append the final group

        return common_y

    def _filter_points_by_common_y(self, lane_points, common_y):
        """Filter out points that do not share similar y-values across all lanes, using concurrency."""

        def filter_points(pts):
            """Helper function to filter points for a single lane, parallelizing the point filtering."""

            def is_valid_point(p):
                """Check if the point's y-value is within the tolerance of any common_y value."""
                return any(abs(p[1] - y) <= self.y_tolerance for y in common_y)

            with ThreadPoolExecutor() as point_executor:
                point_futures = [point_executor.submit(is_valid_point, p) for p in pts]
                filtered_points = [p for p, future in zip(pts, point_futures) if future.result()]

            return filtered_points

        with ThreadPoolExecutor() as executor:
            futures = {
                lane_id: executor.submit(filter_points, pts)
                for lane_id, pts in lane_points.items() if pts is not None
            }

            for lane_id, future in futures.items():
                lane_points[lane_id] = future.result()

        return lane_points

    def _get_lane_x_coords(self, lane_points, y):
        """Retrieve the x coordinates for each lane at a given y-value, using concurrency."""

        def get_x_coordinate(lane_id):
            """Helper function to get the x-coordinate for a specific lane."""
            if lane_points[lane_id] is not None:
                return next((p[0] for p in lane_points[lane_id] if abs(p[1] - y) <= self.y_tolerance), None)
            return None

        with ThreadPoolExecutor() as executor:
            futures = {lane_id: executor.submit(get_x_coordinate, lane_id) for lane_id in range(4)}

            ll_x = futures[0].result()
            lc_x = futures[1].result()
            rc_x = futures[2].result()
            rr_x = futures[3].result()

        return ll_x, lc_x, rc_x, rr_x

    def _correct_ll_lc_overlap(self, ll_points, ll_x, lc_x):
        """Correct the overlap between LL and LC lanes."""
        if ll_x is not None and lc_x is not None and ll_x > lc_x - self.lane_overlap:
            corrected_points = [p for p in ll_points if p[0] <= lc_x - self.lane_overlap]
            return corrected_points
        return ll_points

    def _correct_lc_rc_overlap(self, rc_points, lc_x, rc_x):
        """Correct the overlap between LC and RC lanes (priority on keeping LC)."""
        if lc_x is not None and rc_x is not None and lc_x > rc_x - self.lane_overlap:
            corrected_points = [p for p in rc_points if p[0] >= lc_x + self.lane_overlap]
            return corrected_points
        return rc_points

    def _correct_rc_rr_overlap(self, rr_points, rc_x, rr_x):
        """Correct the overlap between RC and RR lanes (correct RR points)."""
        if rc_x is not None and rr_x is not None and rc_x > rr_x - self.lane_overlap:
            corrected_points = [p for p in rr_points if p[0] >= rc_x + self.lane_overlap]
            return corrected_points
        return rr_points

    def _parallel_correct_lanes(self, lane_points, common_y):
        """Apply the lane corrections in parallel for each y in common_y."""

        def apply_corrections(y):
            """Helper function to apply corrections for a specific y-value."""
            ll_x, lc_x, rc_x, rr_x = self._get_lane_x_coords(lane_points, y)

            with ThreadPoolExecutor() as executor:
                # Create parallel tasks for lane corrections
                corrections = {
                    0: executor.submit(self._correct_ll_lc_overlap, lane_points[0], ll_x, lc_x),
                    2: executor.submit(self._correct_lc_rc_overlap, lane_points[2], lc_x, rc_x),
                    3: executor.submit(self._correct_rc_rr_overlap, lane_points[3], rc_x, rr_x)
                }

                # Gather the corrected points from all tasks
                lane_points[0] = corrections[0].result()
                lane_points[2] = corrections[2].result()
                lane_points[3] = corrections[3].result()

        # Run the corrections in parallel for each y in common_y
        with ThreadPoolExecutor() as executor:
            executor.map(apply_corrections, common_y)

        return lane_points
