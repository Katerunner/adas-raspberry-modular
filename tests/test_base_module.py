import unittest
import random
import time
from src.modules.base_module import BaseModule


class ExampleModule(BaseModule):
    def __init__(self, result_list: list):
        self.result_list = result_list
        super().__init__()

    def perform(self):
        """Simulate some processing with a loop and update the value."""
        while not self._stop_event.is_set():
            # Simulate processing by generating a random number
            self.value = random.random()
            self.result_list.append(self.value)
            time.sleep(0.5)  # Simulate time-consuming processing


class TestExampleModule(unittest.TestCase):
    def test_module_processing(self):
        check_list = []

        module = ExampleModule(result_list=check_list)

        self.assertEqual(len(check_list), 0, "Check list should initially be empty")

        module.start()
        time.sleep(1)

        self.assertGreater(len(check_list), 0, "Check list should have elements after starting the module")

        # Let the module run for a few more seconds
        time.sleep(5)

        module.stop()
        final_count = len(check_list)
        time.sleep(1)

        # Assert no values were added after the module was stopped - otherwise it kept running after stop command.
        self.assertEqual(len(check_list), final_count, "No values should be added after the module is stopped")


if __name__ == "__main__":
    unittest.main()
