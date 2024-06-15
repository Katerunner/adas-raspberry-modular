import random
import unittest
import time
from src.modules.base_module import BaseModule
from src.pools.base_pool import BasePool


class ExampleModuleA(BaseModule):
    def perform(self):
        while not self._stop_event.is_set():
            self.value = 'a' if random.random() > 0.5 else 'b'


class ExampleModule1(BaseModule):
    def perform(self):
        while not self._stop_event.is_set():
            self.value = '1' if random.random() > 0.5 else '2'


class ExampleModuleSym(BaseModule):
    def perform(self):
        while not self._stop_event.is_set():
            self.value = '@' if random.random() > 0.5 else '%'


class TestBasePool(unittest.TestCase):
    def test_pool_last_result_format(self):
        # Create example modules
        workers = [ExampleModuleA(), ExampleModule1(), ExampleModuleSym()]
        pool = BasePool(result_format='last', workers=workers, delay=0.0)

        # Start the pool
        pool.start()
        time.sleep(2)  # Let it run for some time

        # Collect 200 results
        results = []
        for _ in range(200):
            results.append(pool.value)
            time.sleep(0.005)

        # Stop the pool
        pool.stop()

        # Check that the results contain expected characters
        result_set = set(results)
        self.assertTrue({'a', 'b', '1', '2', '@', '%'}.issubset(result_set))

    def test_pool_all_result_format(self):
        # Create example modules
        workers = [ExampleModuleA(), ExampleModule1(), ExampleModuleSym()]
        pool = BasePool(result_format='all', workers=workers, delay=0.1)

        # Start the pool
        pool.start()
        time.sleep(2)  # Let it run for some time

        # Collect results from value list
        values_list = pool.value

        # Stop the pool
        pool.stop()

        # Check that the value list contains expected characters at respective indexes
        self.assertIn(values_list[0], ['a', 'b'])
        self.assertIn(values_list[1], ['1', '2'])
        self.assertIn(values_list[2], ['@', '%'])


if __name__ == "__main__":
    unittest.main()
