import threading
import time
from src.modules.base_module import BaseModule
from typing import List


class BasePool(BaseModule):
    def __init__(self, result_format: str, workers: List[BaseModule], intermodule_start_interval: float = 0.0,
                 intermodule_read_interval: float = 0.0):
        """Initialize the BasePool with a result format, workers, and separate intervals for start and read."""
        super().__init__()
        assert result_format in ['last', 'all'], "result_format must be 'last' or 'all'"
        self.result_format = result_format
        self.workers = workers
        self.intermodule_start_interval = intermodule_start_interval
        self.intermodule_read_interval = intermodule_read_interval
        self.value = None if result_format == 'last' else [None] * len(workers)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start all workers with intermodule start interval and begin collecting results."""
        self._stop_event.clear()

        for worker in self.workers:
            time.sleep(self.intermodule_start_interval)
            worker.start()

        self._thread = threading.Thread(target=self._collect_results)
        self._thread.start()

    def stop(self):
        """Stop all workers and the result collection."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()

        for worker in self.workers:
            worker.stop()

    def _collect_results(self):
        """Internal method to collect results from workers with intermodule read interval."""
        while not self._stop_event.is_set():
            if self.result_format == 'last':
                for worker in self.workers:
                    self.value = worker.value
                    time.sleep(self.intermodule_read_interval)
            elif self.result_format == 'all':
                for i, worker in enumerate(self.workers):
                    self.value[i] = worker.value
                    time.sleep(self.intermodule_read_interval)
