import threading
import time

from src.modules.base_module import BaseModule


class BasePool:
    def __init__(self, result_format: str, workers: list[BaseModule], delay: float = 0.0):
        assert result_format in ['last', 'all'], "result_format must be 'last' or 'all'"
        self.result_format = result_format
        self.workers = workers
        self.delay = delay
        self.value = None
        self.value_list = None if result_format == 'last' else [None] * len(workers)
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        """Start all workers and begin collecting results."""
        self._stop_event.clear()

        for worker in self.workers:
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
        """Internal method to collect results from workers."""
        while not self._stop_event.is_set():
            if self.result_format == 'last':
                for worker in self.workers:
                    self.value = worker.value
                    time.sleep(self.delay)
            elif self.result_format == 'all':
                for i, worker in enumerate(self.workers):
                    self.value_list[i] = worker.value
                    time.sleep(self.delay)
