import threading


class BaseModule:
    def __init__(self):
        self.value = None
        self._thread = None
        self._stop_event = threading.Event()

    def start(self):
        """Start the module's processing in a separate thread."""
        if self._thread is None:
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run)
            self._thread.start()

    def stop(self):
        """Stop the module's processing thread."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join()
            self._thread = None

    def _run(self):
        """Internal method to run the perform method."""
        self.perform()

    def perform(self):
        """The processing logic for the module. Should be overridden in subclasses."""
        raise NotImplementedError("The perform method should be implemented by subclasses.")
