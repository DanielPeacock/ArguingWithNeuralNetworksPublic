import signal


class timeout:
    def __init__(self, minutes: int):
        self.minutes = minutes

    def handle(self, signum, frame):
        msg = f"Execution timed out after {self.minutes} minutes."
        raise TimeoutError(msg)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle)
        signal.alarm(int(self.minutes * 60))

    def __exit__(self, exc_type, exc_value, traceback):
        signal.alarm(0)
