from datetime import timedelta
import time


class Timer(object):
    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, *args):
        self._end = time.time()
        self.interval = self._end - self._start

    def __str__(self):
        return f'{self.interval:.2f}s'

    def to_string(self):
        return str(timedelta(seconds=round(self.interval)))

    def to_int(self):
        return self.interval

    def stop_timer(self):
        self._end = time.time()
        self.interval = self._end - self._start
        return self.to_string()
