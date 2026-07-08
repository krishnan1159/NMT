import logging
from collections import defaultdict

class FirstNLogger:
    def __init__(self, logger, n):
        self.logger = logger
        self.n = n
        self.counts = defaultdict(int)

    def debug(self, key, message):
        if self.counts[key] < self.n:
            self.logger.debug(message)
            self.counts[key] += 1
