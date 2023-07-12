import logging

import psutil


class Profiler:
    def __init__(self, device):
        # Pass in the pid of the main thread to ensure all threads are counted for
        self.logger = logging.getLogger(__name__)

    def execute(self) -> None:
        # Give global information
        self.logger.info(
            "Machine: CPU {:.1f}% CPU Stats: {} Memory: {}".format(
                psutil.cpu_percent(), psutil.cpu_stats(), psutil.virtual_memory()
            )
        )
