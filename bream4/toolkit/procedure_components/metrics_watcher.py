from __future__ import annotations

import logging
import threading
import time
from collections import deque

from grpc import RpcError

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.data_extraction.read_statistics import get_channel_read_classifications


class CurrentMetricsWatcher(threading.Thread):
    def __init__(
        self,
        device: BaseDeviceInterface,
        max_stored: int = 10,
        classification: str = "strand",
        collection_time: int = 15,
        pause_between: int = 45,
    ):
        """Watches the current_statistics of q90-q10 for the specified classification.

        Calling `get_metrics` will give the most recent classifications.
        It will collect for collection_time, append to buffer, then wait for pause_between.
        If no channels had the classification, {} is in the results

        :param device: MinKNOW device wrapper
        :param max_stored: How many data points to store(`None` if unbounded)
        :param classification: Classification to record (default strand)
        :param collection_time: How long a window to collect data for
        :param pause_between: Pause between each aggregation

        """
        self.device = device
        self.process = True

        # Store the most recent local_median info
        self._metrics = deque([], max_stored)

        self.classification = classification
        self.collection_time = collection_time
        self.pause_between = pause_between
        self.logger = logging.getLogger(__name__)

        super(CurrentMetricsWatcher, self).__init__()

    def get_metrics(self) -> list[dict[int, tuple[float, float]]]:
        """Returns a list of {channel-> (q_90-q_10, samples in classification)}
        with the newest being at the head of the list

        :returns: [{channel->(range, samplecount), ..]
        :rtype: list

        """
        return [item for item in self._metrics]

    def clear(self) -> None:
        """Clears the current metrics"""

        self._metrics.clear()

    def run(self) -> None:
        while self.process:

            # Get some statistics and get all the strand local_medians from all the channels
            try:
                stats = get_channel_read_classifications(
                    self.device,
                    self.collection_time,
                    completed_read=False,
                    include_current_statistics=False,
                    include_chunk_statistics=True,
                )
            except RpcError as exception:
                code = exception.code()

                if (
                    (code == code.FAILED_PRECONDITION)
                    or (code == code.ABORTED and "acquisition finished" in exception.details())
                    or self.device.get_acquisition_status() != self.device.ACQ_PROCESSING
                ):
                    self.logger.info("During get_read_statistics, Acquisition was stopped")
                    return

                self.logger.info("During get_read_statistics, RpcError {} was raised.".format(str(exception)))
                continue

            current_setup = self.device.get_channel_configuration()
            channels = {}
            for ((channel, well), classifications) in stats.items():
                # Only include the well we are supposed to be in
                if current_setup[channel].well != well:
                    continue

                if self.classification in classifications:
                    chunk_stats = classifications[self.classification].chunk_statistics
                    if channel in channels:
                        raise RuntimeError("got 2 values for a channel")
                    else:
                        channels[channel] = (
                            chunk_stats.range,
                            classifications[self.classification].samples_duration,
                        )

            # Above could take a long time. Make sure we still want this metric to be added
            if self.process:
                self._metrics.appendleft(channels)

                # Caveat: Won't be able to terminate thread.
                # We could set self.process to False and just not join the thread
                # Swap to condition
                time.sleep(self.pause_between)
