from __future__ import annotations

import logging
import os
import pprint
from typing import Optional

import grpc
import minknow_api.statistics_pb2 as mk_stats
import pandas as pd

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.grpc_streamer import GrpcStreamer
from bream4.toolkit.procedure_components.output_locations import get_run_output_path

DEFAULT_CONFIG = {
    "target_speed": 400,  # What target bps to reach
    "target_speed_tolerance": 2,  # Tolerance for speed (398-402 in this case)
    "bases_per_degree": 30,  # How many more bps one degree C adjust will give you
    "disable_after_target_reached": True,  # Whether to disable the feature once target reached
    "upper_temperature_limit": 40,  # Cap at this max temperature
    "lower_temperature_limit": 30,  # Cap at this min temperature
    "read_count_threshold": 100,  # Need at least this many reads since the last adjust to adjust
}
CONFIG_OPTIONS = set(DEFAULT_CONFIG.keys())
CONFIG_OPTIONS.update(["enabled", "interval"])  # Not specified but we can check subset then


class BoxplotStreamer(GrpcStreamer):
    def __init__(self, device: BaseDeviceInterface):
        super().__init__(device)
        # Set as daemon to ensure program can end (stream doesn't end on acq stop)
        self.daemon = True

        self.acquisition_run_id = device.get_run_id()
        self.results: list[mk_stats.BoxplotResponse.BoxplotDataset] = []

    def get_stream(self) -> grpc.Call:
        return self.device.connection.statistics.stream_basecall_boxplots(
            acquisition_run_id=self.acquisition_run_id,
            data_type=mk_stats.StreamBoxplotRequest.BoxplotType.BASES_PER_SECOND,
            dataset_width=10,  # No choice
        )

    def process_item(self, item: mk_stats.BoxplotResponse) -> None:
        self.results = item.datasets


class TemperatureManager:
    def __init__(self, config: dict, device: BaseDeviceInterface):
        self.logger = logging.getLogger(__name__)

        self.config = DEFAULT_CONFIG.copy()
        self.config.update(config)
        self.device = device

        self.logger.info("Running with options:")
        self.logger.info(pprint.pformat(self.config))

        self.report_path = get_run_output_path(self.device, "temperature_adjust_data.csv")
        self.target_min = self.config["target_speed"] - self.config["target_speed_tolerance"]
        self.target_max = self.config["target_speed"] + self.config["target_speed_tolerance"]

        self.last_length_seen = 0  # How many box plot bins were last seen
        self.last_bucket_used = 0  # Which box plot bucket was last used for a decision

        self._execute = False
        self._streamer = BoxplotStreamer(device)

        if self.check_prerequisites():
            self._streamer.start()
            self._execute = True

    def update_csv_results(
        self, temp: float, num_reads: int, speed: Optional[float] = None, new_temp: Optional[float] = None
    ) -> None:
        """Add a new CSV row to the report_path file. If not yet created, headers will be added"""
        df = pd.DataFrame(
            [[self.device.get_acquisition_duration(), temp, num_reads, speed, new_temp]],
            columns=[
                "acquisition_duration",
                "current_target_temperature",
                "num_reads",
                "current_speed",
                "new_target_temperature",
            ],
        )
        df.to_csv(self.report_path, mode="a", header=not os.path.exists(self.report_path), index=False)

    def check_prerequisites(self) -> bool:
        """Raise an error if any config options are invalid. Return true if
        other prerequisites met. (In this case basecalling is enabled)
        """
        if not set(self.config.keys()).issubset(CONFIG_OPTIONS):
            raise RuntimeError(f"Invalid parameters specified in {list(self.config.keys())}")

        if not self.device.connection.analysis_configuration.get_basecaller_configuration().enable:
            self.logger.warning("Basecalling is not enabled so cannot currently adjust temperature for speed")
            return False

        return True

    def execute(self):
        """Main method of this file. Grab the latest bps from the basecall stream
        And if we have new data, and the speed isn't in range, adjust the temperature.
        """
        if not self._execute:
            return

        results = self._streamer.results[::]
        buckets_to_consider = results[self.last_bucket_used :]
        reads_to_consider = sum([bucket.count for bucket in buckets_to_consider])
        current_temperature = self.device.connection.device.get_temperature().target_temperature

        current_speed = None
        if reads_to_consider:
            # Get a weighted average
            current_speed = sum([(bucket.q50 * bucket.count) for bucket in buckets_to_consider]) / reads_to_consider

        # If we don't have enough reads, wait for more (We haven't used the data so don't increment last_bucket_used)
        if reads_to_consider < self.config["read_count_threshold"]:
            self.logger.info(
                f"Received {reads_to_consider} reads which is below the threshold of "
                + f"{self.config['read_count_threshold']}"
            )
            self.update_csv_results(current_temperature, reads_to_consider, current_speed)
            return

        if buckets_to_consider:

            # Performing a calculation so class all buckets as used
            self.last_bucket_used = len(results)

            if self.target_min <= current_speed <= self.target_max:
                # Speed achieved
                self.logger.info(f"Reached optimum speed {current_speed} with temperature {current_temperature}.")
                if self.config["disable_after_target_reached"]:
                    self.logger.info("Disabling feature as requested")
                    self._execute = False
                    self._streamer.stop()

                self.update_csv_results(current_temperature, reads_to_consider, current_speed)
                return

            speed_adjust = self.config["target_speed"] - current_speed  # too slow
            temperature_delta = round(speed_adjust / self.config["bases_per_degree"], 2)
            new_temperature = current_temperature + temperature_delta  # so speed up

            if new_temperature > self.config["upper_temperature_limit"]:
                self.logger.warning(
                    f"Cannot adjust temperature to {new_temperature} as it's above the limit of "
                    + f"{self.config['upper_temperature_limit']}"
                )
                new_temperature = self.config["upper_temperature_limit"]

            if new_temperature < self.config["lower_temperature_limit"]:
                self.logger.warning(
                    f"Cannot adjust temperature to {new_temperature} as it's below the limit of "
                    + f"{self.config['lower_temperature_limit']}"
                )
                new_temperature = self.config["lower_temperature_limit"]

            self.logger.info(
                f"Adjusting temperature from {current_temperature} to "
                + f"{new_temperature} based on current speed of {current_speed}"
            )
            self.device.set_temperature(target=new_temperature, timeout=0)
            self.update_csv_results(current_temperature, reads_to_consider, current_speed, new_temperature)
