from __future__ import annotations

import copy
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.metrics_watcher import CurrentMetricsWatcher
from bream4.toolkit.procedure_components.output_locations import get_run_output_path
from bream4.toolkit.procedure_components.voltage_operations import maintain_relative_unblock_voltage

DEFAULTS = {
    "setpoint": 32,  # Try to keep (q90-q10) range close to this value (in pA)
    "lower_threshold": 0.8,  # if average of observed medians is outside setpoint +/- threshold then adjust
    "upper_threshold": 1.5,  # if average of observed medians is outside setpoint +/- threshold then adjust
    "initial_pA_adjustment_per_minimum_voltage_adjustment": 1,
    # the amount in pA that a single adjust_value will correct
    "channels_threshold": 50,  # This many channels need to be classified for a dynamic adjustment
    "classification": "strand",  # Use this as the source for the setpoint
    "voltage_fallback": -240,  # Once voltage goes outside this only use a static fallback
    "voltage_fallback_interval": 5400,  # Static fallback duration (seconds). Will correct with adjust_value
    "ewma_factor": 0.7,  # Weighted average. Closer to 1 == weight newest information more
    # Don't adjust outside of these limits
    "voltage_min": -500,
    "voltage_max": -150,
    "report": True,
    "unblock_voltage_gap": 300,
    "enable_relative_unblock_voltage": False,
    "voltage_max_adjustment": 10,
}

REPORT_HEADERS = [
    "current_median",
    "current_Q25",
    "current_Q75",
    "n_channels",
    "seconds_since_last_correction",
    "adjusted_voltage",
    "drift",
    "pa_corrected",
    "current_voltage",
    "previous_voltage",
    "seconds_since_start_of_run",
    "flow_cell_id",
    "sample_id",
    "bias_voltage_offset",
]


def ewma(values: list[float], factor: float) -> float:
    if len(values) == 1:
        return values[0]
    return values[0] * factor + (1 - factor) * ewma(values[1:], factor)


def clamp(n: float, smallest: float, largest: float) -> float:
    return max(smallest, min(n, largest))


class DriftCorrection(object):
    """
    The current flow cell design uses a mediator to facilitate electron current flow between two platinum electrodes.
    This chemical reaction where two molecules react with each other is driving towards an equilibrium for a given
    applied voltage, as the reaction approaches this point the current flow that's being generated decreases.
    Maintaining this rate of current flow is important because the software uses current levels to determine the state
    of the system. Its possible to correct this mediator drift by increasing the voltage and thus changing the
    equilibrium point of the reaction increasing the current flow back to previous levels.

    This feature monitors and determines when the current has dropped too low and then corrects the drift via a common
    voltage adjustment.
    """

    def __init__(
        self,
        config: dict,
        device: BaseDeviceInterface,
        unblock_voltage_gap: Optional[int] = None,
        enable_relative_unblock_voltage: Optional[bool] = None,
    ):

        self.config = copy.deepcopy(DEFAULTS)
        self.config.update(config)
        self.device = device

        self.logger = logging.getLogger(__name__)

        self.setpoint = self.config["setpoint"]

        self.lower_threshold = self.config["lower_threshold"]
        self.upper_threshold = self.config["upper_threshold"]
        self.minimum_voltage_adjustment = self.config.get(
            "minimum_voltage_adjustment", device.get_minimum_voltage_adjustment()
        )

        self.initial_pA_adjustment_per_minimum_voltage_adjustment = self.config[
            "initial_pA_adjustment_per_minimum_voltage_adjustment"
        ]
        self.pA_adjustment_per_minimum_voltage_adjustment = self.config[
            "initial_pA_adjustment_per_minimum_voltage_adjustment"
        ]

        self.channels_threshold = self.config["channels_threshold"]
        self.classification = self.config["classification"]

        self.fallback_voltage = self.config["voltage_fallback"]
        self.fallback_interval = self.config["voltage_fallback_interval"]
        self.in_fallback_mode = False

        self.ewma_factor = self.config["ewma_factor"]
        self.min_voltage = self.config["voltage_min"]
        self.max_voltage = self.config["voltage_max"]
        self.max_voltage_adjustment = self.config["voltage_max_adjustment"]

        if enable_relative_unblock_voltage is None:
            self.enable_relative_unblock_voltage = DEFAULTS["enable_relative_unblock_voltage"]
        else:
            self.enable_relative_unblock_voltage = enable_relative_unblock_voltage

        if unblock_voltage_gap is None:
            self.unblock_voltage_gap = DEFAULTS["unblock_voltage_gap"]
        else:
            self.unblock_voltage_gap = unblock_voltage_gap

        self.flow_cell = device.get_flow_cell_id()
        self.experiment_tag = device.get_sample_id()

        # Dont start the watcher on init, only when feature manager tells us to go
        self.metrics_watcher = None

        self.paused = False

        self.previous_drift_data = None
        self.last_adjusted_time = time.monotonic()

        if self.config["report"]:
            self.report_path = get_run_output_path(self.device, "drift_correction.csv")

    def drift_correction_report(self, df: pd.DataFrame) -> None:
        """Write the DataFrame to the logs directory

        :param df: Pandas DataFrame
        """
        # Append to CSV. Only write header if this file doesn't yet exist
        df.to_csv(
            self.report_path,
            mode="a",
            header=not os.path.exists(self.report_path),
            index=False,
        )

    def check_for_adjustment(self, local_range: float, min_adjustments: int = 0) -> tuple[float, float]:
        """
        Takes the local range and calculates what if any voltage is required to get the range as close to the set point
        as possible.

        Returns the level of drift and the voltage adjustment required to get the signal back to setpoint

        logic:
        first calculate how much the signal has drifted from the set point
        then work out how many minimum adjustments are required to get back to set point

        Make sure we make at least min_adjustments

        :param local_range: the local range of the data that was collected.
        :param min_adjustments: Make sure at least this many adjustments are returned
        :return: new voltage adjustment and the amount of drift from the set point the local_range was.
        """

        initial_drift = local_range - self.setpoint
        adjust = abs(self.pA_adjustment_per_minimum_voltage_adjustment)
        adjust = adjust if initial_drift <= 0 else -adjust

        # this needs to be the same as the drift as a negative voltage increases range
        voltage_multiplier = -1 if initial_drift <= 0 else 1
        num_adjustments = 0
        drift = initial_drift

        # Move until we're as close as we can be
        while abs(adjust) and abs(drift) > abs(adjust / 2):
            num_adjustments += 1
            local_range += adjust
            drift = local_range - self.setpoint

        num_adjustments = max(min_adjustments, num_adjustments)
        return (
            min(
                abs(num_adjustments * self.minimum_voltage_adjustment),
                self.max_voltage_adjustment,
            )
            * voltage_multiplier,
            initial_drift,
        )

    def execute(self, **kwargs) -> None:

        """
        Main execute call that will assess the data collected. On start of the feature a metrics_watcher object is
        started in a new thread. This will collect data from the device on a regular timed interval and aggregates this
        information. This method is then executed on a timed interval and assesses the metrics watcher data. It then
        determines if an adjustment is required.
        """

        if self.paused:
            self.logger.info("Tried to run but paused")
            return

        ###########################
        #   Collect metric data   #
        ###########################

        results = []
        if not self.metrics_watcher:
            raise RuntimeError("Execute called before metrics_watcher instantiated")
        metrics = self.metrics_watcher.get_metrics()

        if not metrics:
            self.logger.info("No metrics received")
            return

        # --------  Filter it on the minimum number of channels -------- #
        # Only include if x channels or more were in the metric, -- too small a number might skew the decision
        all_local_medians = []
        total_unique_channels = []
        for assessment_period in metrics:
            for channel, channel_data in assessment_period.items():
                all_local_medians.append(channel_data[0])
                total_unique_channels.append(channel)
        total_unique_channels = len(set(total_unique_channels))

        if total_unique_channels < self.channels_threshold:
            # If it == set point only fallback will happen
            local_range = self.setpoint
            Q25, Q75 = [np.nan] * 2

        else:
            # -------- calculate the new levels to assess -------- #
            Q25, Q75 = np.percentile(all_local_medians, [25, 75])
            local_range = ewma(
                [np.mean([b[0] for b in metric.values()]) for metric in metrics if metric],
                self.ewma_factor,
            )

        # --------  if already made an adjustment calculate the amount the current was adjusted by -------- #

        if self.previous_drift_data:
            self.pA_adjustment_per_minimum_voltage_adjustment = abs(
                (local_range - self.previous_drift_data[0])
                / (abs(self.previous_drift_data[1]) / self.minimum_voltage_adjustment)
            )

        results.extend(
            [
                round(local_range, 2),
                round(Q25, 2),
                round(Q75, 2),
                total_unique_channels,
                time.monotonic() - self.last_adjusted_time,
            ]
        )

        adjust_value = 0
        drift = 0
        current_voltage = self.device.get_bias_voltage()

        # Check for adjustment

        # check if the local range has dropped out of the threshold
        if current_voltage > self.fallback_voltage and (
            local_range < self.setpoint - self.lower_threshold or local_range > self.setpoint + self.upper_threshold
        ):
            adjust_value, drift = self.check_for_adjustment(local_range, min_adjustments=1)
            self.in_fallback_mode = False

        # We have lots of channels to make an assessment on. So remove fallback
        elif current_voltage > self.fallback_voltage and total_unique_channels > self.channels_threshold:
            self.in_fallback_mode = False

        # Then check for fallback
        elif self.last_adjusted_time + self.fallback_interval <= time.monotonic():
            self.logger.info("Performing fallback adjustment")
            self.in_fallback_mode = True
            adjust_value = -self.minimum_voltage_adjustment

            # Make sure we start from a safe state if we come back out of fallback mode
            self.pA_adjustment_per_minimum_voltage_adjustment = (
                self.initial_pA_adjustment_per_minimum_voltage_adjustment
            )

        results.extend([adjust_value, drift, self.pA_adjustment_per_minimum_voltage_adjustment])

        # If we do need to adjust
        previous_voltage = self.device.get_bias_voltage()
        bias_voltage_offset = self.device.get_bias_voltage_offset()
        if adjust_value:
            new_voltage = previous_voltage + adjust_value

            # Make sure value is within limits
            if new_voltage < self.min_voltage or new_voltage > self.max_voltage:
                self.logger.info("Trying to set voltage to {} but outside range".format(new_voltage))
                new_voltage = clamp(new_voltage, self.min_voltage, self.max_voltage)

            # Adjust voltage
            self.logger.info("Adjusting voltage from {} to {}".format(previous_voltage, new_voltage))
            self.device.set_bias_voltage(new_voltage)
            if self.enable_relative_unblock_voltage:
                maintain_relative_unblock_voltage(self.device, new_voltage, self.unblock_voltage_gap)

            self.last_adjusted_time = time.monotonic()

            # We have no good data to rely on if we are in fallback
            if self.in_fallback_mode:
                self.previous_drift_data = None
            else:
                self.previous_drift_data = local_range, adjust_value

            # Used this data. Start again
            self.metrics_watcher.clear()
        else:
            self.previous_drift_data = None

        if self.config["report"] and (metrics or adjust_value):
            bias_voltage = self.device.get_bias_voltage()
            number_of_samples = self.device.get_current_sample_number()
            sample_rate = self.device.get_sample_rate()
            results.extend(
                [
                    bias_voltage,
                    previous_voltage,
                    number_of_samples / sample_rate,
                    self.flow_cell,
                    self.experiment_tag,
                    bias_voltage_offset,
                ]
            )
            data = pd.DataFrame([results], columns=REPORT_HEADERS)

            self.drift_correction_report(data)

    def stop(self) -> None:
        self.logger.info("Asked to stop")

        if self.metrics_watcher:
            self.metrics_watcher.process = False
        else:
            self.logger.info("No metrics watcher was present")
        self.pA_adjustment_per_minimum_voltage_adjustment = self.initial_pA_adjustment_per_minimum_voltage_adjustment
        self.previous_drift_data = None
        self.paused = True

    def resume(self) -> None:
        self.logger.info("Asked to resume")
        self.paused = False
        if self.metrics_watcher and self.metrics_watcher.process:
            raise RuntimeError("Previous watcher manager not exited correctly")
        self.metrics_watcher = CurrentMetricsWatcher(self.device, classification=self.classification)
        self.metrics_watcher.start()

    def reset(self) -> None:
        self.logger.info("Asked to reset")
        # Don't reset the metrics on mux change
        self.paused = True

    def exit(self) -> None:
        self.logger.info("Asked to exit")
        self.paused = True
        if self.metrics_watcher:
            self.metrics_watcher.process = False
        else:
            self.logger.info("No metrics watcher was present")
