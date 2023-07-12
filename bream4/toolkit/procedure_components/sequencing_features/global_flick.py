from __future__ import annotations

import logging
import warnings

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.voltage_operations import apply_voltages_for_durations


class GlobalFlick(object):
    def __init__(self, config: dict, device: BaseDeviceInterface):
        """
        global flick class designed to run with the feature manager in the sequencing experiments.
        It has limited flick structure scope only allowing a single flick voltage and rest voltage to be passed.
        :param config: config for class
        :param device: device wrapper
        """
        warnings.warn(
            "deprecated, please use voltage_operations.global_flick for global flicking",
            DeprecationWarning,
        )
        self.config = config
        self.device = device

        self.logger = logging.getLogger(__name__)

        self.voltages = self.config.get("voltages", [0, 120, 0])
        self.pause_between = self.config.get("adjustment_pause", [1, 2, 1])

        if len(self.voltages) != len(self.pause_between):
            raise RuntimeError("Voltages and adjustment_pause must be the same length")

    def execute(self) -> None:
        apply_voltages_for_durations(self.device, self.voltages, self.pause_between)  # type: ignore
