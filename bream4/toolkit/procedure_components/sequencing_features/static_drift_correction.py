from __future__ import annotations

import copy
import logging

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.voltage_operations import maintain_relative_unblock_voltage

DEFAULTS = {
    "voltage_min": -500,
    "voltage_max": -150,
    "unblock_voltage_gap": 300,
    "enable_relative_unblock_voltage": False,
}


class StaticDriftCorrection(object):
    def __init__(self, config: dict, device: BaseDeviceInterface):
        self.config = copy.deepcopy(DEFAULTS)
        self.config.update(config)
        self.device = device

        self.logger = logging.getLogger(__name__)

        self.minimum_voltage_adjustment = self.config.get(
            "minimum_voltage_adjustment", self.device.get_minimum_voltage_adjustment()
        )
        self.voltage_min = self.config["voltage_min"]
        self.voltage_max = self.config["voltage_max"]

        self.unblock_voltage_gap = None
        if self.config["enable_relative_unblock_voltage"]:
            self.unblock_voltage_gap = self.config["unblock_voltage_gap"]

    def execute(self) -> None:
        old_voltage = new_voltage = self.device.get_bias_voltage()
        new_voltage -= self.minimum_voltage_adjustment

        if new_voltage <= self.voltage_min or new_voltage >= self.voltage_max:
            self.logger.info(
                "Can't adjust to voltage {} as it is outside of {}, {} limits".format(
                    new_voltage, self.voltage_min, self.voltage_max
                )
            )
        else:
            self.logger.info("Adjusting voltage from {} to {}".format(old_voltage, new_voltage))
            self.device.set_bias_voltage(new_voltage)

            if self.unblock_voltage_gap is not None:
                maintain_relative_unblock_voltage(self.device, new_voltage, self.unblock_voltage_gap)
