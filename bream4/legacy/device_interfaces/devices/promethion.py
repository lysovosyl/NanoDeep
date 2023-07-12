from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bream4.device_interfaces.devices.promethion import PromethionGrpcClient


# INST-73
def set_channels_to_well(device: "PromethionGrpcClient", channel_config: dict[int, int]) -> None:
    # We need to make sure saturation is cleared during a mux change
    # will not clear latched saturation

    # We also shouldn't fully disable saturation control as that will cause a full reset
    # of saturation including channels that aren't changing wells

    # Grab the old overload mode
    settings = device.connection.promethion_device.get_pixel_settings(pixels=[1])
    overload_mode = settings.pixels[0].overload_mode

    # Set the specific channels to clear
    new_settings = device.prom_msgs.PixelSettings()
    new_settings.overload_mode = device.prom_msgs.PixelSettings.OVERLOAD_CLEAR
    device.connection.promethion_device.change_pixel_settings(
        pixels={channel: new_settings for channel in channel_config}
    )

    # Put them back to the old overload mode
    new_settings = device.prom_msgs.PixelSettings()
    new_settings.overload_mode = overload_mode
    device.connection.promethion_device.change_pixel_settings(
        pixels={channel: new_settings for channel in channel_config}
    )


# INST-73
def set_all_channels_to_well(device: "PromethionGrpcClient", well: int) -> None:
    # We need to make sure saturation is cleared as a mux change
    # will not clear latched saturation
    device.clear_saturation(device.get_overload_mode(), saturation_control_enabled=True)


# INST-2505
def step_bias_voltage(device: "PromethionGrpcClient", bias_voltage: float) -> None:
    # This is to sidestep capacitance spikes appearing on certain chip batches
    # Final voltage will be set by caller

    current_voltage = int(device.get_bias_voltage())

    voltages = []
    if current_voltage < bias_voltage:
        voltages = list(range(current_voltage, int(bias_voltage), 10))
    else:
        voltages = list(range(current_voltage, int(bias_voltage), -10))

    voltage_offset = device.get_bias_voltage_offset()

    # Ignore first voltage step as we are already there
    for voltage in voltages[1:]:
        device.connection.device.set_bias_voltage(bias_voltage=voltage + voltage_offset)


# INST-1509
# BREAM-518
# BREAM-527
def set_sample_rate(device: "PromethionGrpcClient", sample_rate: int) -> None:
    # Give device chance to settle on the new sample_rate to mitigate jumbling
    time.sleep(10)
