from __future__ import annotations

import time
from typing import Optional

import pandas as pd

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.device_interfaces.devices.promethion import PromethionGrpcClient

DEFAULT_VOLTAGE = -180
DEFAULT_DURATION = 1.0
DELAY_HW_SATURATION_RESTORE_SECONDS = 0.5


def _promethion_manual_flick(device: PromethionGrpcClient, well: int, duration: float) -> None:
    """Because of certain chip issues, MinKNOW have implemented a work around.
    To test the effect without the workaround, this nanodeep was implemented
    """

    # Grab the old overload mode
    settings = device.connection.promethion_device.get_pixel_settings(pixels=[1])
    overload_mode = settings.pixels[0].overload_mode

    # Move to clear overload mode
    to_clear = device.prom_msgs.PixelSettings()
    to_clear.overload_mode = device.prom_msgs.PixelSettings.OVERLOAD_CLEAR
    device.connection.promethion_device.change_pixel_settings(pixel_default=to_clear)

    # Move to regen mux
    to_regen = device.prom_msgs.PixelSettings()
    to_regen.input.input_well = 0
    to_regen.input.regeneration_well = well
    device.connection.promethion_device.change_pixel_settings(pixel_default=to_regen)

    # Wait for flick to happen
    time.sleep(duration)

    # Move back to non regen mux
    to_non_regen = device.prom_msgs.PixelSettings()
    to_non_regen.input.input_well = well
    to_non_regen.input.regeneration_well = 0
    device.connection.promethion_device.change_pixel_settings(pixel_default=to_non_regen)

    time.sleep(DELAY_HW_SATURATION_RESTORE_SECONDS)

    # Move back to old mode
    reset = device.prom_msgs.PixelSettings()
    reset.overload_mode = overload_mode
    device.connection.promethion_device.change_pixel_settings(pixel_default=reset)


def collect_active_unblock_data(
    device: BaseDeviceInterface,
    duration: Optional[float] = None,
    voltage: Optional[float] = None,
    muxes: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Collects information on whether channels can survive an active unblock

    :param device:
    :param duration: How long to flick for
    :param voltage: What voltage to set the well under
    :param muxes: Which wells to perform on. If None specified, all will be used

    :returns: df with `channel/mux/saturated_after_unblock`
    :rtype: Pandas DataFrame
    """

    if not duration:
        duration = DEFAULT_DURATION

    if not voltage:
        voltage = DEFAULT_VOLTAGE

    if not muxes:
        muxes = device.get_well_list()

    device.set_bias_voltage(0)

    metrics = []

    for well in muxes:
        device.set_all_channels_to_well(well)
        device.set_bias_voltage(voltage)

        saturated_before = device.get_disconnection_status_for_active_wells()

        if not isinstance(device, PromethionGrpcClient):
            device.unblock(device.get_channel_list(), duration)
            time.sleep(duration)
        else:
            _promethion_manual_flick(device, well, duration)

        saturated_after = device.get_disconnection_status_for_active_wells()

        # Calculate the difference
        saturated = {
            channel: not saturated_before[channel] and saturated_after[channel] for channel in saturated_before.keys()
        }

        for channel, is_saturated in saturated.items():
            metrics.append(
                {
                    "channel": channel,
                    "well": well,
                    "saturated_after_unblock": is_saturated,
                }
            )

        device.set_bias_voltage(0)

    df = pd.DataFrame(metrics)
    return df.set_index(["channel", "well"])
