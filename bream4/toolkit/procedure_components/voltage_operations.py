from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.device_interfaces.devices.promethion import PromethionGrpcClient

DEFAULT_FLICK_DURATION = 3.0
DEFAULT_REST_VOLTAGE = 0.0
DEFAULT_REST_DURATION = 1.0

DEFAULT_FLICK_VOLTAGE = 120.0
DEFAULT_VOLTAGE_ADJUSTMENT = -5
DEFAULT_VOLTAGE_GAP = 300


def maintain_relative_unblock_voltage(
    device: BaseDeviceInterface, current_bias_voltage: float, voltage_gap: int
) -> None:
    """
    This nanodeep will maintain a relative voltage gap between the unblock voltage and the common voltage, this will
    mean the unblock voltage is maintained as a consistent strength as the run progresses. the nanodeep will first
    assess if an adjustment is required and then if it is make the change.

    :param device: wrapper for communicating with hardware
    :param current_bias_voltage: The current common run voltage in mv
    :param voltage_gap: the relative gap that is to be maintained between the common and the unblock voltage
    """
    logger = logging.getLogger(__name__)
    is_promethion = isinstance(device, PromethionGrpcClient)

    if is_promethion:
        current_progressive_voltage = device.get_regeneration_current_voltage_clamp()
    else:
        current_progressive_voltage = device.get_unblock_voltage()
    new_raw_unblock_voltage = -(voltage_gap - abs(current_bias_voltage))
    new_unblock_flick_voltage = calculate_to_the_nearest_voltage(
        new_raw_unblock_voltage, device.get_minimum_unblock_voltage_multiplier()
    )

    # Clamp value if out of range and log-warn user
    if is_promethion:
        clamped_voltage = min(1000, max(-1000, new_unblock_flick_voltage))
        if clamped_voltage != new_unblock_flick_voltage:
            logger.warning(
                "Trying to set unblock voltage to %s but must be 1000 >= x >= -1000. Clamping to %s"
                % (new_unblock_flick_voltage, clamped_voltage)
            )
    else:
        clamped_voltage = min(0, max(-372, new_unblock_flick_voltage))
        if clamped_voltage != new_unblock_flick_voltage:
            logger.warning(
                "Trying to set unblock voltage to %s but must be 0 >= x >= -372. Clamping to %s"
                % (new_unblock_flick_voltage, clamped_voltage)
            )

    # Update value if needed
    if clamped_voltage != current_progressive_voltage:
        if is_promethion:
            device.set_regeneration_current_voltage_clamp(clamped_voltage)
        else:
            device.set_unblock_voltage(clamped_voltage)
        logger.info("Adjusting unblock voltage from %s to %s" % (current_progressive_voltage, clamped_voltage))


def calculate_to_the_nearest_voltage(voltage: float, minimum_voltage_adjustment: float) -> int:
    flick_voltage = int(minimum_voltage_adjustment * round(float(voltage) / minimum_voltage_adjustment))
    return flick_voltage


def global_flick(
    device: BaseDeviceInterface,
    current_run_voltage: Optional[float] = None,
    flick_voltage: Optional[float] = None,
    flick_duration: Optional[float] = None,
    rest_voltage: Optional[float] = None,
    rest_duration: Optional[float] = None,
    voltage_gap: Optional[int] = None,
    perform_relative_flick: bool = False,
) -> None:
    """
    Execute a global flick to attempt to free the pores.
    If parameters are missing, defaults will be filled in.

    This will perform:
    * rest_voltage for rest_duration
    * flick_voltage for flick_duration
    * rest_voltage for rest_duration

    The bias_voltage will be restored.
    :param current_run_voltage: the current running voltage of the experiment
    :param device: MinKNOW device wrapper
    :param flick_voltage: The voltage to raise to
    :param flick_duration: How long to stay at the raised voltage
    :param rest_voltage: The voltage to rest for
    :param rest_duration: How long to stay in the rest voltage
    :param voltage_gap: The proportion of the current voltage to offset the flick voltage by
    :param perform_relative_flick: bool that will make the flick response relative to the current voltage
    """
    logger = logging.getLogger(__name__)

    if flick_duration is None:
        flick_duration = DEFAULT_FLICK_DURATION
    if voltage_gap is None:
        voltage_gap = DEFAULT_VOLTAGE_GAP
    if rest_voltage is None:
        rest_voltage = DEFAULT_REST_VOLTAGE
    if rest_duration is None:
        rest_duration = DEFAULT_REST_DURATION

    if perform_relative_flick:
        if flick_voltage is not None:
            logger.warning(
                "The flick voltage passed in will not be used due to perform_relative_flick being set to True"
            )

        if current_run_voltage is None:
            current_run_voltage = device.get_bias_voltage()
        if current_run_voltage == 0:
            logger.warning("Current_voltage is 0mV the global flick will have no effect")
        multiplier = -1 if current_run_voltage < 0 else 1

        raw_voltage = (abs(current_run_voltage) - voltage_gap) * multiplier
        flick_voltage = calculate_to_the_nearest_voltage(raw_voltage, device.get_minimum_voltage_adjustment())

    elif current_run_voltage is not None:
        logger.warning(
            "The current_run_voltage passed in will not be used due to perform_relative_flick being set to False"
        )

    if flick_voltage is None:
        flick_voltage = DEFAULT_FLICK_VOLTAGE

    voltages = [rest_voltage, flick_voltage, rest_voltage]
    durations = [rest_duration, flick_duration, rest_duration]
    apply_voltages_for_durations(device, voltages, durations)


def apply_voltages_for_durations(device: BaseDeviceInterface, voltages: list[float], durations: list[float]) -> None:
    """Performs a series of voltage changes.
    Make sure voltages/durations are the same length as they will be paired

    bias_voltage will be restored.

    :param device: MinKNOW device wrapper
    :param voltages: list of voltages to apply
    :param durations: How long to stay at the voltages
    :returns: None

    """
    if len(voltages) != len(durations):
        raise RuntimeError(f"Voltages {voltages} and durations {durations} need to be same length")

    old_voltage = device.get_bias_voltage()
    for (voltage, pause) in zip(voltages, durations):
        device.set_bias_voltage(voltage)
        time.sleep(pause)

    device.set_bias_voltage(old_voltage)


def calculate_new_bias_voltage(
    device: BaseDeviceInterface,
    start_time: float,
    end_time: float,
    minimum_voltage_adjustment: float,
    voltages_to_exclude: Optional[list[float]] = None,
    tolerance: float = 0.01,
) -> float:
    """
    calculates the new bias voltage to run the experiment at, takes into account all voltage adjustments made during a
    set period and weights them for the amount of tie spent at the voltage, then rounds the value to the nearest viable
    voltage for the device being used
    :param device: the device wrapper
    :param start_time: seconds value for the start of the time to be assessed (monotonic)
    :param end_time: seconds time for the end of the time to be assessed (monotonic)
    :param minimum_voltage_adjustment: the minimum amount the device can adjust the voltage by, the new voltage should
    be a multiple of this value
    :param voltages_to_exclude: A list of voltages that should be excluded from the calculation
    :param tolerance: How close the voltage in question has to be to voltages_to_exclude
    :return: a new bias voltage to run the next part of the experiment with
    """

    logger = logging.getLogger(__name__)
    bias_voltages = device.get_bias_voltage_times(start=start_time, end=end_time)
    voltage_times = sorted(bias_voltages.keys())

    # If start_time isn't on a voltage boundary, nudge the closest one forward
    if voltage_times[0] < start_time:
        bias_voltages[start_time] = bias_voltages.pop(voltage_times[0])

    # Similarly, make sure the end time is included
    time_points = sorted(list(bias_voltages.keys()))
    time_points.append(end_time)

    voltages = []
    weights = []

    if not voltages_to_exclude:
        voltages_to_exclude = []
    exclude = set(voltages_to_exclude)

    for start_time, end_time in zip(time_points, time_points[1:]):
        voltage = bias_voltages[start_time]

        # Check we want this voltage
        if voltage in exclude:
            continue

        voltages.append(voltage)
        weights.append(end_time - start_time)

    logger.info("calculating new voltage using {} voltages with {} weights".format(voltages, weights))
    raw_voltage = np.average(voltages, weights=weights)
    new_bias_voltage = calculate_nearest_bias_voltage(minimum_voltage_adjustment, raw_voltage)
    return new_bias_voltage


def calculate_nearest_bias_voltage(minimum_voltage_adjustment, bias_voltage):
    """
    calculated the average voltage for a list of voltages and then rounds it to the nearest multiplier
    This is to ensure the value is valid for a given device, e.g on minknow like devices the voltage needs to be a
    multiple of 5
    """

    new_bias_voltage = int(minimum_voltage_adjustment * round(float(bias_voltage) / minimum_voltage_adjustment))

    return new_bias_voltage
