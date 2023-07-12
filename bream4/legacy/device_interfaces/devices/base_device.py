from __future__ import annotations

import time
from typing import Optional

# Temporary store for which channels passed/failed calibration
# TODO: Will be replaced by CORE-32
CALIB_PASS_CHANNELS: dict[int, bool] = {}

# TODO: Remove for [CORE-906] to tell how long a channel has been in each well
CHANNEL_TIME_IN_WELL: dict[int, dict] = {}

BIAS_VOLTAGE_OVER_TIME: dict[float, float] = {}

BIAS_VOLTAGE_OFFSET = 0.0


def reset_well_timings() -> None:
    """Resets any timings recorded for how long each channel has been in their wells"""
    global CHANNEL_TIME_IN_WELL
    CHANNEL_TIME_IN_WELL = {}


def _push_well_config() -> None:
    t = time.monotonic()
    for channel, info in CHANNEL_TIME_IN_WELL.items():
        last_well = info["last_well"]
        if last_well not in info:
            info[last_well] = t - info["last_time"]
        else:
            info[last_well] += t - info["last_time"]

        info["last_time"] = t


def channel_well_configuration(config: dict[int, int]) -> None:
    """Record how long channels have been asked to be in specific wells

    :param config dict: channel->well
    """
    t = time.monotonic()
    _push_well_config()
    for (channel, well) in config.items():
        if channel not in CHANNEL_TIME_IN_WELL:
            CHANNEL_TIME_IN_WELL[channel] = {"last_well": well, "last_time": t}
        else:
            CHANNEL_TIME_IN_WELL[channel]["last_time"] = t
            CHANNEL_TIME_IN_WELL[channel]["last_well"] = well


def channel_well_times(well_list: list[int]) -> dict[int, dict[int, float]]:
    """Return how long each channel has spent in each well
    You can include amount of time it has been disconnected by including 0

    :param well_list: The wells to get stats for
    :rtype: dict of dict
    :returns: channel-> (well->seconds)

    """
    # Make sure most up to date times
    _push_well_config()
    # Generate the return values
    ret = {}
    for (channel, info) in CHANNEL_TIME_IN_WELL.items():
        ret[channel] = {well: round(info.get(well, 0), 3) for well in well_list}
    return ret


def get_bias_voltage_times(start: Optional[float] = None, end: Optional[float] = None) -> dict[float, float]:
    """
    will be removed with the completion of INST-1730
    returns a dict of voltages, the keys are the time points the values are the voltages
    :param start: seconds that the search will start from (monotonic)
    :param end: seconds the search will return on (monotonic)

    if no values are passed all data is returned
    :return: a dict of time_stamps : voltage, for the given start and end times.
    """

    time_points = sorted(BIAS_VOLTAGE_OVER_TIME.keys())
    if start is None:
        start = time_points[0]
    if end is None:
        end = time_points[-1]

    time_points_to_return = [time_point for time_point in time_points if start <= time_point <= end]
    if time_points_to_return[0] > start:
        initial_index = time_points.index(time_points_to_return[0])
        if initial_index != 0:
            added_index = initial_index - 1
            time_points_to_return.insert(0, time_points[added_index])
    return {
        time_point: voltage
        for time_point, voltage in BIAS_VOLTAGE_OVER_TIME.items()
        if time_point in time_points_to_return
    }


###############
# Get Methods #
###############


def get_calibration_status_all_channels() -> dict[int, bool]:
    # TODO: This nanodeep will be replaced by CORE-32
    """
    Return dict whether channel passed or failed calibration

    :return: dict of 1-indexed channel to boolean
    """
    return CALIB_PASS_CHANNELS


def get_bias_voltage_offset() -> float:
    # Remove when CORE-1416/INST-443 in
    """Return the offset that is applied to any set_bias_voltage calls

    :returns: offset
    :rtype: int

    """
    return BIAS_VOLTAGE_OFFSET


###############
# Set Methods #
###############


def set_channel_calibration_status(channel_dict: dict[int, bool]) -> None:
    # TODO: This nanodeep will be replaced by CORE-32
    """
    Store which channels passed/failed calibration

    :param channel_dict: dict of 1-indexed channel to booleans representing
        whether channel passed or failed calibration
    """
    global CALIB_PASS_CHANNELS
    CALIB_PASS_CHANNELS = channel_dict


def set_bias_voltage_offset(offset: float) -> None:
    # Remove when CORE-1416/INST-443 in
    """
    Store offset that is applied to any set_bias_voltage calls
    :param offset: int of offset

    """
    global BIAS_VOLTAGE_OFFSET
    BIAS_VOLTAGE_OFFSET = offset


def set_bias_voltage(value: float) -> None:
    """
    will be removed with the completion of INST-1730
    stores the voltage that's being set against the time that it was set (monotonic)
    :param value: voltage to be set
    """
    time_point = time.monotonic()
    BIAS_VOLTAGE_OVER_TIME[time_point] = value
