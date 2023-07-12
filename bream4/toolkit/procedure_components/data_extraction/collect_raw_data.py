from __future__ import annotations

import logging
import time

import numpy as np
from minknow_api.data import get_signal

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface


def collect_raw_data(
    device: BaseDeviceInterface, collection_time_sec: float, calibrated: bool = True
) -> list[np.ndarray]:
    """
    Captures raw data, for a given amount of time up to a maximum amount.

    [ [raw_data], [raw_data], ...] is returned.

    If more information(Such as channel config) Please look at, and call,
    minknow_api.data.get_signal directly.

    :param device: Client used to communicate with MinKNOW
    :param collection_time_sec: The amount of data to be collected in seconds
    :param calibrated: If True return pA data, if False return ADC data
    """

    logger = logging.getLogger(__name__)
    logger.info(
        "Capturing {} seconds of raw data, current "
        "sample number: {}".format(collection_time_sec, device.get_current_sample_number())
    )

    start = time.monotonic()

    signal_data = get_signal(
        device.connection,
        seconds=collection_time_sec,
        calibrated_data=calibrated,
        first_channel=1,
        last_channel=device.channel_count,
        include_channel_configs=True,
    )

    time_taken = time.monotonic() - start
    level = logging.DEBUG
    if time_taken > 2 + (collection_time_sec * 1.1):
        level = logging.WARNING
    logger.log(
        level,
        "Getting raw data took {:.2f} seconds,"
        " current sample number: {}".format(time_taken, device.get_current_sample_number()),
    )

    return [channel.signal for channel in signal_data.channels]
