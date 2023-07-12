from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd

from bream4.device_interfaces.devices.minion import MinionGrpcClient
from bream4.toolkit.procedure_components.data_extraction.collect_continuous_data import (
    collect_continuous_data_current_setup,
)

DEFAULT_CONFIG = {
    "offset_current": 0,
    "scaling_current": 100,
    "actual_current_delta": 80,
    "collection_time": 1.0,
    "offset_current_min": -1,
    "offset_current_max": 1,
    "scale_current_min": 60,
    "scale_current_max": 100,
    "pass_percent": 0.9,
}


def collect_data(device: MinionGrpcClient, config: Optional[dict] = None) -> pd.DataFrame:
    """Generates the mean of each channel at the offset and scale current

    :param device: MinKNOW device wrapper
    :param config: dict of which parameters to alter. If none, then the defaults are used
    :returns: Pandas DataFrame of ['channel', 'offset_mean', 'scale_mean']
    :rtype: Pandas DataFrame
    """

    if not config:
        config = {}

    temp_config = copy.deepcopy(DEFAULT_CONFIG)
    temp_config.update(config)
    config = temp_config

    # get data at offset current
    device.set_all_channels_to_test_current()
    device.set_test_current(config["offset_current"])
    offset_data = collect_continuous_data_current_setup(
        device, 0, config["collection_time"], ["mean"], calibrated=False
    )

    # get data at scale current
    device.set_all_channels_to_test_current()
    device.set_test_current(config["scaling_current"])
    scale_data = collect_continuous_data_current_setup(device, 0, config["collection_time"], ["mean"], calibrated=False)

    data = offset_data.rename(columns={"mean": "offset_mean"})
    data["scale_mean"] = scale_data["mean"]

    # Cleanup
    device.set_test_current(0)
    device.set_all_channel_inputs_to_disconnected()

    return data.reset_index().drop(["well"], "columns")


def assess_data(
    device: MinionGrpcClient, data: pd.DataFrame, config: Optional[dict] = None
) -> tuple[pd.DataFrame, bool]:
    """Assess the data collected for calibration

    :param device: MinKNOW device wrapper
    :param data: The std/mean for each channel over a voltage
    :param config: dict of which parameters to alter.
    :returns: Table with channel/offset/range
    :rtype: Pandas DataFrame, whether passed

    """
    if not config:
        config = {}

    temp_config = copy.deepcopy(DEFAULT_CONFIG)
    temp_config.update(config)
    config = temp_config

    # Find the range
    mean_delta_current = (data["scale_mean"] - data["offset_mean"]).mean()

    if mean_delta_current == 0:
        raise RuntimeError("Cannot calibrate: device signal is not changing " + "when test current is altered")

    scaling = config["actual_current_delta"] / mean_delta_current
    range_pa = scaling * device.digitisation

    # Add range to df
    data["range"] = range_pa

    # Calculate the actual offsets which is just the inverse of the offset current levels
    data["offset"] = np.rint(-1 * data["offset_mean"])

    # Check the offsets/scales are in bounds
    data["offset_pA"] = scaling * (data["offset_mean"] + data["offset"])
    data["offset_pass"] = data["offset_pA"].between(config["offset_current_min"], config["offset_current_max"])

    data["scale_pA"] = scaling * (data["scale_mean"] + data["offset"])
    data["scale_pass"] = data["scale_pA"].between(config["scale_current_min"], config["scale_current_max"])

    # Work out how many passed
    data["pass"] = data["offset_pass"] & data["scale_pass"]
    percent_passed = data["pass"].sum() / len(data.index)

    return data, bool(percent_passed >= config["pass_percent"])
