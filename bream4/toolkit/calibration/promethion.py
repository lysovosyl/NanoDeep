from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np
import pandas as pd

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.device_interfaces.devices.promethion import PromethionGrpcClient
from bream4.toolkit.procedure_components.data_extraction.collect_continuous_data import (
    collect_continuous_data_current_setup,
)

# PromethION variables
# TODO: nasty to hard-code these here, see bream4 issue #93
V_REF = 1.49
FRAME_TIME_DIFF = 21.1 * 10 ** -6  # from the datasheet
ADC_GAIN = 1 / 0.92  # from the datasheet

PICOAMP_CONVERT = 1 * 10 ** 12
FF_CONVERT = 1 * 10 ** -15

QUANTILES = [0, 0.01, 0.25, 0.5, 0.75, 0.99, 1]

DEFAULT_CONFIG = {
    "offset_voltage": 0,
    "collection_time": 1.0,
    "offset_current_min": -1.5,
    "offset_current_max": 1.5,
    "sd_max": 10,
    "hard_min": -1000,
    "hard_max": 1000,
    "pass_percent": 0.9,
}


def collect_data(device: PromethionGrpcClient, config: Optional[dict] = None) -> pd.DataFrame:
    """Generates the mean and std of each channel on the offset_voltage config option

    :param device: MinKNOW device wrapper
    :param config: dict of which parameters to alter. If none, then the defaults are used
    :returns: Pandas DataFrame of ['channel', 'mean', 'std']
    :rtype: Pandas DataFrame

    """
    if not config:
        config = {}

    temp_config = copy.deepcopy(DEFAULT_CONFIG)
    temp_config.update(config)
    config = temp_config

    # Set the ramp voltage and get the mean/std of all channels
    device.set_ramp_voltage(ramp_voltage=config["offset_voltage"])
    data = collect_continuous_data_current_setup(
        device, 0, config["collection_time"], ["mean", "std"], calibrated=False
    )

    # Cleanup
    device.set_ramp_voltage(0)
    device.set_all_channel_inputs_to_disconnected()

    return data.reset_index().drop(["well"], "columns")


def assess_data(
    device: PromethionGrpcClient, data: pd.DataFrame, config: Optional[dict] = None
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

    data["range"] = get_promethion_range(device)

    # Calculate the actual offsets which is just the inverse of the offset current levels
    data["offset"] = np.rint(-1 * data["mean"])

    # Check which stds and offsets pass
    data["offset_pass"] = data["offset"].between(config["hard_min"], config["hard_max"])
    data["sd_pass"] = data["std"] <= config["sd_max"]

    # Work out how many passed
    data["pass"] = data["offset_pass"] & data["sd_pass"]
    percent_passed = data["pass"].sum() / len(data.index)

    return data, bool(percent_passed >= config["pass_percent"])


def get_promethion_range(device: PromethionGrpcClient) -> float:
    r"""
    Formula:
                    (Vref)      IntegrationCapactiance   /ADCoutput       1.25V*ADCgain\
    TO_PICOS * -------------- * ---------------------- *(---------- - 1 + ------------- )
               ADCgain * gain       IntegrationTime      \ADCRange            Vref     /

    Sample:
                    1.49             100fF      / 2000       1.25*(1/0.92)\
    TO_PICOS * ------------ * --------------- *(  ---- - 1 + ------------- )
               (1/0.92) * 2    71.9 Microsecs   \ 2048            1.49    /

    The 2 values minknow takes is the range and the offset and performs this calculation:
         (range/digitisation)*(ADCoutput+offset) to get the current.

    The offset is simply the negation of the average

    The range has to be the components outside of the brackets. (* by picos to get range)
    This means we are losing slight accuracy of ( scalar * ( -1 + (1.25*ADCGain/Vref) ) ),
    however we cannot adjust for this in the offset or then we will skew the zero adjustment.
    """

    sample_rate = device.get_sample_rate()
    gain, int_cap = device.get_gain_and_int_capacitor()

    integration_time = (1.0 / sample_rate) - FRAME_TIME_DIFF
    int_cap_actual = int_cap * FF_CONVERT

    scaling = (V_REF / (gain * ADC_GAIN)) * (int_cap_actual / integration_time)
    range_pa = scaling * PICOAMP_CONVERT
    return range_pa


def generate_ping(
    device: BaseDeviceInterface, calibration_information_list: list[tuple[pd.DataFrame, bool]], purpose: str
) -> dict[str, Any]:
    """Given calibration metrics/assessments/success per calibration attempt,
    Generate a ping with the stats including per pixel block

    Calibration_information_list is a list of:
       (metricsDataFrame, assessmentDataFrame, overallResult)
    for each calibraiton attempt

    :param device: MinKNOW Device wrapper
    :param calibration_information_list: list of calibration attempts
    :param purpose: Reason for calibration

    :returns: Ping of stats
    :rtype: dict

    """
    ping = {
        "calibration_results": {},
        "calibration_pixel_block_results": {},
        "calibration_summary_data": {},
        "calibration_purpose": purpose,
    }

    for (attempt, (assessment, result)) in enumerate(calibration_information_list, start=1):
        calibration_pixel_block_results = []
        calibration_summary_data = []

        for pixel_block in range(1, 13):
            step = device.channel_count / 12
            start_channel = (pixel_block - 1) * step
            end_channel = pixel_block * step
            channels_assessment = assessment[
                (assessment.channel >= start_channel) & (assessment.channel <= end_channel)
            ]

            # Say whether they are all calibrated or not
            calibration_pixel_block_results.append(bool(channels_assessment["pass"].all()))

            # Record mean/sd of the pixel block
            calibration_summary_data.append(
                {
                    "pb": pixel_block,
                    "means": channels_assessment["mean"].quantile(QUANTILES).tolist(),
                    "sds": channels_assessment["std"].quantile(QUANTILES).tolist(),
                }
            )

        ping["calibration_summary_data"]["calibration_{}".format(attempt)] = calibration_summary_data
        ping["calibration_pixel_block_results"]["calibration_{}".format(attempt)] = calibration_pixel_block_results
        ping["calibration_results"]["calibration_{}_pass".format(attempt)] = result

    ping["calibration_summary_data"]["quantiles"] = QUANTILES

    passes = [cal_info[-1] for cal_info in calibration_information_list]
    ping["calibration_results"]["calibration_pass"] = any(passes)

    return ping
