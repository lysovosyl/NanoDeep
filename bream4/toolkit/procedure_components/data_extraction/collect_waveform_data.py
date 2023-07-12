from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Union

import numpy as np
import pandas as pd

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.analysis.metrics.waveform_utilities import waveform_triangle
from bream4.toolkit.procedure_components.data_extraction.collect_raw_data import collect_raw_data

AGGREGATION_FUNCTIONS = {
    "min": np.min,
    "max": np.max,
    "std": np.std,
    "mean": np.mean,
    "median": np.median,
}

WAVEFORM_FUNCTIONS = [
    "meas_res",
    "meas_cap",
    "prob_sq",
    "prob_tri",
    "triangle_ratio",
    "sd_cap",
]

PROMETHION_WAVEFORM_SETTINGS_DEFAULT = {
    "waveform_values": [
        0,
        -4,
        -8,
        -12,
        -16,
        -20,
        -24,
        -28,
        -32,
        -36,
        -40,
        -44,
        -48,
        -52,
        -56,
        -60,
        -64,
        -60,
        -56,
        -52,
        -48,
        -44,
        -40,
        -36,
        -32,
        -28,
        -24,
        -20,
        -16,
        -12,
        -8,
        -4,
    ],
    "frequency": 13.157894737,
}

MINION_WAVEFORM_SETTINGS_DEFAULT = {
    "waveform_values": [
        0,
        5,
        5,
        5,
        10,
        10,
        15,
        15,
        15,
        20,
        20,
        25,
        25,
        25,
        30,
        30,
        25,
        25,
        25,
        20,
        20,
        15,
        15,
        15,
        10,
        10,
        5,
        5,
        5,
        0,
        0,
        -5,
        -5,
        -5,
        -10,
        -10,
        -15,
        -15,
        -15,
        -20,
        -20,
        -25,
        -25,
        -25,
        -30,
        -30,
        -25,
        -25,
        -25,
        -20,
        -20,
        -15,
        -15,
        -15,
        -10,
        -10,
        -5,
        -5,
        -5,
        0,
    ]
}


def collect_waveform_data(
    device: BaseDeviceInterface,
    collection_period: float,
    aggregations: list[str],
    waveform: Optional[Sequence[Union[float, int]]] = None,
    frequency: Optional[float] = None,
    muxes: Optional[list[int]] = None,
    round_dps: int = 3,
) -> pd.DataFrame:
    """Collects data whilst generating a waveform and returns metrics(aggregations) of the data.

    Will setup a basic waveform if not specified.

    Aggregations allowed are:
       * 'min', 'max', 'std', 'mean'
       * 'meas_res' A measure of the membrane leakage (not accurate until we get supersampled waveforms)
       * 'meas_cap' A measure of the membrane capacitance
       * 'prob_sq' Coefficient of fit to a square response
       * 'prob_tri' Coefficient of fit to a triangle response
       * 'triangle_ratio' prob_tri / prob_sq - if this ratio < 3 then considered a square response
       * 'sd_cap' An estimate of the membrane capacitance using response amplitude, limited by device noise

    :param device: MinKNOW Device wrapper
    :param collection_period: How long to collect in each mux for
    :param aggregations: list of strings defining functions to apply
    :param waveform: list of voltages for waveform
    :param frequency: Frequency of which to apply waveform, if possible
    :param muxes: list of muxes to perform on. If None all valid muxes will be used
    :param round_dps: If not None, round values to this many dps

    :returns: DataFrame of channel, well, aggreagations
    :rtype: Pandas DataFrame

    """

    if not all([fn in AGGREGATION_FUNCTIONS or fn in WAVEFORM_FUNCTIONS for fn in aggregations]):
        remainder = [fn for fn in aggregations if not (fn in AGGREGATION_FUNCTIONS or fn in WAVEFORM_FUNCTIONS)]
        raise KeyError(f"Aggregations {remainder} are not in {AGGREGATION_FUNCTIONS} or {WAVEFORM_FUNCTIONS}")

    # Assign defaults to arguments:

    # Get a waveform if not present
    if waveform is None and device.is_minion_like:
        waveform = MINION_WAVEFORM_SETTINGS_DEFAULT["waveform_values"]
    elif waveform is None and device.is_promethion:
        waveform = PROMETHION_WAVEFORM_SETTINGS_DEFAULT["waveform_values"]

    if waveform is None:
        raise RuntimeError("No waveform provided and couldn't find matching default")

    if not frequency and device.is_promethion:
        frequency = PROMETHION_WAVEFORM_SETTINGS_DEFAULT["frequency"]

    muxes = muxes if muxes is not None else device.get_well_list()

    device.set_bias_voltage(0)

    # Create a blank df to store accumulated results
    accumulated = pd.DataFrame()

    # Begin collection over muxes
    for well in muxes:
        device.set_all_channels_to_well(well)

        device.start_waveform(waveform_values=waveform, frequency=frequency)

        df = collect_waveform_data_current_setup(device, well, collection_period, aggregations, round_dps=round_dps)
        # Rename to the 5mv_max mappings etc
        df.rename(
            index=str,
            columns={fn: "waveform_{}".format(fn) for fn in aggregations},
            inplace=True,
        )

        device.stop_waveform()
        device.set_bias_voltage(0)

        accumulated = pd.concat([accumulated, df])

    # Pandas bug: Index changes from int to str
    accumulated.reset_index(inplace=True)
    accumulated["channel"] = accumulated["channel"].astype(int)
    accumulated["well"] = accumulated["well"].astype(int)

    return accumulated.set_index(["channel", "well"])


def collect_waveform_data_current_setup(
    device: BaseDeviceInterface, well: int, collection_period: float, aggregations: list[str], round_dps: int = 3
) -> pd.DataFrame:
    """Collect waveform data in the current setup

    :param device: MinKNOW device wrapper
    :param well: Which well the channels should be in
    :param collection_period: How long to collect data for
    :param aggregations: list of nanodeep names: What to do with the collected raw data
    :param round_dps: If not None, round values to this many dps

    :returns: df of aggregations per channel
    :rtype: pd.DataFrame

    """
    raw_data = collect_raw_data(device=device, collection_time_sec=collection_period, calibrated=True)
    disconnected = device.get_disconnection_status_for_active_wells()
    sample_rate = device.get_sample_rate()

    all_channels = []
    for (channel, channel_data) in enumerate(raw_data, start=1):
        if not disconnected[channel] or well == 0:
            arr = np.array(channel_data)
            values = []
            waveform_metrics = waveform_triangle(sample_rate, channel_data, device)
            for fn in aggregations:
                if fn in AGGREGATION_FUNCTIONS:
                    values.append(AGGREGATION_FUNCTIONS[fn](arr))
                else:
                    values.append(waveform_metrics[fn])
        else:
            values = [np.nan for fn in aggregations]

        all_channels.append([channel, well] + values)  # type: ignore (list[int]+list[float])

    df = pd.DataFrame(all_channels, columns=["channel", "well"] + aggregations)
    df.set_index(["channel", "well"], inplace=True)

    # Saturated channels should still give 0 from the get_signal_bytes
    if len(df) != device.channel_count:
        raise RuntimeError(f"Only received response from {len(df)}/{device.channel_count} channels")

    if round_dps is not None:
        df = df.round(decimals=round_dps)

    return df
