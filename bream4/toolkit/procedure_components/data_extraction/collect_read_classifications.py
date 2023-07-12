from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import pandas as pd
from typing_extensions import Literal

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.data_extraction.read_statistics import get_channel_read_classifications
from bream4.toolkit.procedure_components.voltage_operations import apply_voltages_for_durations

DEFAULT_VOLTAGE = -180
DEFAULT_GLOBAL_FLICK = {
    "enabled": True,
    "voltages": [0, 120, 0],
    "adjustment_pause": [1.0, 3.0, 1.0],
}


# Expects a nanodeep that takes a minknow PerClassificationData object
AGGREGATION_FUNCTIONS = {
    "count": lambda c: int(c.seconds_duration / c.duration_statistics.mean),
    "samples": lambda c: int(c.samples_duration),
    "min": lambda c: c.current_statistics.min,
    "max": lambda c: c.current_statistics.max,
    "mean": lambda c: c.current_statistics.mean,
    "std": lambda c: c.current_statistics.s_d,
    "median": lambda c: c.current_statistics.median,
    "chunk_median": lambda c: c.chunk_statistics.median,
    "chunk_std_median": lambda c: c.chunk_statistics.median_sd,
    "chunk_range": lambda c: c.chunk_statistics.range,
}
AggregationFunctionOptions = Literal[
    "count", "samples", "min", "max", "mean", "std", "median", "chunk_median", "chunk_std_median", "chunk_range"
]

CHUNK_STATS = set(["chunk_median", "chunk_std_median", "chunk_range"])
CURRENT_STATS = set(["min", "max", "mean", "std", "median"])

# Used to map names to {classification}_statistics_{fn} as opposed to {classification}_{fn}
STATISTICS_FUNCTIONS = ["mean", "std", "median", "min", "max"]


def collect_read_classification_data(
    device: BaseDeviceInterface,
    classifications: list[str],
    collection_period: float,
    aggregation_stats: list[AggregationFunctionOptions],
    voltage: Optional[float] = None,
    global_flick_config: Optional[dict] = None,
    completed_read: bool = False,
    muxes: Optional[list[int]] = None,
    round_dps: int = 3,
) -> pd.DataFrame:
    """Returns a dataframe for every channel, well with aggregating over specified read classifications

    If aggregation_stats == ["std", "samples"] and classifications == ["pore", "zero"]
    Column examples will be:
        channel, well, saturated, pore_statistics_std, pore_samples, zero_statistics_std, zero_samples

    If classification is not present or becomes saturated, samples/counts will be 0. Stats will be NaNs

    :param device: MinKNOW device wrapper
    :param classifications: list of classifications to get stats for
    :param collection_period: How long to collect in each well for
    :param aggregation_stats: Which stats to accumulate for each classification
    :param voltage: What voltage to collect at
    :param global_flick_config: If present, and enabled, global flicks will happen before collection
    :param muxes: If present, determines which wells are used
    :param round_dps: If not None, will round the values to that many dps

    :returns: df of all stats/classifications/channels/wells requested

    :rtype: Pandas DataFrame

    """

    if not voltage:
        voltage = DEFAULT_VOLTAGE

    if not global_flick_config:
        global_flick_config = copy.deepcopy(DEFAULT_GLOBAL_FLICK)
    else:
        # Make sure all defaults are present
        updated_values = copy.deepcopy(DEFAULT_GLOBAL_FLICK)
        updated_values.update(global_flick_config)
        global_flick_config = updated_values

    if not muxes:
        muxes = device.get_well_list()

    device.set_bias_voltage(0)

    # Create a blank df to store accumulated results
    accumulated = pd.DataFrame()

    for well in muxes:
        device.set_all_channels_to_well(well)
        device.set_bias_voltage(voltage)

        if global_flick_config.get("enabled"):
            apply_voltages_for_durations(
                device,
                global_flick_config["voltages"],
                global_flick_config["adjustment_pause"],
            )

        df = collect_read_classification_data_current_setup(
            device,
            well,
            classifications,
            collection_period,
            aggregation_stats,
            round_dps=round_dps,
        )

        accumulated = pd.concat([accumulated, df])

        device.set_bias_voltage(0)

    accumulated.set_index(["channel", "well"], inplace=True)
    return accumulated


def _get_column_name(classification, stat):
    """
    For current stats aggregations, name the columns in line with previous QCs
    """

    if stat in STATISTICS_FUNCTIONS:
        return "{}_statistics_{}".format(classification, stat)
    else:
        return "{}_{}".format(classification, stat)


def collect_read_classification_data_current_setup(
    device: BaseDeviceInterface,
    well: int,
    classifications_to_collect: list[str],
    collection_period: float,
    aggregation_stats: list[AggregationFunctionOptions],
    completed_read: bool = False,
    round_dps: int = 3,
) -> pd.DataFrame:

    metrics = get_channel_read_classifications(
        device,
        collection_period,
        completed_read=completed_read,
        include_chunk_statistics=any([stat in CHUNK_STATS for stat in aggregation_stats]),
        include_current_statistics=any([stat in CURRENT_STATS for stat in aggregation_stats]),
    )

    data = {}
    for key, classification_data in metrics.items():
        channel, actual_well = key

        # If a channel becomes saturated during collection, we may get several items
        # Make sure to preserve any data given
        if actual_well == device.disconnected_well:
            if channel in data:
                data[channel]["saturated"] = True
            else:
                data[channel] = {
                    "channel": int(channel),
                    "well": int(well),
                    "saturated": True,
                }
            continue

        # Store each classification stats
        classification_stats = {}

        for classification in classifications_to_collect:
            if classification in classification_data:
                # If its present do the stats
                classification_stats[classification] = {
                    stat: AGGREGATION_FUNCTIONS[stat](classification_data[classification]) for stat in aggregation_stats
                }
            else:
                # If it's not present, put nans for stats
                classification_stats[classification] = {stat: np.nan for stat in aggregation_stats}

        # flatten dict  flattened["pore_std"].. flattened["tba_std"]...
        # So that a pandas df can be collected easily
        flattened = {}
        for classification, stats in classification_stats.items():
            flattened.update({_get_column_name(classification, stat): value for stat, value in stats.items()})

        # Add data always present
        flattened["channel"] = int(channel)
        flattened["well"] = int(well)
        # Preserve the saturated value if it was ever true
        flattened["saturated"] = data.get(channel, {}).get("saturated", False)
        data[channel] = flattened

    df = pd.DataFrame(data.values())

    if round_dps is not None:
        df = df.round(decimals=round_dps)

    # If all are saturated immediately, you can be missing the columns you asked for
    all_columns = []
    for classification in classifications_to_collect:
        all_columns.extend([_get_column_name(classification, stat) for stat in aggregation_stats])
    for column in set(all_columns).difference(df.columns):
        df[column] = np.NaN

    for column in df.columns:
        # Replace count/sample NaNs with 0
        if column.endswith("_count") or column.endswith("_samples"):
            df[column].fillna(0, inplace=True)

            # If any nans were present, the column would have been pushed to a float
            df[column] = df[column].astype(np.int64)

    return df
