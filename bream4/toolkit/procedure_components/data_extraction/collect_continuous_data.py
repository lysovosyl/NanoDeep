from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.data_extraction.collect_raw_data import collect_raw_data

# Should all be functions that take 1 argument: A numpy array
AGGREGATION_FUNCTIONS = {
    "min": np.min,
    "max": np.max,
    "std": np.std,
    "mean": np.mean,
    "median": np.median,
}


def collect_continuous_data(
    device: BaseDeviceInterface,
    voltages: list[float],
    collection_periods: list[float],
    aggregations: list[str],
    muxes: Optional[list[int]] = None,
    clean_field_names: bool = False,
    round_dps: int = 3,
    on_set_mux: Optional[Callable[[], None]] = None,
    extra_aggregations: Optional[dict[str, Callable[[np.ndarray], Any]]] = None,
) -> pd.DataFrame:
    """Collects continuous data over collection times
    voltages = [5, 10]
    collection_periods = [1, 1]
    aggreagtions = [ 'max', 'min', 'std' ]

    Will return:
    DataFrame:
    channel, well, 5mv_max, 5mv_min, 5mv_std, 10mv_max, 10mv_min, ..

    If a well becomes disconnected/saturated these will be filled with nan

    :param device: MinKNOW Device wrapper
    :param voltages: list of voltages
    :param collection_periods: list of seconds pair wise with voltages
    :param aggregations: list of aggreagation functions
    :param muxes: list of muxes to perform on.  Default is to do all muxes available
    :param clean_field_names: Helpful for analysis; Flips the mv sign and changes `-` to `neg`
    :param round_dps: If not None, will round the values
    :param on_set_mux: If not None, the nanodeep will attempt to call on_set_mux after the mux
                       has been set
    :param extra_aggregations: dict of [name -> callable that takes a np array as an arg]
                               For examples {"first_sample": lambda x: x[0]}
    :returns: DataFrame with channel, well, stats
    :rtype: DataFrame

    """

    if len(voltages) != len(collection_periods):
        raise RuntimeError(
            f"Voltages {voltages} and collection periods {collection_periods}" + " expected to be the same length"
        )

    if len(set(voltages)) != len(voltages):
        raise RuntimeError(f"Voltages {voltages} contain duplicate items. Columns will be ambiguous")

    if not all([fn in AGGREGATION_FUNCTIONS for fn in aggregations]):
        remainder = [fn for fn in aggregations if fn not in AGGREGATION_FUNCTIONS]
        raise KeyError(f"Aggregations {remainder} are not in {AGGREGATION_FUNCTIONS}")

    device.set_bias_voltage(0)

    # Create a blank df to store accumulated results
    accumulated = pd.DataFrame()

    if muxes is None:
        muxes = device.get_well_list()

    for well in muxes:

        device.set_all_channels_to_well(well)

        if on_set_mux:
            on_set_mux()

        # Add each collection period to this
        well_collection = pd.DataFrame()

        for voltage, collection_period in zip(voltages, collection_periods):
            device.set_bias_voltage(voltage)

            df = collect_continuous_data_current_setup(
                device,
                well,
                collection_period,
                aggregations,
                round_dps=round_dps,
                extra_aggregations=extra_aggregations,
            )
            df = df[aggregations]

            # Rename to the offset_5mv_max mappings etc
            new_col_name = f"offset_{voltage}mv"
            if clean_field_names:
                new_col_name = new_col_name.replace(str(voltage), str(voltage * -1)).replace("-", "neg")

            df.rename(
                columns={fn: f"{new_col_name}_{fn}" for fn in aggregations},
                inplace=True,
            )
            well_collection = pd.concat([well_collection, df], axis=1)

        device.set_bias_voltage(0)

        # Got all well data so append to big frame
        accumulated = pd.concat([accumulated, well_collection])

    # Pandas bug: Index changes from int to str
    accumulated.reset_index(inplace=True)
    accumulated["channel"] = accumulated["channel"].astype(int)
    accumulated["well"] = accumulated["well"].astype(int)

    return accumulated.set_index(["channel", "well"])


def collect_continuous_data_current_setup(
    device: BaseDeviceInterface,
    well: int,
    collection_period: float,
    aggregations: list[str],
    calibrated: bool = True,
    round_dps: int = 3,
    extra_aggregations: Optional[dict[str, Callable[[np.ndarray], Any]]] = None,
):
    raw_data = collect_raw_data(device=device, collection_time_sec=collection_period, calibrated=calibrated)
    disconnected = device.get_disconnection_status_for_active_wells()

    all_channels = []

    aggregation_lookups = AGGREGATION_FUNCTIONS.copy()
    if extra_aggregations:
        aggregations = aggregations + list(extra_aggregations)
        aggregation_lookups.update(extra_aggregations)

    for (channel, channel_data) in enumerate(raw_data, start=1):
        # Add data if we aren't disconnected or if we were told we were already in well 0
        if not disconnected[channel] or well == 0:
            arr = np.array(channel_data)
            values = [aggregation_lookups[fn](arr) for fn in aggregations]
        else:
            values = [np.nan] * (len(aggregations))

        all_channels.append([channel, well] + values)  # type: ignore (list[int] + list[Any])

    df = pd.DataFrame(all_channels, columns=["channel", "well"] + aggregations)
    df.set_index(["channel", "well"], inplace=True)

    # Saturated channels should still give 0 from the get_signal_bytes
    if len(df) != device.channel_count:
        raise RuntimeError(f"Only received response from {len(df)}/{device.channel_count} channels")

    if round_dps is not None:
        df = df.round(decimals=round_dps)

    return df
