from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from bream4.toolkit.pings.ping_generation import translate_keys

try:
    from production.sawtooth_utils import add_sawtooth_session_info
except ImportError:
    pass


from typing import Any, Optional

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface

CHANNEL_MUX_DECODE_DICT = {
    "waveform_mean": "60mv_triangle_waveform_mean",
    "waveform_min": "60mv_triangle_waveform_min",
    "waveform_max": "60mv_triangle_waveform_max",
    "waveform_sd_cap": "60mv_triangle_waveform_sd_cap",
    "pore_classification": "classification",
    "tba_count": "tba_event_count",
    "tba_samples": "tba_event_samples",
    "tba_statistics_median": "tba_event_statistics_median",
    "tba_statistics_mean": "tba_event_statistics_mean",
    "tba_statistics_std": "tba_event_statistics_std",
    "tba_chunk_std_median": "tba_event_chunk_std_median",
    "well": "mux",
}

SUMMARY_DECODE_DICT = {
    "small": "small_membranes",
    "good_membrane": "good_membranes",
    "large": "large_membranes",
    "extra_large": "extra_large_membranes",
    "leaky": "leaky_membranes",
    "oil_seal": "oil_seals",
}


def generate_summary(
    device: BaseDeviceInterface, data: pd.DataFrame, config_assessment: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Generates a top level summary to include in the ping
    Includes:
      * multiplexing_group_x
      * General pore/tba stats if present
      * Counts for all assessment types (Config passed in as param)
      * total_failed_calibration

    :param device: MinKNOW device wrapper
    :param data: Pandas DF of channel mux data
    :param config_assessment: Dict of assessments -> {assessment_name: dict_of_details}
    :returns: Summary dict
    :rtype: dict

    """
    result = {}

    if "pore_classification" in config_assessment:
        # If we have classification, then we probably have pore information

        # Add how many good singles per mux -> 1 = at least 1 good well 2 = at least 2 good wells
        group = data[(data.pore_classification == "good_single")].groupby("channel").size()
        for well in device.get_well_list():
            result["multiplexing_group_" + str(well)] = group[group >= well].size

        # Only aggregate statistics for good_singles
        gs_data = data[(data.pore_classification == "good_single")]

        if not gs_data.empty:
            # If we have some pore data, add pore stats
            if "pore_statistics_median" in gs_data and not gs_data["pore_statistics_median"].isnull().all():
                result["single_open_pore_mean"] = gs_data["pore_statistics_median"].mean()
                result["single_open_pore_std"] = gs_data["pore_statistics_median"].std()

                if "single_open_pore_std" in result and pd.isnull(result["single_open_pore_std"]):
                    # Pandas can generate nan for std if only 1 datapoint is entered
                    result["single_open_pore_std"] = 0

            # If we have tba data, add tba stats
            if "tba_statistics_median" in gs_data and not gs_data["tba_statistics_median"].isnull().all():
                result["tba_event_mean"] = gs_data["tba_statistics_median"].mean()

    # Add counts for all assessment types
    for assessment, config_assessments in config_assessment.items():
        # Each one should be of the form:
        # {"x": ..., "y":.... "default":.... }

        for count_item in config_assessments.keys():
            if count_item not in {"default", "apply_order"}:
                result[count_item] = data[(data[assessment] == count_item)].shape[0]

    result["total_failed_calibration"] = device.channel_count - sum(
        device.get_calibration_status_all_channels().values()
    )

    # Make sure to rename any keys
    return translate_keys(result, SUMMARY_DECODE_DICT)


def generate_channelwise_ping(data: pd.DataFrame, drop_from_pings: Optional[list[str]] = None) -> list[dict[str, Any]]:
    renamed = data.reset_index().rename(CHANNEL_MUX_DECODE_DICT, axis=1)

    if drop_from_pings:
        # Remove matching regex columns
        to_drop = np.any([renamed.columns.str.match(x) for x in drop_from_pings], axis=0)
        renamed = renamed.loc[:, ~to_drop]

    channelwise = []

    records: list[dict[str, any]] = renamed.to_dict("records")  # type: ignore
    for record in records:
        # Make sure to drop any nans
        channel = {k: v for k, v in record.items() if pd.notnull(v)}

        # If membrane data present, some have to live in a subdocument. Go figure.
        new_section = {}
        for subsection, full_section in zip(
            ["meas_cap", "meas_res", "prob_tri", "prob_sq"],
            ["waveform_meas_cap", "waveform_meas_res", "waveform_prob_tri", "waveform_prob_sq"],
        ):
            if full_section in channel:
                new_section[subsection] = channel[full_section]
                del channel[full_section]

        if new_section:
            channel["60mv_triangle_waveform"] = new_section

        channelwise.append(channel)

    return channelwise


def generate_ping(
    device: BaseDeviceInterface,
    data: pd.DataFrame,
    config: dict,
    sawtooth_info: Optional[dict[str, Any]] = None,
    calculated_voltage_offset: Optional[float] = None,
) -> dict[str, Any]:
    summary = generate_summary(device, data, config.get("assessment", {}))
    channelwise = generate_channelwise_ping(data, config.get("drop_from_pings", []))

    result = {}

    # Add the classification rules
    rules = device.get_analysis_configuration()["read_classification"]["parameters"]["rules_in_execution_order"]
    result["config"] = {
        "classification_rules_hash": hashlib.sha1(str(rules).encode("utf-8")).hexdigest(),
        "applied_voltage_offset_adjustment_mv": device.get_bias_voltage_offset(),
    }
    if calculated_voltage_offset is not None:
        result["config"]["calculated_voltage_offset_adjustment_mv"] = calculated_voltage_offset

    purpose = device.get_exp_script_purpose()
    if purpose == "platform_qc":
        result["pores_for_state"] = summary
        result["platform_channel_metrics"] = channelwise
    elif purpose == "membrane_qc":
        result["membrane_summary"] = summary
        result["membrane_channel_metrics"] = channelwise
    else:
        result["summary"] = summary
        result["channel_metrics"] = channelwise

    # Add any necessary sawtooth info
    if sawtooth_info:
        add_sawtooth_session_info(device, result, sawtooth_info)

    return result
