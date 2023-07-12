from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import bream4.toolkit.procedure_components.command.keystore as keystore
import numpy as np
import pandas as pd
from bream4.device_interfaces import device_wrapper
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.calibration.calibrate_device import calibrate
from bream4.toolkit.calibration.calibrate_voltage_offset import calibrate_voltage_offset

# Grab the modular sections from bream
from bream4.toolkit.procedure_components.data_extraction.collect_active_unblock_data import collect_active_unblock_data
from bream4.toolkit.procedure_components.data_extraction.collect_continuous_data import collect_continuous_data
from bream4.toolkit.procedure_components.data_extraction.collect_read_classifications import (
    collect_read_classification_data,
)
from bream4.toolkit.procedure_components.data_extraction.collect_waveform_data import collect_waveform_data
from bream4.utility.simulation import setup_playback
from shared.modular_qc.modular_qc_ping import generate_ping
from utility.config_argparse import config_argument_parser
from utility.config_file_utils import find_path, recursive_merge

try:
    from production.sawtooth_utils import request_sawtooth_session_info, send_sawtooth_telemetry
except ImportError:
    pass


DEFAULT_CONFIG: dict[str, Any] = {"custom_settings": {"temperature": {"target": 37.0, "tolerance": 0.5}}}


def assessment(df: pd.DataFrame, column_name: str, assessment_config: dict[str, str]) -> None:
    """Create a column based on the config provided. These will be evaluated on the dataframe.
    The corresponding rule name will be used as the value for the column.

    Example assessment_config:
    [custom_settings.assessment.membrane_quality]
    apply_order = ["good", "small"]
    good        = "waveform_meas_cap >= 0.5 & waveform_meas_cap <= 6"
    small       = "waveform_meas_cap < 0.5"
    default     = "bad"

    So a column named membrane_quality should be created.
    The column value will be either good/small/bad depending on the df passed in.

    The rule string will be directly evaluated on the df.

    :param df: DF to apply to
    :param column_name: Column name to add
    :param rule_order: Which order to apply the rules in
    :param rules: Dict of rule -> pandas eval string

    """
    df[column_name] = assessment_config.get("default", np.nan)

    for next_rule in assessment_config["apply_order"]:
        df.loc[(df.eval(assessment_config[next_rule])), column_name] = next_rule


def main(device: Optional[BaseDeviceInterface] = None, config: Optional[dict] = None) -> None:
    """
    This is a modular quality check script

    Via the toml configuration system you can set up the QC to check various aspects

    The sections that are allowed in this script are:
      * continuous_data collects min/max/std/mean/median from a set of voltages
      * waveform applies a waveform and gets meas_cap/sd_cap/median
      * read_classifications get a set of classifications and get the count/samples/mean/median/std
      * active_unblock will check if flicks cause channels/muxes to get saturated

    The saturated_during_name(s) record if parts of the tests caused channels/muxes to become saturated.

    [custom_settings]
    section_order = ["continuous_data", "waveform"]  # Says what order the sections should go in

    [custom_settings.continuous_data]
    type = "continuous_data"
    enabled = true
    voltages = [0, -10, -180]
    collection_periods = [1, 1, 1]
    aggregation_stats = ["mean", "median", "std"]
    saturated_during_names = ["zero", "ten", "180"]
    """

    # Grab device and general setup
    if device is None:
        device = device_wrapper.create_grpc_client()

    if config is None:
        config = {}

    # Merge the config passed with the defaults
    custom_settings = copy.deepcopy(DEFAULT_CONFIG)
    recursive_merge(custom_settings, config)
    custom_settings = custom_settings["custom_settings"]

    logger = logging.getLogger(__name__)

    # Internal specific
    sawtooth_url = find_path("meta.protocol.sawtooth_url", config)
    sawtooth_info = None

    calculated_voltage_offset = None

    if sawtooth_url:
        sawtooth_info = request_sawtooth_session_info(device, sawtooth_url)
        if not sawtooth_info:
            logger.error("sawtooth_utils.no_session")
            raise RuntimeError("Invalid sawtooth session")

    device.set_temperature(**custom_settings["temperature"])

    if "simulation" in custom_settings:
        setup_playback(device, custom_settings["simulation"])
    else:
        calibrate(device, output_bulk=False, ping_data=True, purpose=device.get_exp_script_purpose())

        # If asked for voltage calibration(And not done a server lookup), then perform it
        vc = custom_settings.get("voltage_calibration")
        if vc and vc.get("enabled", True) and not find_path("meta.protocol.bias_voltage_offset_lookup_server", config):
            calculated_voltage_offset, _, data = calibrate_voltage_offset(device, config=vc)

    # Used to store all data retrieved
    accumulated = None

    # No way to easily estimate this. Nobody would really care if this was incorrect. Mainly for sequencing.
    device.set_estimated_run_time(60 + 100 * device.wells_per_channel)

    # Setup done. Start the modular qc
    device.start_acquisition()

    # Give the UI temperature information
    temperature = custom_settings["temperature"]["target"]
    tolerance = custom_settings["temperature"]["tolerance"]
    keystore.set_protocol_data(
        device,
        temp_min=temperature - tolerance,
        temp_max=temperature + tolerance,
    )

    # Loop through sections and collect metrics accordingly
    sections = custom_settings.get("section_order", [])
    for section in sections:
        section_settings = custom_settings[section]

        if section_settings.get("enabled", True):
            if section_settings["type"] == "continuous_data":
                logger.log_to_gui("modular_qc.collect_continuous")
                df = collect_continuous_data(
                    device,
                    section_settings["voltages"],
                    section_settings["collection_periods"],
                    section_settings["aggregation_stats"],
                    clean_field_names=True,
                )

                # Add offset_0mv_std, zero_voltage etc to the saturated info
                if "saturated_during_names" in section_settings:
                    any_stat = section_settings["aggregation_stats"][0]

                    # Go in reverse order so that first one saturated is the cause
                    for saturated_during, voltage in zip(
                        section_settings["saturated_during_names"][::-1], section_settings["voltages"][::-1]
                    ):

                        # Do something sensible with voltages: 10 -> neg10mv, -10 is 10mv
                        header = f"offset_{voltage*-1}mv_{any_stat}".replace("-", "neg")
                        saturated = df[header].isna()
                        df.loc[saturated, "applied_voltage"] = voltage
                        df.loc[saturated, "saturated_during"] = saturated_during

            elif section_settings["type"] == "waveform":
                logger.log_to_gui("modular_qc.collect_waveform")
                df = collect_waveform_data(
                    device,
                    section_settings["collection_period"],
                    section_settings["aggregation_stats"],
                    waveform=section_settings.get("waveform_values"),
                    frequency=section_settings.get("frequency"),
                )

                saturated_during = section_settings.get("saturated_during_name")
                if saturated_during:
                    if "waveform_triangle_ratio" in df.columns:
                        # Triangle_ratio is divide by another value, so if that's 0 this could be nan
                        # So exclude that from the saturated nan check
                        saturated = df.drop(["waveform_triangle_ratio"], axis=1).isnull().any(axis=1)
                    else:
                        saturated = df.isnull().any(axis=1)
                    df.loc[saturated, "applied_voltage"] = min(section_settings["waveform_values"])
                    df.loc[saturated, "saturated_during"] = saturated_during

            elif section_settings["type"] == "active_unblock":
                logger.log_to_gui("modular_qc.collect_unblock")
                df = collect_active_unblock_data(
                    device,
                    duration=section_settings.get("flick_duration"),
                    voltage=section_settings.get("sequencing_voltage"),
                )
            elif section_settings["type"] == "read_classifications":
                logger.log_to_gui("modular_qc.collect_read_classifications")
                df = collect_read_classification_data(
                    device,
                    section_settings["classifications"],
                    section_settings["collection_period"],
                    section_settings["aggregation_stats"],
                    voltage=section_settings["sequencing_voltage"],
                    global_flick_config=section_settings.get("global_flick"),
                )

                saturated_during = section_settings.get("saturated_during_name")
                if saturated_during:
                    df.loc[df["saturated"], "applied_voltage"] = section_settings["sequencing_voltage"]
                    df.loc[df["saturated"], "saturated_during"] = saturated_during

            else:
                raise RuntimeError("Section {} not a valid type".format(section))

            # Uniqueness of column names is determined by pandas:
            # _x suffixes are added to any column names that are the same
            if accumulated is None:
                accumulated = df
            else:
                accumulated = accumulated.merge(df, left_index=True, right_index=True, suffixes=(None, "_x"))
                # Make sure to combine applied_voltage/saturated_during as newer info can become available
                if "applied_voltage_x" in accumulated.columns:
                    accumulated["applied_voltage"] = accumulated.applied_voltage.combine_first(
                        accumulated.applied_voltage_x
                    )
                    accumulated["saturated_during"] = accumulated.saturated_during.combine_first(
                        accumulated.saturated_during_x
                    )
                    accumulated.drop(["applied_voltage_x", "saturated_during_x"], axis=1, inplace=True)

    assert isinstance(accumulated, pd.DataFrame)  # nosec Help type checker

    # Perform any assessments asked for
    logger.info("Columns available to make assessment on: {}".format(accumulated.columns))
    for column_name, assessment_config in custom_settings.get("assessment", {}).items():
        assessment(accumulated, column_name, assessment_config)

    # Send the ping
    ping = generate_ping(
        device,
        accumulated,
        config["custom_settings"],
        sawtooth_info=sawtooth_info,
        calculated_voltage_offset=calculated_voltage_offset,
    )

    device.send_ping_data(ping)

    # Make sure to mostly follow keep_power_on
    purpose = device.get_exp_script_purpose()
    if purpose == "platform_qc" or not custom_settings.get("keep_power_on"):
        device.stop_acquisition(keep_power_on=False)
    else:
        device.stop_acquisition(keep_power_on=True)

    # Do config specific things
    if purpose == "platform_qc":
        # Report the information to the UI etc
        good_singles = ping["pores_for_state"]["good_single"]

        logger.log_to_gui(
            "platform_qc.report", params={"flow_cell_id": device.get_flow_cell_id(), "num_pores": good_singles}
        )

        if "num_pores_warn" in custom_settings:
            # Send results to keystore so UI can be populated
            passed = good_singles > custom_settings["num_pores_warn"]
            keystore.set_platform_qc_result(device, passed, good_singles)

        if sawtooth_info and sawtooth_url:
            send_sawtooth_telemetry(
                device=device, url=sawtooth_url, session_info=sawtooth_info, good_single=good_singles
            )

    elif purpose == "membrane_qc":
        logger.log_to_gui("membrane_qc.summary", params={"summary": str(ping["membrane_summary"])})

        if sawtooth_info and sawtooth_url and custom_settings.get("membrane_screening"):
            send_sawtooth_telemetry(device=device, url=sawtooth_url, session_info=sawtooth_info)


if __name__ == "__main__":
    parser = config_argument_parser()
    args = parser.parse_args()
    main(config=args.config)
