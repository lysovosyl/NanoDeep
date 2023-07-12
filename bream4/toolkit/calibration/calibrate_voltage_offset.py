from __future__ import annotations

import copy
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from numpy.linalg import LinAlgError

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.data_extraction.collect_continuous_data import (
    collect_continuous_data_current_setup,
)
from bream4.toolkit.procedure_components.voltage_operations import calculate_nearest_bias_voltage

DEFAULT_STEPS = [-5, -10, -15]
DEFAULT_MUX = [2]  # What mux to use if called without argument
DEFAULT_PERIOD = 6
DEFAULT_MINIMUM_CONDUCTANCE = 0.5
DEFAULT_MAXIMUM_CONDUCTANCE = 3  # Limit to exclude extremely conductive channels
DEFAULT_MAXIMUM_CONFIDENCE = 50
DEFAULT_MINIMUM_CONFIDENCE = -50
THRESHOLD_PICOAMPS = 1.0  # Minimum absolute current to use for offset calibration

DEFAULT_SETTINGS = {
    "voltage_steps": DEFAULT_STEPS,
    "muxes": DEFAULT_MUX,
    "collection_time": DEFAULT_PERIOD,
    "threshold_pa": THRESHOLD_PICOAMPS,
    "minimum_conductance": DEFAULT_MINIMUM_CONDUCTANCE,
    "maximum_conductance": DEFAULT_MAXIMUM_CONDUCTANCE,
    "maximum_confidence": DEFAULT_MAXIMUM_CONFIDENCE,
    "minimum_confidence": DEFAULT_MINIMUM_CONFIDENCE,
    "maximum_adjustment": 5,
    "min_wells": 100,
}


def _get_state(device: BaseDeviceInterface) -> dict[str, Any]:
    # Things we need to change in the calibration script.
    # Save them so we can restore them later
    return {
        "exp_script_purpose": device.get_exp_script_purpose(),
        "writer_configuration": device.get_writer_configuration(),
    }


def _apply_state(device: BaseDeviceInterface, state: dict[str, Any]) -> None:
    # Apply the state saved from _get_state
    device.set_exp_script_purpose(state["exp_script_purpose"])
    device.set_writer_configuration(state["writer_configuration"])


def calibrate_voltage_offset(
    device: BaseDeviceInterface,
    output_bulk: bool = False,
    config: Optional[dict] = None,
    control_acquisition: bool = True,
    default_offset: float = 0.0,
) -> tuple[float, float, pd.DataFrame]:
    """
    Attempt to calibrate the device's bias_voltage_offset property by collecting data at voltage steps and regressing
    them against the measured current levels.

    1. Raw data collection & aggregation
    2. Filtering out of channels that do not show any useful conductance
    3. Regression of (voltage) against (median current level of conductive channels) to find Y-intercept. This is
    the estimated voltage offset of the system
    4. Add this offset to the current bias_voltage_offset that was used to collect the data

    :param device: Device interface
    :param output_bulk: Whether to write a bulk file for the raw data collection period
    :param config: Parameters in `DEFAULT_SETTINGS` to override
    :param control_acquisition: A flag that indicates if this is run as part of a script or stand alone requiring
    data acquisition to be started.
    :param default_offset: If the code fails to calculate an offset this is the value it will return.
    :returns: tuple (new_offset_calculated, new_offset_actual, data) where new_offset_actual is the final adjustment
    with the cap maximum_adjustment applied, and new_offset_calculated is the value before applying the cap and data is
    a pandas Dataframe containing the intercept values collected during the calibration
    """
    logger = logging.getLogger(__name__)

    current_offset = device.get_bias_voltage_offset()

    settings = copy.deepcopy(DEFAULT_SETTINGS)

    if config is not None:
        settings.update(config)
        logger.info("Updating settings: {}".format(settings))

    # Save any settings we will clobber during calibration
    old_state = _get_state(device)
    if control_acquisition:
        if output_bulk:
            device.set_writer_configuration(
                {
                    "bulk": {
                        "raw": {"all_channels": True},
                        "device_metadata": True,
                        "device_commands": True,
                        "file_pattern": old_state["writer_configuration"]["bulk"]["file_pattern"],
                    }
                }
            )

        device.set_exp_script_purpose("voltage_calibration")
        device.start_acquisition(file_output=output_bulk)
    if output_bulk and not control_acquisition:
        logger.warning(
            "Bulk file is set to be configured but acquisition is already on, bulk configuration changes "
            "will not take effect while acquisition is active"
        )
    data = collect_voltage_data(device, settings["muxes"], settings["voltage_steps"], settings["collection_time"])

    if control_acquisition:
        device.stop_acquisition()

    data.reset_index(inplace=True)
    # This is the offset of the device
    calculated_offset, data = estimate_offset(
        data,
        minimum_conductance=settings["minimum_conductance"],
        maximum_conductance=settings["maximum_conductance"],
        maximum_confidence=settings["maximum_confidence"],
        minimum_confidence=settings["minimum_confidence"],
        min_wells=settings["min_wells"],
        default_offset=default_offset,
    )

    # Check that the calculated offset is not above the maximum allowed adjustment
    offset_to_apply = limit_voltage_offset(calculated_offset, settings["maximum_adjustment"])
    offset_to_apply = calculate_nearest_bias_voltage(device.get_minimum_voltage_adjustment(), offset_to_apply)

    logger.info("Applying estimated offset value of {}".format(offset_to_apply))

    # Subtract the offset of the device -> new bias voltage adjustment
    new_offset_calculated = current_offset + calculated_offset
    new_offset_actual = current_offset + offset_to_apply

    logger.info("Changing bias voltage offset from {} to {}".format(current_offset, new_offset_actual))

    device.set_bias_voltage_offset(new_offset_actual)

    # Undo state that was changed because of calibration
    if control_acquisition:
        _apply_state(device, old_state)
    logger.info("new_offset_calculated = {} new_offset_actual = {}".format(new_offset_calculated, new_offset_actual))
    return new_offset_calculated, new_offset_actual, data


def collect_voltage_data(
    device: BaseDeviceInterface, muxes: list[int], steps: list[float], collection_period: float
) -> pd.DataFrame:

    """
    Collect current data at voltage steps in long format for regression calculation.
    Will return df of length channels*muxes*voltage steps.

    :return: `pd.DataFrame` with columns [ channel | mux | voltage | mean ]
    """

    logger = logging.getLogger(__name__)

    data = pd.DataFrame()

    for mux in muxes:

        device.set_all_channels_to_well(mux)

        for voltage_step in steps:
            logger.info("Collecting calibration data from mux {} at {} mV".format(mux, voltage_step))
            device.set_bias_voltage(voltage_step)

            # | channel | mux | mean |
            step_data = collect_continuous_data_current_setup(
                device,
                well=mux,
                collection_period=collection_period,
                aggregations=["mean"],
                calibrated=True,
            )
            step_data.loc[:, "voltage"] = voltage_step

            data = data.append(step_data)

        device.set_bias_voltage(0)
    device.set_all_channels_to_well(0)

    return data


def _analyse_current_voltage_trend(
    data: pd.DataFrame,
    minimum_conductance: float = 0.5,
    maximum_conductance: float = 3,
    maximum_confidence: float = 50,
    minimum_confidence: float = -50,
) -> pd.DataFrame:
    """
    :param data: Indexed on [channel, well], with columns mean/voltage
    :param minimum_conductance: lower bound to filter on for the conductance of the well
    :param maximum_conductance: upper bound to filter on for the conductance of the well
    :param maximum_confidence: upper bound to filter on for the straightness of the intercept line
    :param minimum_confidence: lower bound to filter on for the straightness of the intercept line
    :return: analysed_dataframe (conductance, intercept, collinear_confidence, voltages mean per chanmux)
    """

    def confidence(voltage, mean):

        return (
            (voltage[0] * (mean[1] - mean[2])) + (voltage[1] * (mean[2] - mean[0])) + (voltage[2] * (mean[0] - mean[1]))
        )

    def polyfit_or_0(x, y, deg=1, val=1):
        try:
            return np.polyfit(x, y, deg=deg)[val]
        except LinAlgError:
            return 0
        except ValueError:
            return 0

    # One row per channel, well -> one column per voltage, values are measured current
    pivoted_data = data.fillna(0).pivot_table(columns="voltage", values="mean", index=["channel", "well"])

    # Matches the order of the columns
    voltage_steps = pivoted_data.columns.tolist()
    voltages = pivoted_data[voltage_steps]

    # Calculate the conductance etc for each well
    pivoted_data["conductances_ns"] = voltages.apply(
        lambda row: np.polyfit(voltage_steps, row.values, deg=1)[0], axis=1
    ).abs()
    pivoted_data["intercept_ns"] = voltages.apply(lambda row: polyfit_or_0(row.values, voltage_steps), axis=1)
    pivoted_data["collinear_confidence"] = voltages.apply(lambda row: confidence(voltage_steps, row.values), axis=1)
    pivoted_data = pivoted_data.round(3)

    pivoted_data["conductance_filtered_data"] = pivoted_data["conductances_ns"].between(
        minimum_conductance, maximum_conductance
    ) & pivoted_data["collinear_confidence"].between(minimum_confidence, maximum_confidence)

    return pivoted_data


def estimate_offset(
    data: pd.DataFrame,
    minimum_conductance: float = 0.5,
    maximum_conductance: float = 3,
    maximum_confidence: float = 2,
    minimum_confidence: float = -2,
    min_wells: int = 100,
    default_offset: float = 0.0,
) -> tuple[float, pd.DataFrame]:
    """
    1. Filtering out of channels that do not show any useful conductance
    2. Regression of (voltage) against (median current level of conductive channels) to find Y-intercept. This is
    the estimated voltage offset of the system

    :param data: Data returned from `collect_voltage_data` - of length channel*mux*voltages
    :param default_offset: If the code fails to calculate an offset this is the value it will return.
    :param minimum_conductance: The minimum threshold that the conductance can be for the well to be included in the
    intercept calculation
    :param maximum_conductance:he maximum threshold that the conductance can be for the well to be included in the
    intercept calculation
    :param maximum_confidence: The maximum threshold that the collinear_confidence can be for the well to be included in
     the intercept calculation
    :param minimum_confidence: The minimum threshold that the collinear_confidence can be for the well to be included in
     the intercept calculation
    :param min_wells: The minimum number of wells that can be used after filtering to calculate a global intercept
    :return: Estimated offset(mV), analysed_data
    """
    logger = logging.getLogger(__name__)

    logger.info(
        f"Estimating offset channels with conductance between {minimum_conductance} and {maximum_conductance} nS"
    )

    analysed_data = _analyse_current_voltage_trend(
        data,
        minimum_conductance,
        maximum_conductance,
        maximum_confidence,
        minimum_confidence,
    )

    good_analysed_data = analysed_data[analysed_data["conductance_filtered_data"]]
    if good_analysed_data.shape[0] < min_wells:
        logger.warning(
            "Not enough usable data to perform voltage calibration on this flow cell."
            "Using default bias voltage offset."
        )
        return default_offset, analysed_data

    voltages = data.voltage.unique()
    median_mean_current = [good_analysed_data[voltage].median() for voltage in voltages]

    grad, intercept = np.polyfit(  # deg=1 -> straight line fit, returns (gradient, intercept)
        median_mean_current, voltages, deg=1
    )

    logger.info(
        "Regression on voltages: {} currents:{} gave result \n{}".format(voltages, median_mean_current, intercept)
    )

    return round(intercept, 1), analysed_data


def limit_voltage_offset(calculated_offset: float, maximum_adjustment: float) -> float:

    logger = logging.getLogger(__name__)

    if abs(calculated_offset) > maximum_adjustment:
        sign = abs(calculated_offset) / calculated_offset
        new_offset = sign * maximum_adjustment
        logger.info(
            "Calculated offset was {} mV but maximum adjustment set to {} mV. "
            "Adjusting voltage offset by {} mV".format(calculated_offset, maximum_adjustment, new_offset)
        )
        return new_offset
    else:
        return calculated_offset
