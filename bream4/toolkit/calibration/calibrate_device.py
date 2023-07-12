from __future__ import annotations

import logging
from typing import Any, Optional

import bream4.toolkit.calibration.minion_like as minion_calibration
import bream4.toolkit.calibration.promethion as promethion_calibration
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.device_interfaces.devices.minion import MinionGrpcClient
from bream4.device_interfaces.devices.promethion import PromethionGrpcClient


def _get_state(device: BaseDeviceInterface) -> dict[str, Any]:
    # Things we need to change in the calibration script.
    # Save them so we can restore them later
    state = {
        "exp_script_purpose": device.get_exp_script_purpose(),
        "writer_configuration": device.get_writer_configuration(),
    }

    if isinstance(device, PromethionGrpcClient):
        state["overload_mode"] = device.get_overload_mode()

    return state


def _apply_state(device: BaseDeviceInterface, state: dict[str, Any]) -> None:
    # Apply the state saved from _get_state
    device.set_exp_script_purpose(state["exp_script_purpose"])
    device.set_writer_configuration(state["writer_configuration"])

    if isinstance(device, PromethionGrpcClient):
        device.set_overload_mode(state["overload_mode"])


def calibrate(
    device: BaseDeviceInterface,
    output_bulk: bool = False,
    ping_data: bool = True,
    purpose: str = "calibration",
    config: Optional[dict] = None,
) -> None:
    """
    This will collect various bits of data from the device at different settings
    and set the offsets/scale on the device

    :param device: MinKNOW device wrapper
    :param output_bulk: boolean whether to output bulk file for calibration
    :param ping_data: boolean whether to ping calibration results
    :param purpose: Why this calibration is taking place (Platform QC/Sequencing/..)
    :param config: Any extra configuration parameters to pass

    :raises: RuntimeError if calibration was not successful

    """
    logger = logging.getLogger(__name__)

    # Save any settings we will clobber during calibration
    old_state = _get_state(device)

    ######################################
    # configure data output  as required #
    ######################################

    device.set_writer_configuration({})
    file_output = False

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
        # For calibration, MinKNOW disables file output by default. So force bulks
        file_output = True

    # set experiment script purpose
    device.set_exp_script_purpose("calibration")

    ##################
    # Main procedure #
    ##################

    calibration_information = []

    calibration_attempts = 1

    # If on promethion, overload_mode needs to be 'set_flag'
    # so that saturation doesn't trigger mux 0
    # Also retry calibration on promethion due to command errors
    if isinstance(device, PromethionGrpcClient):
        device.set_overload_mode("set_flag")
        calibration_attempts = 3

    # acquisition should not already be running
    if device.get_acquisition_status() == device.ACQ_PROCESSING:
        raise RuntimeError("Data acquisition should not already be running.")

    for attempt in range(calibration_attempts):
        device.start_acquisition(purpose="calibration", file_output=file_output)

        if isinstance(device, MinionGrpcClient):
            data = minion_calibration.collect_data(device, config=config)
            assessment, cal_pass = minion_calibration.assess_data(device, data, config=config)
        elif isinstance(device, PromethionGrpcClient):
            data = promethion_calibration.collect_data(device, config=config)
            assessment, cal_pass = promethion_calibration.assess_data(device, data, config=config)
        else:
            raise RuntimeError("Invalid device type")

        calibration_information.append((assessment, cal_pass))
        keep_power_on = False
        # if any of the calibrations passed then keep the power on but if all fail turn power to asic off
        if any([cal_result[1] for cal_result in calibration_information]):
            keep_power_on = True
        device.stop_acquisition(keep_power_on=keep_power_on)

    # ------------------------------

    # Find the most recent good calibration
    successful_calibration = False

    for index in reversed(range(calibration_attempts)):
        (assessment, cal_pass) = calibration_information[index]

        if cal_pass:

            logger.info("Using calibration {}/{}".format(index + 1, calibration_attempts))

            # Make sure its in channel order
            assessment.sort_values(["channel"], axis=0, inplace=True)

            # Save to MinKNOW
            device.set_calibration(assessment["range"].tolist(), assessment["offset"].tolist())

            # Save channels that passed/failed
            pass_fail_list = assessment["pass"].tolist()
            device.set_channel_calibration_status(pass_fail_list)

            successful_calibration = True
            break

    if ping_data and device.is_promethion:
        logger.info("Pinging calibration data")
        ping = promethion_calibration.generate_ping(device, calibration_information, purpose)
        device.send_ping_data(ping)

    # Undo state that was changed because of calibration
    _apply_state(device, old_state)

    if not successful_calibration:
        logger.error("calibration_failed")
        raise RuntimeError("Calibration failed; not enough channels passed")
