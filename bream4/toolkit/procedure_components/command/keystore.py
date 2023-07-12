import logging
import time
import warnings
from typing import Union

import minknow_api.manager
from grpc import RpcError
from minknow_api.keystore_pb2 import PERSIST_ACROSS_RESTARTS
from minknow_api.keystore_service import WatchResponse
from typing_extensions import Literal

import bream4.pb.bream.hardware_check_result_pb2 as hardware_check
import bream4.pb.bream.platform_qc_results_pb2 as platform_qc_results
import bream4.pb.bream.protocol_communication_pb2 as protocol_communication
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface

logger = logging.getLogger(__name__)

KEYSTORE_LOOKUP = {
    "bream.HARDWARE_CHECK_RESULT": hardware_check.HardwareCheckResult,
    "bream.protocol_communication": protocol_communication.ProtocolDataResults,
    "bream.PLATFORM_QC_RESULTS": platform_qc_results.PlatformQcResultsWithRunIds,
}
ValidKeystoreEntry = Union[
    hardware_check.HardwareCheckResult,
    protocol_communication.ProtocolDataResults,
    platform_qc_results.PlatformQcResultsWithRunIds,
]


def parse_keystore_message(path: str, to_parse: WatchResponse) -> ValidKeystoreEntry:
    msg = KEYSTORE_LOOKUP[path]()
    to_parse.values[path].Unpack(msg)
    return msg


def get_keystore_message(device: BaseDeviceInterface, path: str, manager_level: bool = False) -> ValidKeystoreEntry:
    """Grab message from path in keystore using an internal lookup. Unpack to protobuf

    :param device: MinKNOW device wrapper
    :param path: Str of path to retrieve from keystore
    :param manager_level: Whether it is stored by manager or control server
    """

    new_value = KEYSTORE_LOOKUP[path]()

    try:
        if manager_level:
            instance_ks = minknow_api.manager.Manager(port=device.provided_manager_port).keystore()
            old_value = instance_ks.get_one(name=path)
            old_value.value.Unpack(new_value)
            return new_value
        else:
            old_value = device.connection.keystore.get_one(name=path)
            old_value.value.Unpack(new_value)
            return new_value

    except RpcError:
        # Not yet in the keystore
        return KEYSTORE_LOOKUP[path]()


def set_hw_check_result(device: BaseDeviceInterface, passed: bool) -> None:
    result = hardware_check.HardwareCheckResult(passed=passed)
    device.connection.keystore.store(
        values={"bream.HARDWARE_CHECK_RESULT": result},
        lifetime=PERSIST_ACROSS_RESTARTS,
    )


def set_platform_qc_result(device: BaseDeviceInterface, passed: bool, total_pore_count: int) -> None:
    """Store the platform QC result in the keystore, to allow UI access to result.

    :param device: MinKNOW Device wrapper
    :param passed: Whether the check passed or not
    :param total_pore_count: How many pores present
    """

    warnings.warn(
        "Keystore no longer for PQC results. See device.set_platform_qc_result",
        DeprecationWarning,
    )

    # Get flow cell
    flowcell_id = device.get_flow_cell_id()
    if not flowcell_id:
        flowcell_id = "UNKNOWN"

    # Get Run ID
    try:
        run_id = device.get_protocol_run_id()
    except RpcError:
        run_id = "UNKNOWN"

    # Now grab old results
    results = get_keystore_message(device, "bream.PLATFORM_QC_RESULTS", manager_level=True)
    assert isinstance(results, platform_qc_results.PlatformQcResultsWithRunIds)  # nosec B101 Type checker

    stored = results.results[flowcell_id]

    # Create new result and append
    new_result = platform_qc_results.ResultWithRunId(passed=passed, total_pore_count=total_pore_count, run_id=run_id)
    stored.result.append(new_result)

    # Update with new value
    keystore = minknow_api.manager.Manager(port=device.provided_manager_port).keystore()
    keystore.store(
        values={"bream.PLATFORM_QC_RESULTS": results},
        lifetime=PERSIST_ACROSS_RESTARTS,
    )

    # Also use new version
    device.set_platform_qc_result(passed, total_pore_count)


def set_protocol_data_results(device: BaseDeviceInterface, results: protocol_communication.ProtocolDataResults) -> None:
    device.connection.keystore.store(
        values={"bream.protocol_communication": results},
        lifetime=PERSIST_ACROSS_RESTARTS,
    )


def set_protocol_data(
    device: BaseDeviceInterface,
    speed_min: float = 0,
    speed_max: float = 0,
    q_score_min: float = 0,
    q_score_max: float = 0,
    temp_min: float = 0,
    temp_max: float = 0,
) -> None:

    run_id = device.get_run_id()
    results = get_keystore_message(device, "bream.protocol_communication")
    assert isinstance(results, protocol_communication.ProtocolDataResults)  # nosec B101 Type checker
    results.data[run_id].translocation_speed.min = speed_min
    results.data[run_id].translocation_speed.max = speed_max
    results.data[run_id].q_score.min = q_score_min
    results.data[run_id].q_score.max = q_score_max
    results.data[run_id].temperature.min = temp_min
    results.data[run_id].temperature.max = temp_max
    results.data[run_id].bias_voltage_offset = device.get_bias_voltage_offset()

    set_protocol_data_results(device, results)


def set_state(device: BaseDeviceInterface, state_string: str, clear_trigger: bool = False) -> None:
    run_id = device.get_run_id()
    results = get_keystore_message(device, "bream.protocol_communication")
    assert isinstance(results, protocol_communication.ProtocolDataResults)  # nosec B101 Type checker

    states = protocol_communication.ProtocolData().ProtocolInternalState.keys()
    if state_string not in states:
        raise ValueError(f"State {state_string} not in {states}")

    logger.info("Transitioning to state {}".format(state_string))
    results.data[run_id].state = getattr(protocol_communication.ProtocolData(), state_string)
    results.data[run_id].state_change_time.seconds = int(time.time())

    if clear_trigger:
        results.data[run_id].ClearField("protocol_event_trigger")

    set_protocol_data_results(device, results)


def wait_for_protocol_trigger(
    device: BaseDeviceInterface, trigger: Literal["mux_scan", "pause", "resume", "stop"]
) -> None:
    run_id = device.get_run_id()

    logger.debug("Waiting for trigger: {} to appear".format(trigger))

    for msg in device.connection.keystore.watch(names=["bream.protocol_communication"], allow_missing=True):
        value = parse_keystore_message("bream.protocol_communication", msg)
        assert isinstance(value, protocol_communication.ProtocolDataResults)  # nosec B101 Type checker

        logger.debug("Message: {}".format(value))

        if run_id in value.data and value.data[run_id].HasField(trigger):
            return
