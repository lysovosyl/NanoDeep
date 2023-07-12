from grpc import RpcError
from minknow_api import keystore_service

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.pb.bream.mux_scan_results_pb2 import MuxCategory, MuxGroup, MuxScanResults, Result, Style

KEY_STORE_NAME = "bream.MUX_SCAN_RESULTS"

COLOURS = {
    "single_pore": "00cc00",
    "reserved_pore": "EDE797",
    "multiple": "F57E20",
    "saturated": "333333",
    "other": "0084A9",
    "zero": "90C6E7",
    "unavailable": "54B8B1",
}

STYLES = {
    "single_pore": Style(
        label="Single Pore",
        description="The well appears to show a single pore. Available for sequencing",
        colour=COLOURS["single_pore"],
    ),
    "reserved_pore": Style(
        label="Reserved Pore",
        description="The well appears to show a single pore. Available for sequencing",
        colour=COLOURS["reserved_pore"],
    ),
    "multiple": Style(
        label="Multiple",
        description="The well appears to show more than one pore. Unavailable for sequencing",
        colour=COLOURS["multiple"],
    ),
    "saturated": Style(
        label="Saturated",
        description="The well has switched off due to current levels exceeding hardware limitations",
        colour=COLOURS["saturated"],
    ),
    "other": Style(
        label="Other",
        description="Currently unavailable for sequencing",
        colour=COLOURS["other"],
    ),
    "zero": Style(
        label="Zero",
        description="Current level is between -5 and 10 pA. Currently unavailable for sequencing",
        colour=COLOURS["zero"],
    ),
    "unavailable": Style(
        label="Unavailable",
        description="The well appears to show a pore that is currently unavailable for sequencing",
        colour=COLOURS["unavailable"],
    ),
    "active_wells": Style(
        label="Active wells",
        description="Wells available for sequencing",
        colour=COLOURS["single_pore"],
    ),
    "inactive_wells": Style(
        label="Inactive wells",
        description="Wells unavailable for sequencing",
        colour=COLOURS["zero"],
    ),
}

MUX_CATEGORY = {
    "single_pore": MuxCategory(name="single_pore", style=STYLES["single_pore"], global_order=1),
    "reserved_pore": MuxCategory(name="reserved_pore", style=STYLES["reserved_pore"], global_order=1),
    "multiple": MuxCategory(name="multiple", style=STYLES["multiple"], global_order=2),
    "saturated": MuxCategory(name="saturated", style=STYLES["saturated"], global_order=3),
    "other": MuxCategory(name="other", style=STYLES["other"], global_order=4),
    "zero": MuxCategory(name="zero", style=STYLES["zero"], global_order=5),
    "unavailable": MuxCategory(name="unavailable", style=STYLES["unavailable"], global_order=6),
}

DEFAULT_SELECTED_GROUP = MuxGroup(
    name="active_wells",
    style=STYLES["active_wells"],
    category=[MUX_CATEGORY["single_pore"], MUX_CATEGORY["reserved_pore"]],
)

DEFAULT_UNSELECTED_GROUP = MuxGroup(
    name="inactive_wells",
    style=STYLES["inactive_wells"],
    category=[
        MUX_CATEGORY["unavailable"],
        MUX_CATEGORY["multiple"],
        MUX_CATEGORY["saturated"],
        MUX_CATEGORY["zero"],
        MUX_CATEGORY["other"],
    ],
)


def add_mux_scan_result_to_keystore(
    device: BaseDeviceInterface,
    scan_period: float,
    single_pore: int = 0,
    reserved_pore: int = 0,
    unavailable: int = 0,
    multiple: int = 0,
    saturated: int = 0,
    zero: int = 0,
    other: int = 0,
) -> int:
    """Stores the mux scan results in the keystore

    :param device: MinKNOW device wrapper
    :param scan_period: How often the mux scan happens (seconds)
    :param *: Numbers to populate the mux scan with
    :returns: How many mux scans are now stored for this run
    :rtype: int

    """

    mux_data_result = Result(
        counts={
            "single_pore": single_pore,
            "reserved_pore": reserved_pore,
            "unavailable": unavailable,
            "multiple": multiple,
            "saturated": saturated,
            "zero": zero,
            "other": other,
        },
        mux_scan_timestamp=device.get_acquisition_duration(),
    )

    run_id = device.get_run_id()

    new_value = MuxScanResults()
    # Try to populate with old values
    try:
        old_value = device.connection.keystore.get_one(name=KEY_STORE_NAME)
        old_value.value.Unpack(new_value)

    except RpcError:
        pass  # Not yet in the keystore. Doesn't matter. Added at the end

    if run_id not in new_value.results:
        # new_value.results[run_id] is auto created if it doesn't exist.
        # Convert to hours
        new_value.results[run_id].mux_scan_period = scan_period / (3600.0)
        new_value.results[run_id].muxGroups.extend([DEFAULT_SELECTED_GROUP, DEFAULT_UNSELECTED_GROUP])

    new_value.results[run_id].results.extend([mux_data_result])  # protobuf doesn't have `append`

    device.connection.keystore.store(
        values={KEY_STORE_NAME: new_value},
        lifetime=keystore_service.PERSIST_ACROSS_RESTARTS,
    )

    return len(new_value.results[run_id].results)
