from minknow_api.acquisition_service import MuxScanMetadata

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface

COLOURS = {
    "single_pore": "00CC00",
    "reserved_pore": "EDE797",
    "multiple": "F57E20",
    "saturated": "333333",
    "other": "0084A9",
    "zero": "90C6E7",
    "unavailable": "54B8B1",
}

STYLES = {
    "single_pore": MuxScanMetadata.Style(
        label="Single Pore",
        description="The well appears to show a single pore. Available for sequencing",
        colour=COLOURS["single_pore"],
    ),
    "reserved_pore": MuxScanMetadata.Style(
        label="Reserved Pore",
        description="The well appears to show a single pore. Available for sequencing",
        colour=COLOURS["reserved_pore"],
    ),
    "multiple": MuxScanMetadata.Style(
        label="Multiple",
        description="The well appears to show more than one pore. Unavailable for sequencing",
        colour=COLOURS["multiple"],
    ),
    "saturated": MuxScanMetadata.Style(
        label="Saturated",
        description="The well has switched off due to current levels exceeding hardware limitations",
        colour=COLOURS["saturated"],
    ),
    "other": MuxScanMetadata.Style(
        label="Other",
        description="Currently unavailable for sequencing",
        colour=COLOURS["other"],
    ),
    "zero": MuxScanMetadata.Style(
        label="Zero",
        description="Current level is between -5 and 10 pA. Currently unavailable for sequencing",
        colour=COLOURS["zero"],
    ),
    "unavailable": MuxScanMetadata.Style(
        label="Unavailable",
        description="The well appears to show a pore that is currently unavailable for sequencing",
        colour=COLOURS["unavailable"],
    ),
    "active_wells": MuxScanMetadata.Style(
        label="Active wells",
        description="Wells available for sequencing",
        colour=COLOURS["single_pore"],
    ),
    "inactive_wells": MuxScanMetadata.Style(
        label="Inactive wells",
        description="Wells unavailable for sequencing",
        colour=COLOURS["zero"],
    ),
}

MUX_CATEGORY = {
    "single_pore": MuxScanMetadata.Category(name="single_pore", style=STYLES["single_pore"], global_order=1),
    "reserved_pore": MuxScanMetadata.Category(name="reserved_pore", style=STYLES["reserved_pore"], global_order=1),
    "multiple": MuxScanMetadata.Category(name="multiple", style=STYLES["multiple"], global_order=2),
    "saturated": MuxScanMetadata.Category(name="saturated", style=STYLES["saturated"], global_order=3),
    "other": MuxScanMetadata.Category(name="other", style=STYLES["other"], global_order=4),
    "zero": MuxScanMetadata.Category(name="zero", style=STYLES["zero"], global_order=5),
    "unavailable": MuxScanMetadata.Category(name="unavailable", style=STYLES["unavailable"], global_order=6),
}

DEFAULT_SELECTED_GROUP = MuxScanMetadata.CategoryGroup(
    name="active_wells",
    style=STYLES["active_wells"],
    category=[MUX_CATEGORY["single_pore"], MUX_CATEGORY["reserved_pore"]],
)

DEFAULT_UNSELECTED_GROUP = MuxScanMetadata.CategoryGroup(
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


def add_mux_scan_result_to_minknow(
    device: BaseDeviceInterface,
    scan_period: float,
    single_pore: int = 0,
    reserved_pore: int = 0,
    unavailable: int = 0,
    multiple: int = 0,
    saturated: int = 0,
    zero: int = 0,
    other: int = 0,
) -> None:
    """Stores the mux scan results in MinKNOW

    :param device: MinKNOW device wrapper
    :param scan_period: How often the mux scan happens (seconds)
    :param *: Numbers to populate the mux scan with
    """

    # Update the metadata in-case it changes
    # (Currently never, but we don't have to check it's set if we always set)
    metadata = MuxScanMetadata(
        auto_mux_scan_period_hours=round(scan_period / (60 * 60), 1),
        category_groups=[DEFAULT_SELECTED_GROUP, DEFAULT_UNSELECTED_GROUP],
    )
    device.set_mux_scan_metadata(metadata)

    # Add mux scan
    device.append_mux_scan_result(
        {
            "single_pore": single_pore,
            "reserved_pore": reserved_pore,
            "unavailable": unavailable,
            "multiple": multiple,
            "saturated": saturated,
            "zero": zero,
            "other": other,
        }
    )
