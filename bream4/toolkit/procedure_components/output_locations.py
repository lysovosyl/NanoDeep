from __future__ import annotations

from pathlib import Path
from typing import Optional

from grpc import RpcError

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface


def get_run_output_path(device: BaseDeviceInterface, filename: str, subdirs: Optional[list[str]] = None) -> Path:
    """Returns filename transformed to an absolute path with identifiers in the filename:

    Inputting test.csv and running a protocol would return:
    /data/folder/other_reports/test_MN12345_0fb1c9aa.csv

    If subdirs is None, it will default to ["other_reports"].
    If this isn't required pass subdirs = [].

    If an experiment is not running (Testing etc) it will return it in relation to the log folder

    :param device: MinKNOW device wrapper
    :param filename: filename to transform (With suffix)
    :param subdirs: Put file in subdirs from main run path
    :returns: Path to transformed filename
    """

    stem = Path(filename).stem
    suffix = Path(filename).suffix

    if subdirs is None:
        subdirs = ["other_reports"]

    try:
        base_path = Path(device.get_protocol_output_path())
    except RpcError:
        # Not running as a protocol. Fall back to putting in the logs directory
        base_path = Path(device.get_logs_directory())

    new_fn = f"{stem}_{device.get_flow_cell_id()}_{device.get_run_id()[:8]}{suffix}"

    for subdir in subdirs:
        base_path = base_path / subdir

    # Make sure the paths exist
    base_path.mkdir(parents=True, exist_ok=True)

    return base_path / new_fn
