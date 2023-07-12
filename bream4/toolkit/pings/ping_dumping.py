from __future__ import annotations

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bream4.device_interfaces.devices.base_device import BaseDeviceInterface


# Native numpy types are not json serialisable
# This causes weird type issues when turning them to JSON
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super().default(obj)


def ping_to_str(ping: dict, **kwargs) -> str:
    return json.dumps(ping, cls=NumpyEncoder, **kwargs)


def dump_ping(device: "BaseDeviceInterface", ping: dict, logger=None) -> None:
    """
    Dump the JSONified ping data to a file
    :param device: current device wrapper
    :param ping: JSON object
    :param logger: optional, log message to current logger
    """
    log_dir = device.get_logs_directory()

    time_now = datetime.now()
    now_str = datetime.strftime(time_now, "%Y-%m-%d_%H-%M-%S")
    outfile = os.path.join(log_dir, "result_ping_{}.json".format(now_str))

    # Write to file
    with open(outfile, "w+") as output:
        output.write(ping_to_str(ping, indent=4))
        output.write("\n")

    if logger is not None:
        logger.debug("Written json file to: {}".format(outfile))
