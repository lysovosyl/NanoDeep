from __future__ import annotations

import math
import time
from collections.abc import Generator
from itertools import islice
from typing import TYPE_CHECKING, TypeVar

import bream4.legacy.device_interfaces.devices.base_device as legacy_base

if TYPE_CHECKING:
    from bream4.device_interfaces.devices.minion import MinionGrpcClient

K = TypeVar("K")
V = TypeVar("V")


def group_iterator(data: dict[K, V], size: int) -> Generator[dict[K, V], None, None]:
    it = iter(data)
    for i in range(0, len(data), size):
        yield {k: data[k] for k in islice(it, size)}


def set_channels_to_well(device: "MinionGrpcClient", channel_config: dict[int, int]) -> None:
    """Change in 10 groups to avoid saturation spikes in surrounding channels

    :param device: MinKNOW device wrapper
    :param channel_config: dict of channel-> well

    """
    if len(channel_config) < 20:
        device._set_channels_to_well(channel_config)
        return
    if device.is_flongle:
        # this is because flongle is more sensitive to the hardware issues that require the mux's on this ASIC design
        # to be set in groups instead of all together.
        num_groups = 126
    else:
        num_groups = 10
    group_size = int(math.ceil(len(channel_config) / num_groups))

    # Generate a map of channel to protobuf messages
    config_map = {
        channel: device.dev_msgs.ChannelConfiguration(well=well, test_current=False)
        for (channel, well) in channel_config.items()
    }

    for partial_config_map in group_iterator(config_map, group_size):
        # Apply this config
        device.connection.device.set_channel_configuration(channel_configurations=partial_config_map)
        # Give chance for saturation to occur
        time.sleep(0.1)

    # Final sleep for saturation (INST-4496)
    time.sleep(0.3)

    # Update timings in the wells
    legacy_base.channel_well_configuration(channel_config)


def mk1c_signal_mitigation() -> None:
    """
    Currently we are observing a drop in current levels when setting mux on mk1C, it only lasts ~2 seconds before
    recovering. this is currently under investigation, while its being investigated a mitigation of a wait of 4 seconds
     is being introduced for mk1c devices, the hope is to remove this once the source of the current drop is identified
     and fixed, the Jira ticket tracking this is INST-4024
    """
    time.sleep(4.0)
