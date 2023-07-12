from __future__ import annotations

from collections import defaultdict

from minknow_api.data_service import GetReadStatisticsResponse
from typing_extensions import TypeAlias

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface

# (channel, well) -> ( (classification) -> PerClassificationData )
Returndict: TypeAlias = "dict[tuple[int, int], dict[str, GetReadStatisticsResponse.PerClassificationData]]"


def get_channel_read_classifications(
    device: BaseDeviceInterface,
    collection_time: float,
    completed_read: bool = False,
    include_current_statistics: bool = True,
    include_chunk_statistics: bool = False,
) -> Returndict:
    """Collect metrics for collection_time. This is a passive get as minknow will not be altered

    If a channel is saturated/disconnected at the start of collection,
    There may be nothing returned for that channel

    :param device: MinKNOW device wrapper
    :param collection_time: How long to collect for
    :param completed_read: Whether to do complete reads or read chunks
    :param include_current_statistics: Whether to include current statistics in the response
    :param include_chunk_statistics: Whether to include chunk statstics in the response

    :returns: (channel, well) -> { classification -> statistics }
    :rtype: dict of dict

    """
    raw_stats = device.get_read_statistics(
        collection_time,
        completed_read=completed_read,
        include_current_statistics=include_current_statistics,
        include_chunk_statistics=include_chunk_statistics,
    )
    disconnected_status = device.get_disconnection_status_for_active_wells()
    stats = _transform_read_statistics(raw_stats, disconnected_status, device.get_channel_list())
    return stats


def _transform_read_statistics(
    stats: GetReadStatisticsResponse, disconnected: dict[int, bool], channels_order_requested: list[int]
) -> Returndict:
    """Transforms the stats from protobuf style into:
    (channel, well) -> ( (classification) -> PerClassificationData )
    e.g.
    (100, 1): { 'tba': RCB, 'pore': RCB, ... , 'disconnected': True}

    :param stats: GetReadStatisticsResponse
    :param channels_order_requested: list of channels requested
    :returns: A mapping of channel, well to a dict of classification types
    :rtype: dict of dicts

    """
    transformed = defaultdict(dict)

    for read_per_channel, channel in zip(stats.channels, channels_order_requested):
        for reads_per_channel_conf in read_per_channel.configurations:
            well = reads_per_channel_conf.channel_configuration.well
            if (channel, well) in transformed:
                continue
            for (
                classification,
                data,
            ) in reads_per_channel_conf.classifications.items():
                transformed[(channel, well)][classification] = data
            transformed[(channel, well)]["disconnected"] = disconnected[channel]

    return transformed
