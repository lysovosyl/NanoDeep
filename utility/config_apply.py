"""
----------------------------------------------------------------------------------------------------
                Utility for applying settings to device, from a loaded config
----------------------------------------------------------------------------------------------------
Simply call `config_apply_to_device(dict)` to apply any device settings to the device.
"""
from __future__ import annotations

import logging
import pprint
from copy import deepcopy
from functools import reduce
from typing import Optional

import bream4.device_interfaces.device_wrapper as device_wrapper
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.command.context_tags import set_context_tags
from utility.config_file_utils import expand_grouped_list, find_path, recursive_dict_iter, recursive_merge


def filter_config_for_device(config: dict, device: Optional[BaseDeviceInterface] = None) -> dict:
    """Collapses the config so that any device specific config gets integrated: ::

      [a.b]
        y=10
      [a.b.minion]
        x=20
      [a.b.promethion]
        x=15

    will become: ::

      [a.b]
        y=10
        x=15/20 depending on platform

    :param config: dict of configs loaded from a valid toml
    :param device: MinKNOW device wrapper
    :returns: config that is compressed
    :rtype: dict

    """

    if not device:
        device = device_wrapper.create_grpc_client()

    paths = set()
    collapsable_elements = ["minion_like", "minion", "gridion", "flongle", "promethion", "mk1c", "minit"]

    for (key, value) in recursive_dict_iter(config):
        # Check if the key has .minion./.promethion./.flongle.
        for elem in collapsable_elements:
            remove = ".{}.".format(elem)
            if remove in key:
                # If it does, add the path to be the one to remove
                paths.add(key[: key.index(remove)])

    # We now have all the paths. So collapse them
    return reduce(lambda conf, path: filter_config_path_for_device(conf, path, device), paths, config)


def filter_config_path_for_device(
    config: dict, path: str = "device", device: Optional[BaseDeviceInterface] = None
) -> dict:
    """Given a config and a specific path, return a new config where
    the settings for the specific device is collapsed.

    This is intended as a helper nanodeep for filter_config_for_device but can be used independently

    i.e. Remove minion/flongle/promethion settings and replace them with just whichever device it is

    e.g. ::

      [device.promethion]
      x = 10
      [device.minion]
      x = 20

    would give: ::

      [device]
      x=10/20 depending on device

    If minion_like is specified it will be merged with the specific platform: ::

      [device.minion_like]
      x = 1
      y = 2
      [device.flongle]
      y = 3

    Would give x=1, y=2 for minion/gridion,
               x=1, y=3 for flongle

    :param dict config: Config to look at
    :param str path: Path to filter device settings from
    :param Device device: Device to be checked
    :returns: New config with device settings filtered
    :rtype: dict
    """

    if not device:
        device = device_wrapper.create_grpc_client()

    if not find_path(path, config):
        return config

    # Filtering dict in place, so make a copy
    new_config = deepcopy(config)

    # Separate the specific configs
    lookup: dict = find_path(path, new_config)  # type: ignore
    config_minion_like = lookup.pop("minion_like", {})
    config_flongle = lookup.pop("flongle", {})
    config_minion = lookup.pop("minion", {})
    config_mk1c = lookup.pop("mk1c", {})
    config_minit = lookup.pop("minit", {})
    config_gridion = lookup.pop("gridion", {})
    config_promethion = lookup.pop("promethion", {})

    # Grab config specifics
    device_specific = {}
    if device.is_minion_like:
        device_specific = config_minion_like
        if device.is_gridion:
            recursive_merge(device_specific, config_gridion)
        elif device.is_minit:
            recursive_merge(device_specific, config_minit)
        elif device.is_mk1c:
            recursive_merge(device_specific, config_mk1c)
        elif device.is_minion:
            recursive_merge(device_specific, config_minion)

        if device.is_flongle:
            recursive_merge(device_specific, config_flongle)

    elif device.is_promethion:
        device_specific = config_promethion

    new_location: dict = find_path(path, new_config)  # type: ignore
    recursive_merge(new_location, device_specific)
    return new_config


def config_apply_saturation(config: dict, device: BaseDeviceInterface, logger: logging.Logger) -> None:
    """Handle the setting of saturation as it isn't a straight forward :code:`set_` call

    See the Device nanodeep calls for an explanation

    The saturation config should look like: ::

      [blah.saturation.adc]
      min = -5
      max = 1900
      threshold = 640
      saturation_during_unblock = false

      [blah.saturation.pa]
      min = -10
      max = 100
      general_threshold = 640
      saturation_during_unblock = false

    :param dict config: config dict (See `config_file_utils.load_config`)
    :param device: The device interface to apply the config to.
    :param logger: The logger to cause a pop up of if any attributes are left unapplied
    :raises RuntimeError: If any saturation config values are left unapplied
    """
    saturation = config["device"].pop("saturation", {})

    saturation_adc = saturation.pop("adc", {})
    min_adc = saturation_adc.pop("min", None)
    max_adc = saturation_adc.pop("max", None)
    threshold = saturation_adc.pop("threshold", 0)
    saturation_during_unblock = saturation_adc.pop("saturation_during_unblock", False)

    # Min/max can't be set if both thresholds are 0 but they need to be present in the config
    # so that users can see what attributes can be set
    if threshold != 0:
        if min_adc is not None and max_adc is not None:
            # If there is a threshold and limits
            device.set_saturation_adc(
                min_adc, max_adc, threshold=threshold, saturation_during_unblock=saturation_during_unblock
            )
        else:
            raise RuntimeError("Can't set ADC saturation threshold without min/max values")
    else:
        device._set_saturation_thresholds(general_threshold=0, unblock_threshold=0)

    # Deal with user saturation
    saturation_pa = saturation.pop("pa", {})
    min_pa = saturation_pa.pop("min", None)
    max_pa = saturation_pa.pop("max", None)
    threshold = saturation_pa.pop("threshold", 0)
    saturation_during_unblock = saturation_pa.pop("saturation_during_unblock", False)

    if threshold != 0:
        if min_pa is not None and max_pa is not None:
            # If there is a threshold and limits
            device.set_saturation_pa(
                min_pa, max_pa, threshold=threshold, saturation_during_unblock=saturation_during_unblock
            )
        else:
            raise RuntimeError("Can't set pA saturation threshold without min/max values")

    else:
        device._set_saturation_thresholds(user_general_threshold=0, user_unblock_threshold=0)

    saturation.update(saturation_pa)
    saturation.update(saturation_adc)
    if len(saturation.keys()):
        logger.log_to_gui("config_apply.atts_non_existent", params=dict(keys=" ".join(saturation.keys())))
        msg = "Configuration is trying to set saturation attributes " + "{} but they do not exist".format(
            saturation.keys()
        )
        raise RuntimeError(msg)


def config_apply_writer(config: dict, device: BaseDeviceInterface) -> None:
    """Applies writer configuration to the device

    The writer configuration should look like: ::

      [writer_configuration.read_fast5]
      raw = [1, 10, [20, 30]]
      ...

    Look at the library file for more information

    :param config: dict of settings to apply.
    :param device: MinKNOW device wrapper

    """

    if "writer_configuration" not in config:
        return

    all_channels = set(device.get_channel_list())

    channel_list_attributes = [
        "enable",
        "raw",
        "fastq",
        "events",
        "reads",
        "multiplex",
        "channel_states",
        "trace_table",
        "move_table",
        "modifications_table",
    ]

    # Make sure if bulk is switched on for something
    bulk_settings = config["writer_configuration"].get("bulk", {})
    if any([bulk_settings.get(item) for item in channel_list_attributes]):
        bulk_settings["device_metadata"] = True
        bulk_settings["device_commands"] = True

    for entry in config["writer_configuration"].values():
        for attribute in entry:
            if attribute in channel_list_attributes:
                channels = expand_grouped_list(entry[attribute])
                if channels == []:
                    entry[attribute] = {"all_channels": False}
                elif all_channels.issubset(set(channels)):
                    entry[attribute] = {"all_channels": True}
                else:
                    entry[attribute] = {"specific_channels": {"channels": channels}}

    device.set_writer_configuration(config["writer_configuration"])


def config_apply_analysis_configuration(config: dict, device: BaseDeviceInterface) -> None:
    """Applies analysis_configuration to the device with transforming channel states

    :param config: dict config from toml file
    :param device: MinKNOW device wrapper

    """

    if "analysis_configuration" not in config:
        return

    # Editing inplace so use a deep copy
    analysis_configuration = deepcopy(config["analysis_configuration"])

    current_analysis_configuration = device.get_analysis_configuration()

    # If we've specified channel states, expand the group definitions
    if "channel_states" in analysis_configuration:
        channel_states = analysis_configuration["channel_states"]

        # Transform channel state groups to be compatible
        groups = {}
        for (group_name, group_info) in channel_states.pop("groups", {}).items():
            if "style" not in group_info:
                raise RuntimeError("Group {} in channel states does not define a style".format(group_name))

            groups[group_name] = group_info
            groups[group_name]["name"] = group_name

        # Expand group definitions
        for channel_state_info in channel_states.values():
            if "group" not in channel_state_info:
                continue

            group_name = channel_state_info["group"]
            if group_name in groups:
                channel_state_info["group"] = groups[group_name]
            else:
                # This is under the assumption that the group is already defined in the UI somehow
                channel_state_info["group"] = {"name": group_name}

            # Certain protobufs are old-school and are expecting ints instead of bools
            int_casts = {
                "reset_on_mux_change",
                "reset_on_well_change",
                "latch",
                "reset_on_effective_mux_change",
                "reset_on_effective_well_change",
            }
            channel_state_behaviour = find_path("logic.behaviour", channel_state_info) or {}
            for cast in int_casts:
                if cast in channel_state_behaviour:
                    channel_state_behaviour[cast] = int(channel_state_behaviour[cast])

    current_analysis_configuration.update(analysis_configuration)

    # And apply
    device.set_analysis_configuration(current_analysis_configuration, merge=False)


def config_apply_basecaller_configuration(config: dict, device: BaseDeviceInterface) -> None:
    """Apply basecaller configuration

    :param config: Dict of config
    :param device: MinKNOW device wrapper

    """

    if "basecaller_configuration" not in config:
        return

    device.preload_basecaller_configuration(config["basecaller_configuration"])
    device.set_basecaller_configuration(config["basecaller_configuration"])


def config_apply_to_device(config: dict, device: Optional[BaseDeviceInterface] = None) -> dict:
    """Input a config generated from config utility's load_config, and this will set the device
    settings based on the device platform

    Note any more specific device settings will overwrite more general settings

    If there are any values that cannot be set, log to gui

    This assumes that the config is a VALID config. i.e. it has been successfully loaded from
    config_file_utils.load_config

    :param dict config: config dict (See `config_file_utils.load_config`)
    :param device: The device interface to apply the config to.
                   If not specified, one will be created
    :returns: dict of values successfully applied
    :rtype: dict
    """

    if device is None:
        device = device_wrapper.create_grpc_client()

    logger = logging.getLogger(__name__)

    config = filter_config_for_device(config, device=device)

    device.reset()

    # Deal with saturation first
    if "saturation" in config["device"]:
        config_apply_saturation(config, device, logger)

    # Loop over and call the functions that will assign the values
    for (name, value) in config["device"].items():
        try:
            getattr(device, "set_" + name)(value)
            logger.info("Attribute %s set to %s" % (name, value))
        except AttributeError:
            logger.log_to_gui("config_apply.func_non_existent", params=dict(key=name, value=value))
            msg = "Configuration is trying to set " + "%s to %s, but that functionality does not exist" % (name, value)
            raise RuntimeError(msg)

    device.set_exp_script_purpose(config["meta"]["exp_script_purpose"])
    set_context_tags(device, config)

    config_apply_writer(config, device)

    config_apply_analysis_configuration(config, device)
    config_apply_basecaller_configuration(config, device)

    logger.info("Configuration applied:")
    logger.info(pprint.pformat(config))

    return config
