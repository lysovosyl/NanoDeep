from __future__ import annotations

import logging
from typing import Any

import bream4
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface

DEFAULT: dict[str, str] = {}


def reset_context_tag_information(device: BaseDeviceInterface) -> None:
    """
    Resets the context tags set in minknow from the previous experiment. Clears
     any currently set context tags

    :param device: an instance of the connection class to MinKNOW
    """
    logger = logging.getLogger(__name__)
    logger.info("Resetting context tags in MinKNOW")
    device.set_context_tags(DEFAULT)


def add_context_tags(device: BaseDeviceInterface, tags_to_add: dict) -> dict[str, str]:
    """
    Adds context tags to the current ones

    :param device: an instance of the connection class to MinKNOW
    :param dict tags_to_add: dictionary containing the fields to add. Both the
     key and the value must be convertible to string

    :return: the updated context tags
    """
    logger = logging.getLogger(__name__)

    def format_for_context_tags(s):
        """
        :param s: object convertible to string
        :return: the formatted string as should appear in the context tags
        """
        ret = str(s)
        ret = ret.replace(" ", "_")
        return ret.lower()

    tags_to_add = {format_for_context_tags(k): format_for_context_tags(v) for k, v in tags_to_add.items()}
    tags = device.get_context_tags()
    tags.update(tags_to_add)
    device.set_context_tags(tags)

    logger.info("Updating context tags in MinKNOW with {}".format(tags))

    return device.get_context_tags()


def set_context_tags(device: BaseDeviceInterface, config: dict[Any, Any]) -> None:
    """Sets the context tags based on the configuration dictionary passed in.

    This assumes that the config is a VALID config. i.e. it has been successfully loaded from
    config_file_utils.load_config

    :param Device device: Device to apply the context tags to
    :param dict config: config dict (See `config_file_utils.load_config`)
    """

    context_tags = {
        "sample_frequency": str(device.get_sample_rate()),
        "experiment_type": config["meta"]["protocol"]["experiment_type"],
        "package": "bream4",
        "package_version": str(bream4.__version__),
    }

    # Make sure any in the config override what are the defaults above
    if "context_tags" in config["meta"]:
        context_tags.update(config["meta"]["context_tags"])

    reset_context_tag_information(device)
    add_context_tags(device, context_tags)
