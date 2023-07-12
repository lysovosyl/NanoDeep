from __future__ import annotations

import logging
from typing import Optional

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.feature_manager import ChannelNotifierState
from bream4.toolkit.procedure_components.sequencing_features.internal_state_information import InternalStateInformation


class GroupManager(object):
    """This is the manager for dealing with swapping channels from one
    valid mux to the next.  There are 2 modes for the group
    manager.

    The first, by default, is "Active channel selection" this
    is to swap out a mux in a channel if it becomes unusable".

    The second, is to transition all the channels at the same time to swap
    out a mux in a channel if it becomes unusable.

    :code:`swap_out_disabled_channels` determines whether or not to change
    mux as soon as it becomes unavailable. (Active channel selection)

    """

    def __init__(self, config: dict, device: BaseDeviceInterface, internal_state_object: InternalStateInformation):
        self.config = config
        self.device = device
        self.internal_state_object = internal_state_object

        self.swap_out = config.get("swap_out_disabled_channels", True)

        self.logger = logging.getLogger(__name__)

    def set_next_group(self, bias_voltage: Optional[float] = None) -> None:
        """
        Trigger a group change. Every channel will move on to the next
           available mux in their group - As determined by the set_groups method.
           Most likely called by a global mux change

        :param bias_voltage: Which bias voltage to go to after the group switch.
                             If None specified, use the current bias voltage
        """
        # Most likely called by a global_mux_change
        # Should only be called from an execute method

        new_groups = self.internal_state_object.get_next_group()

        # Reset bias voltage for group change
        if not bias_voltage:
            bias_voltage = self.device.get_bias_voltage()
        self.device.set_bias_voltage(0)
        self.device.set_all_channels_to_well(0)
        self.device.set_channels_to_well(new_groups)
        self.device.set_bias_voltage(bias_voltage)

    def execute(self, states: Optional[dict[int, ChannelNotifierState]] = None) -> None:
        """Swaps out any locked channels as it sees them. Provided
            `swap_out_disabled_channels` == True in the config

        :param states: namedtuple(state_name, well)

        """
        # Handles swapping of locked channels
        if states:
            to_swap = {}
            for (channel, state) in states.items():
                if state.state_name == "locked":
                    if self.swap_out:
                        to_swap[channel] = self.internal_state_object.get_next_group_for_channel(channel)
                    else:
                        # If don't want to switch it out, should switch it off at least
                        to_swap[channel] = 0
            if to_swap:

                channels_to_0 = [channel for (channel, well) in to_swap.items() if well == 0]
                channels_not_to_0 = [channel for (channel, well) in to_swap.items() if well != 0]

                if channels_to_0:
                    # No muxes to swap to!
                    self.device.lock_channel_states(channels_to_0, "disabled")
                    self.logger.info("Keeping {} locked, as no replacement wells available".format(channels_to_0))

                if channels_not_to_0:
                    # Make sure to unlock first as latched channel states persist after unlock on a mux change
                    self.device.unlock_channel_states(channels=channels_not_to_0)

                self.device.set_channels_to_well(to_swap)
