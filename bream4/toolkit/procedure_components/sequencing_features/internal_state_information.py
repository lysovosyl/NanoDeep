from __future__ import annotations

import logging
from collections.abc import Iterator
from itertools import chain, cycle, repeat


class InternalStateInformation(object):
    """
    This object is a supportive internal state object designed to hold state information that can be shared between
    other features running together, with the aim of increasing the response times and improving the decision making of
    these features

    """

    def __init__(self, cycle_groups: bool, cycle_limit_per_channel: int):
        """
        :code:`cycle_groups` determines what happens when a channel has run out of muxes to swap to. If true, groups
        will be cycled and the first will be picked again. If false, the channel will be disconnected.

        :code:`cycle_limit_per_channel` is to limit how many muxes can be in
        the group.  This is useful if you are not doing active channel
        selection then the last group may not have many channels in it so it may be
        worth not bothering and thus go back to the first group where
        there are more available muxes
        """
        self.groups: dict[int, Iterator[int]] = {}
        self.cycle_groups = cycle_groups
        self.cycle_limit_per_channel = cycle_limit_per_channel

        self.logger = logging.getLogger(__name__)

    def set_groups(self, groups: dict[int, list[int]]) -> None:
        """
        This sets up the groups to be used by the group_manager. These
           will be used with the next_group method (For a global mux change) or
           with the execute method (Swapping out disabled channels)
        :param groups: channel -> list of muxes
        """

        # This sets up a dictionary, indexed by channel
        # You call next() on the channel and it will give you the next mux to move to

        if self.cycle_groups:
            # Limit up to cycle_limit_per_channel
            groups = {channel: wells[: self.cycle_limit_per_channel] for (channel, wells) in groups.items()}
            # Pad with 0's up to cycle_limit_per_channel
            groups = {
                channel: wells + [0] * (self.cycle_limit_per_channel - len(wells))
                for (channel, wells) in groups.items()
            }
            # Create a cycle
            iter_groups: dict[int, Iterator[int]] = {channel: cycle(wells) for (channel, wells) in groups.items()}
        else:
            # If want to swap out then cycling isn't going to happen. Make sure to pad with 0s
            iter_groups: dict[int, Iterator[int]] = {
                channel: chain(iter(wells[::]), repeat(0)) for (channel, wells) in groups.items()
            }
        self.groups = iter_groups

    def get_next_group_for_channel(self, channel: int) -> int:
        """
        :param channel: the channel being queried
        returns the next mux for the channel passed in
        """
        return next(self.groups[channel])

    def get_next_group(self) -> dict[int, int]:
        """Returns the next group. Every channel will move on to the next
        available mux in their group - As determined by the set_groups method.
        Most likely called by a global mux change

        """
        # Most likely called by a global_mux_change

        self.logger.info("Next group requested")
        # Should only be called from an execute method
        new_groups = {channel: next(mux) for (channel, mux) in self.groups.items()}
        return new_groups
