from __future__ import annotations

import copy
import logging
import time
from collections import defaultdict
from collections.abc import Iterator
from itertools import chain, repeat
from typing import Optional

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.feature_manager import ChannelNotifierState

# Progressive unblock will flick any channels that are in states_to_flick. And keep flicking
# until a state that is in states_to_reset comes up

DEFAULTS = {
    "flick_duration": [],
    "rest_duration": [],
    "repeats": [],
    "states_to_flick": ["unavailable"],
    "states_to_reset": ["strand", "pore", "event"],
    "change_mux_after_last_tier": False,
}


class ProgressiveUnblock(object):
    def __init__(self, config: dict, device: BaseDeviceInterface):
        self.config = copy.deepcopy(DEFAULTS)
        self.config.update(config)
        self.device = device
        self.logger = logging.getLogger(__name__)

        # Store the next flick time for each channel
        self.next_flick_time = {}

        # Calling stage_tracker[channel].next() will give (flick_time, rest_time) or StopIterator
        self.stage_tracker = defaultdict(self._flick_iterator)

        self.paused = False

        # Grab the tiers
        self.flick_duration = self.config["flick_duration"]
        self.rest_duration = self.config["rest_duration"]
        self.repeats = self.config["repeats"]

        # Which mux we expect each channel to be in
        self.expected_mux = {}

        # Should have the same number of flicks/rests/repeats
        if not (len(self.flick_duration) == len(self.rest_duration) == len(self.repeats)):
            raise RuntimeError(
                "Progressive unblock has different lengths. Check flick_duration, resst_duration and repeats."
            )

        self.states_to_flick = self.config["states_to_flick"]
        self.states_to_reset = self.config["states_to_reset"]

        self.flag_on_last_tier = self.config["change_mux_after_last_tier"]

        if "unblock_voltage" in config:
            device.set_unblock_voltage(self.config["unblock_voltage"])

    def _flick_iterator(self) -> Iterator[tuple[float, float]]:
        # Returns an iterator. So you can keep calling next to iterate through the tiers:
        # next(..) will give (flick_duration, rest_time) until the last one is reached
        # A StopIterator will be raised on the last one
        flick_list = []
        for (flick, rest, times) in zip(self.flick_duration, self.rest_duration, self.repeats):
            flick_list.append(repeat((flick, rest), times))

        return chain(*flick_list)

    def _unblock(self, channels: list[int]):
        if not channels:
            return

        # Group the flicks in durations
        # flick_tiers[duration] = [ channels to flick ]
        flick_tiers = defaultdict(list)

        # Stores channels that need to be flagged to be replaced
        flagged_channels = []

        for channel in channels:
            try:
                (flick_duration, rest_duration) = next(self.stage_tracker[channel])
                # Update next flick time to ensure don't flick before
                self.next_flick_time[channel] = time.monotonic() + flick_duration + rest_duration

                # Add to correct tier
                flick_tiers[flick_duration].append(channel)

            except StopIteration:
                self.logger.info("Channel {} has come to the end of the tiers".format(channel))
                # Delete from stage_tracker -> Will get restarted
                del self.stage_tracker[channel]

                # And if we need to replace, then add it to the list of channels to be replaced
                if self.flag_on_last_tier:
                    flagged_channels.append(channel)

        # Flick the channels
        for (flick_duration, channels_per_tier) in sorted(flick_tiers.items()):
            self.logger.debug("Flicking channels {} for {} seconds".format(str(channels_per_tier), flick_duration))
            self.device.unblock(channels_per_tier, flick_duration)

        if flagged_channels:
            self.device.lock_channel_states(channels=flagged_channels, state_name="locked")
            self.device.set_channels_to_well({channel: 0 for channel in flagged_channels})

    def _cancel_unblocks(self) -> None:
        self.logger.info("Stopped {} unblock calls".format(self.device.cancel_unblocks()))
        self.next_flick_time.clear()
        self.stage_tracker.clear()

    def execute(self, states: Optional[dict[int, ChannelNotifierState]] = None) -> None:
        """Run the progressive unblock feature.

        This will flick any that are waiting to be flicked and also parse any
        new state information.

        States is expected to be a dict of channel -> state_info
        Where state_info has .well and .state_name

        Due to how the feature manager works, states= will be filled in on channel state updates
        And will not be popualted if just scheduled to run

        E.g. {100: (well=1, state_name="unavailable")}
        :param states: (Optional) Any extra state information.

        """
        if self.paused:
            return

        # If we have extra state information
        if states:
            for (channel, state) in states.items():

                # If we should flick
                if state.state_name in self.states_to_flick and state.well != 0:

                    # If it has changed mux, reset the tiers
                    if channel in self.expected_mux and self.expected_mux[channel] != state.well:
                        self.next_flick_time.pop(channel, None)
                        self.stage_tracker.pop(channel, None)

                    # ... and if we haven't already scheduled it to flick
                    if channel not in self.next_flick_time:
                        # ..Set it to immediately flick
                        self.next_flick_time[channel] = 0

                    self.expected_mux[channel] = state.well

                # If we're in a reset state or we hopped to a disconnected state..
                elif state.state_name in self.states_to_reset:
                    # Get rid of any scheduled flicks and reset where we are in the tiers
                    self.next_flick_time.pop(channel, None)
                    self.stage_tracker.pop(channel, None)
                else:
                    # This means it's a valid well but in one of the ignored states
                    pass

        # Find any channels that are schedule to flick
        # So that any previously waiting also get a chance
        to_flick = [channel for (channel, flick_time) in self.next_flick_time.items() if flick_time <= time.monotonic()]
        self._unblock(to_flick)

    def stop(self) -> None:
        self.logger.info("Asked to stop")
        self.paused = True
        self._cancel_unblocks()

    def resume(self) -> None:
        self.paused = False

    def reset(self) -> None:
        self.logger.info("Asked to reset")
        self._cancel_unblocks()
        self.expected_mux.clear()

    def exit(self) -> None:
        self.logger.info("Asked to exit")
        self._cancel_unblocks()
        self.expected_mux.clear()
