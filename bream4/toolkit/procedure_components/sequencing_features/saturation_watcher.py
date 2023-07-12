from __future__ import annotations

from collections import defaultdict
from typing import Optional

import pandas as pd

from bream4.toolkit.procedure_components.feature_manager import ChannelNotifierState


class SaturationWatcher:
    """Used to watch and record when channels become saturated. Example:
     * Instantiate
     * With a feature manager...
        * Use update_conditions whenever the well configuration changes (Can't be streamed from minknow)
           * You can also pass any extra information that you want to be recorded when saturation is triggered
     * Check results with how_many_saturated or get the raw df (self.saturation_df)

     Example:
     >>> x = SaturationWatcher(extra_columns=['voltage'])
     >>> x.update_conditions(channel_configuration={100: 1}, voltage=55)
     # x receives a saturation event on channel 100 from the feature manager
     >>> x.update_conditions(channel_configuration={100: 2, 200: 2})
     # x receives a saturation event on channel 100 from the feature manager
     >>> x.update_conditions(voltage=66)
     # x receives a saturation event on channel 200 from the feature manager

    >>> x.saturation_df
     channel | well | state     | switched_off_time | voltage
     --------+------+-----------+-------------------+--------
        100  |   1  | saturated |        2.4        |   55
        100  |   2  | saturated |        3.0        |   55
        200  |   2  | saturated |        3.1        |   66

     >>> x.how_many_saturated(well=[1])
     1
     >>> x.how_many_saturated(well=[2])
     2
     >>> x.how_many_saturated(well=[2], voltage=[55])
     1

    """

    def __init__(self, sample_rate: int, extra_columns: Optional[list[str]] = None):
        """Initialise the saturation watcher

        :param sample_rate: int of sample rate. Used to convert sample timings to seconds
        :param extra_columns: Any extra columns that are expected
        """

        self.extra_columns = extra_columns if extra_columns is not None else []
        self.columns = ["channel", "well", "state", "switched_off_time"]
        self.columns.extend(self.extra_columns)

        # Stores information about any extra information provided by update_conditions
        self.extra_info = {}

        self.saturation_df = pd.DataFrame([], columns=self.columns)
        self.states_to_watch = ["saturated"]

        # Stores what the channel configuration should be
        self.channel_configuration = defaultdict(int)

        # Used to convert trigger_time from samples to seconds
        self.sample_rate = float(sample_rate)

    def update_conditions(self, channel_configuration: Optional[dict[int, int]] = None, **kwargs) -> None:
        """Update the conditions that are stored when a saturation channel state occurs

        If kwargs are specified these will get added to the df.

        :param channel_configuration: What new wells channels are going to be in. dict(channel-> well)
        :param kwargs: Any extra information you want saved in the df when saturation happens
        """

        if channel_configuration is not None:
            self.channel_configuration.update(channel_configuration)

        if kwargs:
            for (k, v) in kwargs.items():
                if k not in self.extra_info:
                    self.extra_info[k] = [v]
                else:
                    self.extra_info[k].append(v)

    def execute(self, states: Optional[dict[int, ChannelNotifierState]] = None) -> None:
        if states:
            new_rows = []
            for (channel, state) in states.items():
                if state.state_name in self.states_to_watch:
                    new_item = {
                        "channel": channel,
                        "well": self.channel_configuration[channel],
                        "state": state.state_name,
                        "switched_off_time": state.trigger_time / self.sample_rate,
                    }

                    # Make sure any extra columns are present even if the value hasn't been established
                    new_item.update({k: None for k in self.extra_columns})
                    # Update with any info that we do have
                    new_item.update({k: v[-1] for (k, v) in self.extra_info.items()})

                    new_rows.append(new_item)

            add_df = pd.DataFrame(new_rows, columns=self.saturation_df.columns)

            if self.saturation_df.empty:
                self.saturation_df = add_df
            else:
                self.saturation_df = self.saturation_df.append(add_df, ignore_index=True)

    def how_many_saturated(self, **kwargs) -> int:
        """Return how many channels have currently been saturated given some criteria.

        :param kwargs: Which extra_info to filter on. dict(item->list)
        """
        subset = self.saturation_df

        if kwargs:
            for (key, value) in kwargs.items():
                subset = subset[subset[key].isin(value)]

        return len(subset)
