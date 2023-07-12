from __future__ import annotations

import logging
import os
from typing import Callable, Optional, Union

from grpc import RpcError

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.feature_manager import ChannelNotifierState


class ChannelDisable(object):
    def __init__(self, config: dict, device: BaseDeviceInterface):
        self.states_to_disable = config.get("states_to_disable", [])
        self.snap_shot = {}
        self.device = device
        self.logger = logging.getLogger(__name__)

        self.csv_output = config.get("csv_output", False)
        self.lines_to_log = ["flow_cell_id,experiment_run_time,channel,mux\n"]

        if self.csv_output:
            self.sample_rate = self.device.get_sample_rate()
            self.flow_cell_id = self.device.get_flow_cell_id()
            try:
                self.report_path = os.path.join(self.device.get_protocol_output_path(), "cool_down_data.csv")
            except RpcError:
                # Protocol not running. Fall back to logs directory
                filename = "bream_{}_{}_{}_cool_down_data.csv".format(
                    self.device.get_flow_cell_id(),
                    self.device.get_sample_id(),
                    self.device.get_run_id(),
                )
                self.report_path = os.path.join(self.device.get_logs_directory(), filename)

        self.stop_feature_manage_enabled = config.get("stop_feature_manage_enabled", False)
        self.min_channel_threshold = config.get("min_channel_threshold", 0)
        self.active_states = config.get("active_states", [])
        self.active_channel_count = device.channel_count

    def get_channel_state_snap_shot(self) -> dict[int, Union[int, str]]:
        return self.snap_shot.copy()

    def exit(self) -> None:
        if self.csv_output:
            with open(self.report_path, "a") as lg:
                for line in self.lines_to_log:
                    lg.write(line)
        self.lines_to_log = []

    def execute(
        self,
        states: Optional[dict[int, ChannelNotifierState]] = None,
        exit_manager_sleep: Optional[Callable[[], None]] = None,
    ) -> None:
        if states:
            well_none = self.device.disconnected_well
            to_disable = {}
            snap_shot = {}

            for (channel, state) in states.items():
                if state.state_name in self.states_to_disable:
                    to_disable[channel] = well_none
                snap_shot[channel] = state.state_name
            self.snap_shot.update(snap_shot)
            if to_disable:
                self.logger.info("Disabling channels: {}".format(list(to_disable.keys())))
                self.device.set_channels_to_well(to_disable)
                self.device.lock_channel_states(channels=list(to_disable.keys()), state_name="locked")

            if self.csv_output and to_disable:
                channel_config = self.device.get_channel_configuration(1, self.device.channel_count)
                run_time = self.device.get_current_sample_number() / self.sample_rate

                for channel in to_disable:
                    line = "{},{},{},{}\n".format(
                        self.flow_cell_id,
                        run_time,
                        channel,
                        channel_config[channel].well,
                    )
                    self.lines_to_log.append(line)

            if self.stop_feature_manage_enabled:
                if exit_manager_sleep is None:
                    raise RuntimeError("Exit nanodeep not passed, but requested")
                active_channel_count = len(
                    [x for x, v in self.get_channel_state_snap_shot().items() if v in self.active_states]
                )
                self.logger.debug("active_channels = {}".format(active_channel_count))
                if active_channel_count < self.min_channel_threshold:
                    exit_manager_sleep()
