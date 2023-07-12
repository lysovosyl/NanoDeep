import logging
import threading
from typing import Optional

from grpc import RpcError

import bream4.pb.bream.protocol_communication_pb2 as protocol_communication
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.command.keystore import set_state, wait_for_protocol_trigger
from bream4.toolkit.procedure_components.command.phase_management import ProtocolPhaseManagement

KEYSTORE_KEY = "bream.protocol_communication"


class ExperimentPauser(object):
    def __init__(self, device: BaseDeviceInterface):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.run_id = device.get_run_id()

    def execute(self, keystore=None, phase_management: Optional[ProtocolPhaseManagement] = None) -> None:
        if phase_management:
            old_phase = phase_management.get_phase()
            phase_management.set_phase("PAUSING")

            # Save all the old state
            bias_voltage = self.device.get_bias_voltage()
            channel_config = self.device.get_channel_configuration()

            # Set bias voltage to 0 and move all wells to their disconnected state
            self.device.set_bias_voltage(0)
            self.device.set_all_channels_to_well(self.device.disconnected_well)

            _barrier = threading.Event()

            def resume():
                phase_management.set_phase("RESUMING")

                self.device.set_channels_to_well({channel: item.well for (channel, item) in channel_config.items()})
                self.device.set_bias_voltage(bias_voltage)

                phase_management.set_phase(old_phase)
                _barrier.set()

            phase_management.set_phase("PAUSED")
            phase_management.subscribe_action("RESUME", resume)
            _barrier.wait()

            # Finished, remove listener
            phase_management.unsubscribe_action("RESUME", resume)

        if keystore and KEYSTORE_KEY in keystore:  # DeprecationWarning: Will be removed in next major bump
            protocol_info: protocol_communication.ProtocolData = keystore[KEYSTORE_KEY].data[self.run_id]

            if protocol_info.HasField("pause"):
                self.logger.info("Pause triggered")

                set_state(self.device, "PAUSING", clear_trigger=True)

                # Save all the old state
                bias_voltage = self.device.get_bias_voltage()
                channel_config = self.device.get_channel_configuration()

                # Set bias voltage to 0 and move all wells to their disconnected state
                self.device.set_bias_voltage(0)
                self.device.set_all_channels_to_well(self.device.disconnected_well)

                # Tell the UI we are now paused and block for the resume
                set_state(self.device, "PAUSED")

                try:
                    wait_for_protocol_trigger(self.device, "resume")
                except RpcError as err:
                    self.logger.info("Received {} so continuing experiment".format(err))
                except Exception as err:
                    self.logger.info("Received {} so continuing experiment".format(err))

                set_state(self.device, "RESUMING", clear_trigger=True)

                self.device.set_channels_to_well({channel: item.well for (channel, item) in channel_config.items()})
                self.device.set_bias_voltage(bias_voltage)

                set_state(self.device, "RUNNING_SEQUENCING")

                self.logger.info("Successfully resumed")
