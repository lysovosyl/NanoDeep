import logging
from typing import Callable, Optional

import bream4.pb.bream.protocol_communication_pb2 as protocol_communication
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.command.keystore import set_state
from bream4.toolkit.procedure_components.command.phase_management import ProtocolPhaseManagement

KEYSTORE_KEY = "bream.protocol_communication"


class MuxScanTrigger:
    def __init__(self, device: BaseDeviceInterface):
        self.device = device
        self.logger = logging.getLogger(__name__)
        self.run_id = device.get_run_id()
        self.enable = False

    def execute(
        self,
        keystore=None,
        phase_management: Optional[ProtocolPhaseManagement] = None,
        exit_manager_sleep: Optional[Callable[[], None]] = None,
    ):
        if keystore and KEYSTORE_KEY in keystore:
            protocol_info: protocol_communication.ProtocolData = keystore[KEYSTORE_KEY].data[self.run_id]
            self.logger.debug("Received keystore message for current run: {}".format(protocol_info))

            if protocol_info.HasField("mux_scan"):
                set_state(self.device, "PREPARING_FOR_MUX_SCAN", clear_trigger=True)
                self.enable = True

                if exit_manager_sleep is not None:
                    self.logger.info("Triggering exit of feature manager")
                    exit_manager_sleep()
                else:
                    self.logger.warning("No exit handler provided")

        elif keystore is None or phase_management:
            self.enable = True

            if phase_management:
                phase_management.set_phase("PREPARING_FOR_MUX_SCAN")

            # Not a keystore update. Must have been a time trigger or new UI request
            if exit_manager_sleep is not None:
                self.logger.info("Triggering exit of feature manager")
                exit_manager_sleep()
            else:
                self.logger.warning("No exit handler provided")
