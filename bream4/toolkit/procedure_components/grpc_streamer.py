import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

import grpc

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface


class GrpcStreamer(threading.Thread, ABC):
    """Implements a streaming thread that can be stopped and will restart
    with temporary grpc blips. If acquisition is cancelled etc then the stream
    will not be restarted.

    Entrypoints:
    def get_stream to return a stream handle
    def process_item(item) to handle items in the stream
    """

    def __init__(self, device: BaseDeviceInterface):
        super().__init__()

        self.device = device
        self.logger = logging.getLogger(self.__class__.__name__)
        self._keep_going = True
        self._stream = None

    @abstractmethod
    def get_stream(self) -> grpc.Call:
        pass

    @abstractmethod
    def process_item(self, item: Any) -> None:
        # Can't type this as it could be any response object
        pass

    def run(self) -> None:
        while self._keep_going:
            self.logger.debug("Starting GRPC Streamer")
            try:
                self._stream = self.get_stream()
                for msg in self._stream:

                    # Early exit instead of waiting for stream to finish
                    if not self._keep_going:
                        return
                    self.process_item(msg)

            except grpc.RpcError as exception:
                code = exception.code()
                self.logger.info("GRPC Streamer threw {}".format(str(exception)))

                # We cancelled it so return, or if the acquisition failed/aborted
                # Failed precondition is an unrecoverable error but is most likely from
                # MinKNOW acquisition stopping but not yet stopped.
                if (
                    code == code.CANCELLED
                    or code == code.ABORTED
                    or code == code.FAILED_PRECONDITION
                    or self.device.get_acquisition_status() != self.device.ACQ_PROCESSING
                ):
                    self.logger.info("GRPC Streamer terminated")
                    self._keep_going = False
                    return

            except Exception as exception:
                self.logger.info("GRPC Streamer threw {}".format(exception))
                self._keep_going = False
                return

    def stop(self) -> None:
        self.logger.debug("Asking GRPC Streamer to stop")

        if self._stream:
            self._stream.cancel()

        self._keep_going = False
