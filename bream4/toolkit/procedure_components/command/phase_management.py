from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, defaultdict
from collections.abc import Generator
from typing import Callable, Optional

import grpc
from grpc import RpcError
from minknow_api import protocol_service

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface


class ProtocolPhaseManagement:
    """Class to manage the GRPC ProtocolPhaseManagement.

    Only one call to the underlying stream is allowed, so only one object at a time is allowed.
    You must call .stop() and then retrieve a new object.

    Usage is as follows:

    >>> def pause_request():
    >>>     logger.info("Pause action received")
    >>> def resume_request():
    >>>     logger.info("Resume action received")
    >>> def mux_request():
    >>>     logger.info("Mux scan action received")
    >>> management = ProtocolPhaseManagement(device)
    >>> management.subscribe_action("PAUSE", pause_request)
    >>> management.subscribe_action("RESUME", resume_request)
    >>> management.subscribe_action("TRIGGER_MUX_SCAN", pause_request)
    ...
    >>> management.stop()

    Callbacks get called when UI (Or others) send a pause/resume/mux_scan to minknow
    """

    def __init__(self, device: BaseDeviceInterface):
        self._device = device
        self._logger = logging.getLogger(__name__)
        self._stop = False

        self.phase_history: OrderedDict[float, str] = OrderedDict()
        self._condition = threading.Condition()  # Default is RLock
        self._request_stream: list[protocol_service.ProtocolPhaseManagementRequest] = []
        self._response_stream: Optional[grpc.RpcContext] = None
        self._subscriptions: defaultdict[str, list[Callable]] = defaultdict(list)

        # Set as a daemon thread as this is a blocking call until we receive a message
        # If this wasn't true, and an exception was raised, python wouldn't exit
        self._stream = threading.Thread(target=self._phase_response_stream, daemon=True)
        self._stream.start()

    def subscribe_action(self, action: str, callback: Callable[[], None]) -> None:
        """Whenever 'action' happens, call the callback passed in.
        If the same callback is registered for the same action multiple times,
        only one callback() will happen.

        action must be a valid action. Currently {"PAUSE", "RESUME", "TRIGGER_MUX_SCAN"}
        are supported.
        """
        try:
            # Try to get the enum for the action to make sure it exists.
            # If we didn't the user could add arbitrary actions and they'd never get called,
            # which is probably not what they want
            protocol_service.Action.Value("ACTION_" + action)
        except ValueError:
            raise ValueError(
                "Invalid action %s to subscribe to. Options are %s" % (action, protocol_service.Action.keys())
            )

        with self._condition:
            if callback not in self._subscriptions[action]:
                self._subscriptions[action].append(callback)
                self._update_capabilities()

    def unsubscribe_action(self, action: str, callback: Callable[[], None]) -> None:
        """Paired with the subscribe_action call to unsubscribe a callable from an action

        If the callback isn't registered with this action, no change is made and no error is raised
        """
        with self._condition:
            if callback in self._subscriptions[action]:
                self._subscriptions[action].remove(callback)
                self._update_capabilities()

    def stop(self) -> None:
        """This cancels all running streams and waits for threads to finish.

        Once this method is called, the object is spent, and a new one must be constructed
        to restart listening to the phases again
        """
        with self._condition:
            self._stop = True
            if self._response_stream:
                self._response_stream.cancel()
            else:
                self._logger.warning("Response stream not present when cancelling")
            self._condition.notify()
            self._stream.join()

    def get_phase(self) -> str:
        """Returns the last phase that was asked to be transitioned to.
        If none, "UNKNOWN" is returned as that is the default phase
        """
        if not self.phase_history:
            return "UNKNOWN"
        return next(reversed(self.phase_history.values()))

    def set_phase(self, phase: str) -> None:
        """Send a new phase to minknow. This must be a valid phase according to the protobuf
        As of writing, the valid ones are:
        UNKNOWN; INITIALISING; SEQUENCING; PREPARING_FOR_MUX_SCAN; MUX_SCAN; PAUSED; PAUSING; RESUMING
        """

        expanded_phase = phase if phase.startswith("PHASE_") else "PHASE_" + phase

        with self._condition:
            try:
                _phase = protocol_service.ProtocolPhase.Value(expanded_phase)
                msg = protocol_service.ProtocolPhaseManagementRequest(phase=_phase)
                self._request_stream.append(msg)
                self.phase_history[time.monotonic()] = phase
                self._condition.notify()

            except ValueError:
                raise ValueError(
                    "Phase %s is an invalid phase. Options are %s"
                    % (expanded_phase, protocol_service.ProtocolPhase.keys())
                )

    def _update_capabilities(self) -> None:
        with self._condition:
            capabilities = protocol_service.ProtocolPhaseManagementRequest.Capabilities(
                can_pause=len(self._subscriptions["PAUSE"]) > 0,
                can_trigger_mux_scan=len(self._subscriptions["TRIGGER_MUX_SCAN"]) > 0,
            )
            request = protocol_service.ProtocolPhaseManagementRequest(set_capabilities=capabilities)
            self._request_stream.append(request)
            self._condition.notify()

    def _phase_request_stream(self) -> Generator[str, None, None]:
        """This is the stream going TO minknow.
        This will advertise whether we can act on pause/resume/mux scan trigger requests (Based on subscriptions)
        And also to send updated phase (SEQUENCING/PREPARING_FOR_MUX_SCAN/etc)
        """
        # Should yield a StopIteration which should break the stream
        while not self._stop:
            value = None
            with self._condition:
                if self._request_stream:
                    value = self._request_stream.pop(0)
                else:
                    self._condition.wait()  # Wait to be notified of a change in situation
            if value:
                yield value

    def _phase_response_stream(self) -> None:
        """This is the stream coming FROM minknow.
        This will be giving us actions to perform such as PAUSE/RESUME/TRIGGER_MUX_SCAN
        """
        with self._condition:
            self._response_stream = self._device.connection.protocol.protocol_phase_management(
                self._phase_request_stream()
            )

        # Go in a loop because the stream can occasionally blip out
        while not self._stop:
            try:
                for msg in self._response_stream:
                    action_raw = protocol_service.Action.Name(msg.action)  # Action == 'ACTION_NONE' etc
                    action = action_raw.split("ACTION_")[1]

                    for callback in self._subscriptions[action]:
                        callback()

            except RpcError as exception:
                code = exception.code()
                if code in {code.CANCELLED, code.ABORTED, code.FAILED_PRECONDITION} or self._stop:
                    return  # Intentional ending of stream
