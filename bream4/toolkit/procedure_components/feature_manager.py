from __future__ import annotations

import inspect  # Used to see what kwargs the functions need
import logging
import threading
import time
from collections import namedtuple
from typing import Any, Callable, Optional

from grpc import RpcError

import bream4.toolkit.procedure_components.command.keystore as keystore
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.command.phase_management import ProtocolPhaseManagement
from bream4.toolkit.procedure_components.data_types import StateMap, TimelineList

ChannelNotifierState = namedtuple("ChannelNotifierState", ["state_name", "well", "trigger_time"])
FeatureInfo = namedtuple("FeatureInfo", ["triggers_on", "on_trigger", "allow_exit"])


class KeystoreNotifier(threading.Thread):
    def __init__(
        self, device: BaseDeviceInterface, notify: Callable[[dict], None], name: str = "bream.protocol_communication"
    ):
        self.device = device
        self.notify = notify
        self.keystream_handle = None
        self.ks_name = name
        self.logger = logging.getLogger(__name__)
        threading.Thread.__init__(self)
        self._run = True

    def run(self) -> None:
        while self._run:
            self.logger.info("Starting KeystoreStreamer")
            try:
                self.keystream_handle = self.device.connection.keystore.watch(names=[self.ks_name], allow_missing=True)
                for msg in self.keystream_handle:
                    if not self._run:
                        return

                    value = keystore.parse_keystore_message(self.ks_name, msg)
                    self.logger.debug("Received keystore message: {}".format(value))

                    self.notify({self.ks_name: value})

            except RpcError as exception:
                code = exception.code()
                self.logger.info("GRPC KeystoreStreamer threw {}".format(str(exception)))

                # We cancelled it so return, or if the acquisition failed/aborted
                # Failed precondition is an unrecoverable error but is most likely from
                # MinKNOW acquisition stopping but not yet stopped.
                if (
                    code == code.CANCELLED
                    or code == code.ABORTED
                    or code == code.FAILED_PRECONDITION
                    or self.device.get_acquisition_status() != self.device.ACQ_PROCESSING
                ):
                    self.logger.info("Keystore notifier terminated")
                    self._run = False
                    return

            except Exception as exception:
                self.logger.info("GRPC KeystoreStreamer threw {}".format(exception))
                self._run = False
                return

    def push_current(self) -> None:
        self.notify({self.ks_name: keystore.get_keystore_message(self.device, self.ks_name)})

    def stop(self) -> None:
        self.logger.info("Asking KeystoreStreamer to stop")

        if self.keystream_handle:
            self.keystream_handle.cancel()
        self._run = False


class ChannelNotifier(threading.Thread):
    """This is the class that listens for any channel states and passes
    them back to the feature manager. This is because watching for
    channel states needs to be on a separate thread

    """

    def __init__(self, device: BaseDeviceInterface, notify: Callable[[dict[int, ChannelNotifierState]], None]):
        """Initialisation of the channel state. This is a subclass of
        `threading.Thread` so it will need a `run()` call

        :param device: MinKNOW device wrapper
        :param notify: A nanodeep to call when a new channel state has been received

        """
        self.device = device
        self.notify = notify
        threading.Thread.__init__(self)
        self.stream_handle = None
        self.logger = logging.getLogger(__name__)
        self.last_time = time.monotonic()  # For debug of minknow channel state loss
        self.channel_states_received = 0
        self._run = True

    def run(self) -> None:
        """Start the thread. This will call the self.notify method passing in
        any channel states it received. If the stream is cancelled a
        CANCELLED/ABORTED exception will be raised and this nanodeep will exit
        and the thread will be completed. If any other RPC error
        happens, retry to stream channel states

        """
        while self._run:
            try:
                self.logger.info("Starting ChannelStateStreamer")
                # Add a heartbeat to make sure we get notified every 30 seconds at least
                self.stream_handle = self.device.get_channel_states_watcher(heartbeat=30)
                for msg in self.stream_handle:
                    if not self._run:
                        return

                    # --- Debug information for minknow channel states dropped ---
                    self.channel_states_received += len(msg.channel_states)
                    if self.last_time < time.monotonic() - 60:
                        self.logger.info(
                            "{} channel states in the last minute".format(str(self.channel_states_received))
                        )
                        self.channel_states_received = 0
                        self.last_time = time.monotonic()
                    # ------------------------------------------------------------
                    state_map = {
                        state.channel: ChannelNotifierState(state.state_name, state.config.well, state.trigger_time)
                        for state in msg.channel_states
                    }
                    self.notify(state_map)
            except RpcError as exception:
                code = exception.code()
                self.logger.info("GRPC ChannelStateStreamer threw {}".format(str(exception)))

                # We cancelled it so return, or if the acquisition failed/aborted
                # Failed precondition is an unrecoverable error but is most likely from
                # MinKNOW acquisition stopping but not yet stopped.
                if (
                    code == code.CANCELLED
                    or code == code.ABORTED
                    or code == code.FAILED_PRECONDITION
                    or self.device.get_acquisition_status() != self.device.ACQ_PROCESSING
                ):
                    self.logger.info("ChannelStateStreamer terminated")
                    self._run = False
                    return

            except Exception as exception:
                self.logger.info("GRPC ChannelStateStreamer threw {}".format(exception))
                self._run = False
                return

    def stop(self) -> None:
        self.logger.info("Asking ChannelStateStreamer to stop")

        if self.stream_handle:
            self.stream_handle.cancel()
        self._run = False


class FeatureManager(object):
    """
    The Feature Manager is where the scheduling takes place for each of the features

    For the setup you can do something similar to: ::

     with FeatureManager(device) as fm:
         fm.start(some_feature)
         # Do some cool stuff where the features automatically do stuff
         fm.stop_all()
    """

    def __init__(
        self,
        device: BaseDeviceInterface,
        phase_management: Optional[ProtocolPhaseManagement] = None,
        add: Optional[list[tuple[float, Any]]] = None,
    ):
        """Initialisation of the feature manager optionally takes an add
        parameter This is to add any features to be scheduled in the
        timeline. This is useful if you want to save a state from a
        previous feature manager and apss it into a new one. The
        expectation is that this is passed in from the `stop_all`
        method

        The phase_management, if present, allows the new mux_scan/pause actions
        that appear in Core 4.4

        :param device: MinKNOW device wrapper
        :param phase_management: bream4.toolkit.procedure_components.command.phase_management.ProtocolPhaseManagement
        :param add: list of tuples (time_to_run, feature)

        """
        self.device = device
        self.features = {}
        self.wait_condition = threading.Condition()  # Used as a lock/notifier on features
        self.timeline = TimelineList(self.wait_condition)
        self.channel_state_map = StateMap()
        self.keystore_state_map = StateMap()
        self._main_thread: Optional[threading.Thread] = None  # Started on enter
        self.do_run = threading.Event()
        self._channel_notifier: Optional[ChannelNotifier] = None
        self._channel_update = []
        self._keystore_notifier: Optional[KeystoreNotifier] = None
        self._keystore_update = []

        if phase_management:
            self._pause_update = []
            self._mux_scan_update = []

        self._pause_request = False  # Set to true when an action has come in
        self._mux_scan_request = False  # Set to true when an action has come in
        self.phase_management = phase_management

        self.sleep_event = threading.Event()

        self.logger = logging.getLogger(__name__)

        # If we want to restore the timeline of certain features, store it here
        # This under the expectation that a `start` of the same feature object is going to happen
        # So this will eventually be emptied
        self.delayed_add = add

        self.logger.info("Scheduling restoring events: {}".format(add))

    def __enter__(self):
        self.features = {}

        # do_run is the flag that communicates with the main thread loop (run method)
        # Once the thread is instantiated it will set this flag.
        # When this flag is set to false it will exit
        self.do_run.clear()

        self._main_thread = threading.Thread(target=self.run)
        self._main_thread.start()
        self.do_run.wait()  # Wait for the main thread to signal its going

        # Start the channel state streamer
        # Pass the channel_state_map which all watching channels will be notified with
        self._channel_notifier = ChannelNotifier(self.device, self.channel_state_map.update)
        self._channel_notifier.start()
        self._keystore_notifier = KeystoreNotifier(self.device, self.keystore_state_map.update)
        self._keystore_notifier.start()

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        # Exit the thread by stopping the run and notifying if it's
        # blocking for a future timeline call

        start_thread_count = threading.active_count()

        # Remove listeners to phase manager actions. Phase manager doesn't error if action isn't registered
        if self.phase_management:
            self.phase_management.unsubscribe_action("PAUSE", self._pause)
            self.phase_management.unsubscribe_action("TRIGGER_MUX_SCAN", self._trigger_mux_scan)

        # Trigger a notification so that the run thread will awaken if blocked
        # then exit
        with self.wait_condition:
            self.do_run.clear()
            self.wait_condition.notify()

        if self._main_thread:
            self._main_thread.join()

        # And trigger the stopping of the channel streamer
        if self._channel_notifier:
            self._channel_notifier.stop()
            self._channel_notifier.join()
        if self._keystore_notifier:
            self._keystore_notifier.stop()
            self._keystore_notifier.join()

        # Add a stop_all here to ensure that if the user forgot to call stop_all in the
        # with block, the features are still stopped
        self.stop_all()

        self.exit_all()

        end_thread_count = threading.active_count()
        self.logger.info(
            "Exiting FeatureManager with %s threads left. %s before exit",
            end_thread_count,
            start_thread_count,
        )

    def _pause(self) -> None:
        with self.wait_condition:
            self._pause_request = True
            self.wait_condition.notify()

    def _trigger_mux_scan(self) -> None:
        with self.wait_condition:
            self._mux_scan_request = True
            self.wait_condition.notify()

    def sleep(self, amount: float) -> None:
        """Use instead of time.sleep in main scripts so that features
        can exit the feature manager with block. Features still run whilst
        in this sleep. If the main thread dies this will also throw an exception

        :param amount: How long to sleep (in seconds)
        """

        end_time = time.monotonic() + amount

        while True:
            entry_time = time.monotonic()
            self.sleep_event.wait(min(1, end_time - entry_time))

            if self.sleep_event.is_set() or entry_time > end_time:
                return
            if self._main_thread and not self._main_thread.is_alive():
                raise RuntimeError("FeatureManager main thread died")

    def exit_all(self) -> None:
        """Call all registered features :code:`exit` call if applicable"""
        self.logger.info("Exiting all features")

        with self.wait_condition:
            for feature in self.features:
                self._maybe_run_function(feature, "exit")

    def resume_all(self) -> None:
        """Signal all currently running features to resume."""
        self.logger.info("Resuming all features")

        with self.wait_condition:
            for feature in self.features:
                self._maybe_run_function(feature, "resume")

    def stop_all(self, save=None) -> list[tuple[float, Any]]:
        """Signal all currently running features to stop

        If features are specified to be saved, the output of this can be fed into
        another feature_managers init's add parameter

        :param save: A list of feature objects of who's timelines to return. If not present, all will be returned.
        :returns: A tuple of (time_to_run, feature) for each feature requested
        """
        current_timeline = self.timeline.get_copy()

        self.logger.info("Stopping all features. Timeline looks like: %s", current_timeline)

        saved_timeline = []

        if save:
            for (scheduled_time, feature) in current_timeline:
                if feature in save:
                    saved_timeline.append((scheduled_time, feature))
        else:
            saved_timeline = current_timeline

        with self.wait_condition:
            for feature in self.features:
                self._maybe_run_function(feature, "stop")

        return saved_timeline

    def run(self) -> None:
        """This nanodeep runs in a separate thread when we enter a with block: ::

         with FeatureManager(...) as fm:
             # this run() method has now been called

        This will constantly drain 2 event queues:
          * :code:`worker_queue` which is a list of events that the channel states get fed into.
            This will run any events who have been set to trigger with the `channel_update` property
          * :code:`timeline` which is a list of features and when they should be next executed

        Then it will wait for either of these queues to be populated
        """

        # wait() releases the lock and reobtains it on a notify so this isn't held indefinitely
        with self.wait_condition:

            self.do_run.set()  # Signal that the thread has started

            while self.do_run.is_set():

                if self._pause_request:
                    for feature in self._pause_update:
                        self._execute_feature(feature, phase_management=self.phase_management)
                    self._pause_request = False

                if self._mux_scan_request:
                    for feature in self._mux_scan_update:
                        self._execute_feature(feature, phase_management=self.phase_management)
                    self._mux_scan_request = False

                # Drain the channel state updates
                channel_states = self.channel_state_map.pop_all()

                if channel_states:
                    # We don't need to reschedule this etc so just run it
                    for feature in self._channel_update:
                        self._execute_feature(feature, states=channel_states)

                keystore_states = self.keystore_state_map.pop_all()
                if keystore_states:
                    for feature in self._keystore_update:
                        self._execute_feature(feature, keystore=keystore_states)

                # Then if there are any scheduled to execute now then do it
                execute_up_to = time.monotonic()
                while self.timeline and self.timeline.peek()[0] <= execute_up_to:
                    # Wrap it in an execute so it is rescheduled as it is a timeline feature
                    self.execute_feature(self.timeline.pop()[1])

                # Then work out how long to rest for:

                # If we are reliant on keystore or channel updates make sure for just a small rest period
                if self._channel_update or self._keystore_update:
                    self.wait_condition.wait(0.05)  # Channel state map/keystore map will need to be propagated
                # If we aren't using notifications, and we have a timeline, use that
                elif self.timeline:
                    self.wait_condition.wait(self.timeline.peek()[0] - time.monotonic())
                else:
                    # Shouldn't be in here very often. It should just be when first entering a with block
                    # If we're hitting it more often, it means there are no features scheduled to run
                    # which seems bizarre
                    self.wait_condition.wait(1)

    def start(
        self,
        feature,
        triggers_on: Optional[list[tuple]] = None,
        on_trigger: Optional[str] = None,
        allow_exit_manager_sleep: bool = False,
    ) -> None:
        """This will schedule a feature to be started.
        :code:`triggers` are what events should cause this event to be run. Examples include:
          * :code:`interval(1)` would trigger this feature every 1 second
          * :code:`channel_update()` would trigger this feature every time there is a new channel_state
          * :code:`keystore_update()` would trigger this features every time there is a keystore update
        :code:`on_trigger` is a single value on what should happen when this feature is triggered.
        For example:
          * :code:`just_run` means just run this feature. The :code:`execute()` method will be
            called without any other feautres been stopped etc
          * :code:`pause_all` would call every other functions :code:`pause` method and would
            prevent them from running until the feature has executed. Then they would all have
            their :code:`resume` methods called
          * :code:`restart_all` would call every other functions :code:`stop` method and
            :code:`reset` method called. Then the feature would execute.
            Then every feature will have their :code:`resume` method called
        :param feature: feature to schedule
        :param triggers_on: list of values on when to run the feature
        :param on_trigger: What would happen when the feature was executed
        :param allow_exit_manager_sleep: Whether this feature has the ability to
                                         cancel a pending feature_manager.sleep call
        """
        if not triggers_on:
            raise RuntimeError("No triggers specified for feature {}".format(feature))

        on_trigger = on_trigger or self.just_run

        # Make sure that if you're adding a feature you do it with the condition to not
        # interfere with the run method
        with self.wait_condition:
            # Add to our cache so we can reschedule stuff later
            self.features[feature] = FeatureInfo(triggers_on, on_trigger, allow_exit_manager_sleep)

            # If waiting to schedule it then add it
            if self.delayed_add:
                # For every [(time, feature)] that we saved in delayed_add,
                # Find it and move them to the timeline

                # Find indices of the feature we want to add. (In reverse as removing during iterating)
                feature_indices = [i for i, x in enumerate(self.delayed_add) if x[1] == feature][::-1]

                # If we have a delayed_add to schedule for this feature, then do it
                if feature_indices:
                    self._maybe_run_function(feature, "resume")
                    # Add all to timeline
                    for index in feature_indices:
                        # Remove (time, feature) from our save list
                        (time_scheduled, _) = self.delayed_add.pop(index)
                        # And insert into our timeline
                        self.timeline.insert_ordered((time_scheduled, feature))
                    return

            # Or just schedule it but make sure it is resumed
            self._maybe_run_function(feature, "resume")
            self._schedule_feature(feature)

    def _maybe_run_function(self, obj, function_name, **kwargs) -> None:
        # If nanodeep exists, call it and pass in any keyword arguments asked for
        # This is so if a feature doesn't have a resume/pause nanodeep then just dont call it
        # to make it simpler to create features

        # Python introspection - Give the nanodeep only the keywords it is expecting
        # This means adding one later doesn't mean updating every single feature

        if hasattr(obj, function_name):
            function = getattr(obj, function_name)
            info = inspect.getfullargspec(function)

            if info.varkw:
                # Asked for all, so give all
                function(**kwargs)
            else:
                function(**{k: v for k, v in kwargs.items() if k in info.args})

    def _trigger_sleep_exit(self) -> None:
        self.logger.info("Triggering sleep exit")
        self.sleep_event.set()
        self.logger.info("Notified")

    def _execute_feature(self, feature_to_run, per_channel_interrupt=None, **kwargs) -> None:
        # per_channel_interrupt will get passed to the stop/resume/start
        # It's up to the functions if they want to do something with it
        feature_info = self.features[feature_to_run]

        if feature_info.allow_exit:
            kwargs["exit_manager_sleep"] = self._trigger_sleep_exit

        if feature_info.on_trigger == self.pause_all:
            feature_list = list(self.features.keys())
            for feature in feature_list:
                if feature != feature_to_run:
                    self._maybe_run_function(feature, "stop", per_channel_interrupt=per_channel_interrupt)
            feature_to_run.execute(**kwargs)
            for feature in feature_list[::-1]:
                if feature != feature_to_run:
                    self._maybe_run_function(feature, "resume", per_channel_interrupt=per_channel_interrupt)

        elif feature_info.on_trigger == self.reset_all:
            feature_list = list(self.features.keys())
            for feature in feature_list:
                if feature != feature_to_run:
                    self._maybe_run_function(feature, "stop", per_channel_interrupt=per_channel_interrupt)
                    self._maybe_run_function(feature, "reset", per_channel_interrupt=per_channel_interrupt)
            feature_to_run.execute(**kwargs)
            for feature in feature_list[::-1]:
                if feature != feature_to_run:
                    self._maybe_run_function(feature, "resume", per_channel_interrupt=per_channel_interrupt)

        elif feature_info.on_trigger == self.just_run:
            feature_to_run.execute(**kwargs)

        else:
            raise RuntimeError(
                "trigger {} for feature {} is not a valid trigger".format(feature_info.on_trigger, feature_to_run)
            )

    def execute_feature(self, feature, **kwargs) -> None:
        """Run the feature and maybe reschedule it on the timeline

        :param feature: Feature object to run
        """
        # If it's a scheduled feature, reschedule it
        self._schedule_feature(feature)

        # For now, we don't need any special execute behaviours so just run the feature
        self._execute_feature(feature, **kwargs)

    def _schedule_feature(self, feature) -> None:
        """Set the feature to be scheduled. If it has interval triggers, add
        it to the correct place in the timeline. If it has channel_update
        triggers, add it to be notified when there is a new channel_state

        :param feature: Feature to be scheduled
        """
        # Find registered info about feature:
        info = self.features[feature]

        for trigger in info[0]:
            # If a scheduled feature, schedule it
            if trigger[0] == "timeline":
                interval = trigger[1]
                self.timeline.insert_ordered((interval + time.monotonic(), feature))
            elif trigger[0] == "channel_update" and feature not in self._channel_update:
                self._channel_update.append(feature)
            elif trigger[0] == "keystore_update" and feature not in self._keystore_update:
                self._keystore_update.append(feature)
                # With keystore watchers, make sure that the feature is aware of any current states
                if self._keystore_notifier:
                    self._keystore_notifier.push_current()
            elif trigger[0] == "pause_action":
                if self.phase_management and feature not in self._pause_update:
                    self._pause_update.append(feature)
                    # Shouldn't need to kickstart message as pause isn't possible until something has subscribed to it
                    self.phase_management.subscribe_action("PAUSE", self._pause)
                elif not self.phase_management:
                    raise RuntimeError("Action watching requested nut no phase_management passed to feature_manager")
            elif trigger[0] == "mux_scan_action":
                if self.phase_management and feature not in self._mux_scan_update:
                    self._mux_scan_update.append(feature)
                    self.phase_management.subscribe_action("TRIGGER_MUX_SCAN", self._trigger_mux_scan)
                elif not self.phase_management:
                    raise RuntimeError("Action watching requested nut no phase_management passed to feature_manager")

    @classmethod
    def interval(cls, pause: float):
        """This is a parameter for the `start` method's :code:`triggers` parameter.
        Example: ::

         feature_manager.start(some_feature, triggers_on=[ FeatureManager.interval(10) ])

        Will schedule some_feature to run every 10 seconds

        :param cls: Class-Method
        :param pause: Eevry pause seconds run the nanodeep
        :returns: A tuple to be passed to the :code:`triggers_on` argument
        """
        return ("timeline", pause)

    @classmethod
    def channel_update(cls):
        """This is a parameter for the `start` method's :code:`triggers_on` parameter.
        Example: ::

         feature_manager.start(some_feature, triggers_on=[ FeatureManager.channel_update() ])

        Will schedule some_feature to run every time there is a new channel state

        :param cls: Class-Method
        :returns: A tuple to be passed to the :code:`triggers_on` argument
        """

        return ("channel_update",)

    @classmethod
    def keystore_update(cls):
        """This is a parameter for the `start` method's :code:`triggers_on` parameter.
        Example: ::

         feature_manager.start(some_feature, triggers_on=[ FeatureManager.keystore_update() ])

        Will schedule some_feature to run every time there is a new keystore notification

        :param cls: Class-Method
        :returns: A tuple to be passed to the :code:`triggers_on` argument
        """

        return ("keystore_update",)

    @classmethod
    def pause_action(cls):
        """This is a parameter for the `start` method's :code:`triggers_on` parameter.
        Example: ::

         feature_manager.start(some_feature, triggers_on=[ FeatureManager.pause_action() ])

        Will schedule some_feature to run every time there is new pause request from the UI (Core >= 4.4)

        :param cls: Class-Method
        :returns: A tuple to be passed to the :code:`triggers_on` argument
        """

        return ("pause_action",)

    @classmethod
    def mux_scan_action(cls):
        """This is a parameter for the `start` method's :code:`triggers_on` parameter.
        Example: ::

         feature_manager.start(some_feature, triggers_on=[ FeatureManager.mux_scan_action() ])

        Will schedule some_feature to run every time there is new mux scan request from the UI (Core >= 4.4)

        :param cls: Class-Method
        :returns: A tuple to be passed to the :code:`triggers_on` argument
        """

        return ("mux_scan_action",)

    @property
    def reset_all(cls):
        """This is a parameter for the `start` method's :code:`on_trigger` parameter.
        Example: ::

         feature_manager.start(some_feature, triggers_on=[ FeatureManager.interval(10) ],
                                             on_trigger=FeatureManager.reset_all)

        Will schedule some_feature to run every 10 seconds. When this
        feature runs, all other functions will have their :code:`stop`
        and :code:`reset` methods called. Then
        :code:`some_feature.execute()` will be called. Then all other
        functions :code:`resume` methods will be called
        *No other functions will be in their :code:`execute` methods at the same time.*

        :param cls: Class-Method
        :returns: A string to be passed to the :code:`on_trigger` argument

        """

        return "reset all"

    @property
    def pause_all(cls):
        """This is a parameter for the `start` method's :code:`on_trigger` parameter.
        Example: ::

         feature_manager.start(some_feature, triggers_on=[ FeatureManager.interval(10) ],
                                             on_trigger=FeatureManager.pause_all)

        Will schedule some_feature to run every 10 seconds. When this
        feature runs, all other functions will have their :code:`stop`
        methods called. Then :code:`some_feature.execute()` will be called. Then all other
        functions :code:`resume` methods will be called
        *No other functions will be in their :code:`execute` methods at the same time.*

        :param cls: Class-Method
        :returns: A string to be passed to the :code:`on_trigger` argument

        """

        return "pause all"

    @property
    def just_run(cls):
        """This is a parameter for the `start` method's :code:`on_trigger` parameter.
        Example: ::

         feature_manager.start(some_feature, triggers_on=[ FeatureManager.interval(10) ],
                                             on_trigger=FeatureManager.just_run)

        Will schedule some_feature to run every 10 seconds. When this
        feature runs, no features will get stopped or reset etc. However, the same applies:
        *No other functions will be in their :code:`execute` methods at the same time.*

        :param cls: Class-Method
        :returns: A string to be passed to the :code:`on_trigger` argument

        """

        return "just run"
