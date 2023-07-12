from __future__ import annotations

import abc
import json
import logging
import os
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from typing import Optional, Union

import minknow_api.manager
from google.protobuf import json_format
from google.protobuf.duration_pb2 import Duration
from google.protobuf.wrappers_pb2 import FloatValue, Int32Value, UInt32Value
from minknow_api import Connection, acquisition_service, analysis_configuration_service, data_service, protocol_service

try:
    from minknow_api.production_service import WriteFlowcellDataRequest
except ImportError:  # CORE-4363
    pass

import bream4.legacy.device_interfaces.devices.base_device as legacy_base
from bream4.pb.bream.experiment_estimated_end_pb2 import Experiment
from bream4.toolkit.pings.ping_dumping import ping_to_str
from bream4.toolkit.procedure_components.dict_utils import recursive_merge
from bream4.utility.logging_utils import attach_device_to_logger

INTERNAL_MINKNOW_TAGS = ["unclassed", "transition", "mux_uncertain"]


def get_env_grpc_port(default: int = 8000) -> int:
    """
    When running against minknow, an environmental variable is created that
    holds the correct port the bream should attempt to connect to in it. This
    nanodeep attempts to retrieve this port.

    :param default: The port number to be returned if the port call fails
    :type default: int
    :return: The port number
    :rtype: int
    """
    grpc_port = os.getenv("MINKNOW_RPC_PORT_SECURE", default)
    try:
        return int(grpc_port)
    except ValueError:
        msg = "Environment variable MINKNOW_RPC_PORT_SECURE contains an" " invalid value ({0})".format(grpc_port)
        raise RuntimeError(msg)


# TODO: unsure of name or if this is the correct place, but classes in the
# devices subdirectory should inherit from this
class BaseDeviceInterface(abc.ABC):
    """
    This class wraps the minknow command APIs for its different hardware and
    provides a singular entry point for communicating with the devices currently
    available from ONT. Command calls are kept the same and expect the same
    inputs wherever possible to reduce script dependency on a particular
    hardware. There are some cases though, due to hardware design, when calls
    will be unique to a device, because of this this class is an abstract class
    and should be used through the correct hardware specific engine wrapper.

    If you are writing a script that is intended for a specific hardware then
    you can invoke the device-specific class directly. If you don't know which
    device you are intending your script to run on or want it cross platform
    specific there is an engine factory class that will automatically detect
    the hardware and invoke the correct hardware wrapper.

    All data request and configuration setting calls are the same on any
    platform.
    """

    KEEP = "keep"

    def __init__(self, connection: Optional[Connection] = None, grpc_port: Optional[int] = None):
        """
        **Constructor method**

        :param connection: An instance of minknow class Connection. If not
            provided a new instance is created internally using the passed
            grpc_port (if not None)
        :param grpc_port: If connection is None this port is used to create
            a new instance of Connection. If None the port is retrieved from
             the MINKNOW_RPC_PORT_SECURE environment variable
        """
        # this is the gRPC calls
        if connection is not None:
            self.connection = connection
        else:
            if grpc_port is None:
                grpc_port = get_env_grpc_port()
            try:
                self.connection = Connection(port=grpc_port, use_tls=True)
            except TypeError:
                # TODO: On major version bump, just use the except case
                self.connection = Connection(port=grpc_port)

        attach_device_to_logger(self)
        self.logger = logging.getLogger(__name__)
        self.device_info = self.connection.device.get_device_info()

        # Allow overriding of manager port. If not present, minknow_api gets a default
        if "MINKNOW_MANAGER_TEST_PORT" not in os.environ:
            self.provided_manager_port = None
        else:
            self.provided_manager_port = int(os.environ["MINKNOW_MANAGER_TEST_PORT"])

        self.host_info = minknow_api.manager.Manager(port=self.provided_manager_port).describe_host()
        self.flow_cell_info = self.connection.device.get_flow_cell_info()

        # Acquisition states, for translation
        self._acq_msgs = self.connection.acquisition._pb
        self.ACQ_ERROR = self._acq_msgs.ERROR_STATUS
        self.ACQ_READY = self._acq_msgs.READY
        self.ACQ_STARTING = self._acq_msgs.STARTING
        self.ACQ_PROCESSING = self._acq_msgs.PROCESSING
        self.ACQ_FINISHING = self._acq_msgs.FINISHING

        self.analysis_conf_msgs = self.connection.analysis_configuration._pb
        self.dev_msgs = self.connection.device._pb
        self.data_msgs = self.connection.data._pb
        self.log_msgs = self.connection.log._pb

        # Which well number indicates that the channel is disconnected
        self.disconnected_well = self.dev_msgs.SelectedWell.Value("WELL_NONE")

    def __del__(self):
        """
        When the device falls out of scope, it is best to close the grpc connection.
        Not doing this can cause issues when reconnecting.
        """
        self.close_grpc_connection()

    ##############
    # Properties #
    ##############

    @property
    def is_minion(self) -> bool:
        """
        Whether the device is a minion. Note this is a bed definition. When MinKNOW takes over
        the overriding system, this will make much more sense.
        """
        return self.device_info.device_type == self.device_info.MINION and not self.is_minit

    @property
    def is_mk1c(self) -> bool:
        """
        Whether the device is a mk1c
        """
        return self.device_info.device_type == self.device_info.MINION_MK1C

    @property
    def is_minit(self) -> bool:
        """
        Whether the device is a minit
        """
        return self.host_info.product_code == "MIN-101B"

    @property
    def is_promethion(self) -> bool:
        """
        Whether the device is a promethion
        """
        return self.device_info.device_type == self.device_info.PROMETHION

    @property
    def is_gridion(self) -> bool:
        """
        Whether the device is a gridion
        """
        return self.device_info.device_type == self.device_info.GRIDION

    @property
    def is_flongle(self) -> bool:
        """
        Whether the device is a flongle
        """
        return getattr(self.flow_cell_info, "has_adapter", False)

    @property
    def is_minion_like(self) -> bool:
        """
        Whether the device uses a chip that is of minion-type.

        This feature is intended to be used as a user-friendly mechanism to set
        IAO2X-specific settings.
        """
        return self.is_minion or self.is_gridion or self.is_flongle or self.is_minit or self.is_mk1c

    @property
    def device_type(self) -> bool:
        """
        The MinKNOW device type (for example, promethion or minion) that has been detected by
        minknow
        """
        return self.device_info.device_type

    @property
    def max_wells_per_channel(self) -> int:
        """
        Maximum number of wells connected to each channel on the device. Note
        that not all of these may be connected.

        :return: Int of total wells
        """
        return self.device_info.max_wells_per_channel

    @property
    def wells_per_channel(self) -> int:
        """
        Total wells supported by the device

        :return: Int of total wells
        """
        return self.flow_cell_info.wells_per_channel

    @property
    def max_channel_count(self) -> int:
        """
        Maximum number of channels that the device can have.

        :return: Int of total channels
        """
        return self.device_info.max_channel_count

    @property
    def channel_count(self) -> int:
        """
        Total channels supported by the device

        :return: Int of total channels
        """
        return self.flow_cell_info.channel_count

    @property
    def digitisation(self) -> int:
        """
        Digitisation of the ADC on the device i.e. total number of values that
        the ADC can measure.

        :return: Digitisation of the device
        """
        return self.device_info.digitisation

    @property
    def device_name(self) -> str:
        """
        Device name (currently minion/gridion/promethion)
        """
        if self.is_promethion:
            return "promethion"
        if self.is_gridion:
            return "gridion"
        return "minion"

    ###############
    # Set Methods #
    ###############

    def set_context_tags(self, tags: dict[str, str], merge: bool = False) -> None:
        """
        Sets tags to be the new set of context tags overriding the current tags.

        :param dict tags: should be a dict where all keys and values are able
            to be converted to strings
        :param merge bool: Whether to merge it with existing tags or not
        """
        if merge:
            old_tags = self.get_context_tags()
            old_tags.update(tags)
            tags = old_tags

        self.connection.protocol.set_context_info(context_info=tags)

    def set_writer_configuration(self, config: dict) -> None:
        """Pass a config style dictionary to MinKNOW explaining writer
        setup. See the default_writer config for more information

        :param config: dict of writer to pass

        """
        # Grab the transformer for json <-> analysis config pb
        writer_conf_pb = analysis_configuration_service.WriterConfiguration()

        # Manipulate dict to json
        json_config = json.dumps(config)

        # Swap to protobuf
        writer_config = json_format.Parse(json_config, writer_conf_pb)

        self.connection.analysis_configuration.set_writer_configuration(writer_config)

    def set_basecaller_configuration(self, basecaller_config: dict) -> None:
        """
        Sets the basecaller configuration in MinKNOW.

        Expects a config style dictionary passed in. See any sequencing configurations
        for what this may look like.

        An example:

        enable = true
        config_filename = "test.cfg"

        [read_filtering]
        min_qscore = 7

        [barcoding_configuration]
        barcoding_kits = ['ABC']
        trim_barcodes = true


        This call will reset any previous calls to this (e.g. the read_filtering)

        The options for read_filtering are:
        * min_qscore
        * min_samples [Deprecated MK4.5]
        * max_samples [Deprecated MK4.5]
        * min_bases
        * max_bases
        * max_failed_chunks [Deprecated MK4.5]

        :param basecaller_config: dict of which basecall parameters to set

        """
        # Grab the transformer for json <-> analysis config pb
        basecall_conf_pb = analysis_configuration_service.BasecallerConfiguration()

        # Manipulate dict to json
        json_config = json.dumps(basecaller_config)

        # Swap to protobuf
        basecaller_config_message = json_format.Parse(json_config, basecall_conf_pb)

        # Create request wrapper
        basecaller_config_request = self.analysis_conf_msgs.SetBasecallerConfigurationRequest(
            configs=basecaller_config_message
        )

        self.connection.analysis_configuration.set_basecaller_configuration(basecaller_config_request)

    @abc.abstractmethod
    def set_all_channels_to_well(self, well: int) -> None:
        """
        Set all channels to the physical well

        :param well: 1-indexed well number
        """
        pass

    def _set_all_channels_to_well(self, well: int) -> None:
        self.logger.info("Set the device to well: {}".format(well))

        self.connection.device.set_channel_configuration_all(well=well, test_current=False)
        legacy_base.channel_well_configuration({channel: well for channel in self.get_channel_list()})

    @abc.abstractmethod
    def set_channels_to_well(self, channel_config: dict[int, int]) -> None:
        """
        Set channels to the desired physical well. This takes a dictionary of
        1-indexed channels to be set to their corresponding 1-indexed wells

        :param channel_config: dictionary of {channel : well}
        """
        pass

    def _set_channels_to_well(self, channel_config: dict[int, int]) -> None:
        config_map = {
            channel: self.dev_msgs.ChannelConfiguration(well=well, test_current=False)
            for channel, well in channel_config.items()
        }

        self.connection.device.set_channel_configuration(channel_configurations=config_map)
        legacy_base.channel_well_configuration(channel_config)

    @abc.abstractmethod
    def set_all_channel_inputs_to_disconnected(self) -> None:
        """
        Disconnect all channels
        """
        pass

    def set_sample_rate(self, sample_rate: int) -> None:
        """
        Set the asic to the sample rate specified

        :param sample_rate: Integer value of sample rate to set
        """
        self.connection.device.set_sample_rate(sample_rate=int(sample_rate))

    @abc.abstractmethod
    def set_integration_capacitor(self, capacitance: float) -> None:
        """
        Set the asic to the integration capacitance specified in femtofarads

        :param capacitance: capacitance in femtofarads
        """
        pass

    @abc.abstractmethod
    def set_gain(self, gain: int) -> None:
        """
        Set the gain of the asic

        :param gain: value of gain
        """
        pass

    @abc.abstractmethod
    def set_unblock_voltage(self, voltage: float) -> None:
        """
        Sets the voltage level to apply for unblocking. self.unblock will use this value

        :param int voltage: Voltage in mV
        """
        pass

    def set_all_channels_to_test_current(self) -> None:
        """
        Set all channels to apply the test current

        :param test_current: Value in pA for the test current
        :param wait_for_change: boolean. Wait until the channels have changed
            to test current before returning.
        """
        self.logger.info("Set the device to test current")

        config_map = {x: self.dev_msgs.ChannelConfiguration(well=0, test_current=True) for x in self.get_channel_list()}
        self.connection.device.set_channel_configuration(channel_configurations=config_map)

    def set_exp_script_purpose(self, script_purpose: str) -> None:
        """
        Set the exp_script_purpose tag in the tracking_id section of the ping
        and fast5 file

        :param script_purpose: string experiment script purpose name
        """

        if type(script_purpose) != str:
            raise ValueError(
                "Script purpose must be a string. Type was: {} "
                "Value was: {}".format(type(script_purpose), script_purpose)
            )
        script_purpose_encoded = script_purpose.encode("utf-8")

        if script_purpose_encoded == "":
            raise ValueError("The script purpose should not be an empty string")

        self.connection.protocol.set_protocol_purpose(purpose=script_purpose_encoded)

    def set_calibration(self, ranges: list[float], offsets: list[float]) -> None:
        """
        :param ranges: 1D numpy array of length n-channels of floats
            representing range in pA for each channel
        :param offsets: 1D numpy array of length n-channels of integers
            representing the offset for each channel
        """
        if len(offsets) != self.channel_count:
            raise RuntimeError("Setting calibration with {} offsets is invalid".format(len(offsets)))
        if len(ranges) != self.channel_count:
            raise RuntimeError("Setting calibration with {} ranges is invalid".format(len(ranges)))

        self.connection.device.set_calibration(
            first_channel=1,
            last_channel=self.channel_count,
            offsets=offsets,
            pa_ranges=ranges,
        )

    def set_temperature(
        self,
        target: float,
        timeout: int = 300,
        tolerance: float = 0.5,
        min_stable_duration: int = 5,
        exit_on_timeout: bool = False,
    ) -> None:
        """Asks MinKNOW to set the temperature to be target. If the temperature can't be reached
        within the timeout period then a message will be logged to the gui and the script
        will continue.

        :param float target: Target temperature to get to in celsius
        :param float timeout: How long to wait in seconds.
        :param tolerance: What +/- value the temperature is okay to be within
        :param min_stable_duration: How many seconds the temperature has to be
                                    in the bounds to be a success
        :param exit_on_timeout: If the timeout is reached waiting for temperature,
                                sys.exit(0). Defaults to false
        """

        if not self.device_info.can_set_temperature:
            self.logger.warning("Device temperature cannot be set")
            return

        if timeout:
            # Create waiting settings
            wait = self.dev_msgs.SetTemperatureRequest.WaitForTemperatureSettings(
                timeout=timeout,
                tolerance=tolerance,
                min_stable_duration=min_stable_duration,
            )

            self.logger.log_to_gui(  # type: ignore
                "base_device.wait_for_temperature",
                params=dict(target=target, timeout=timeout),
            )

            # And apply
            response = self.connection.device.set_temperature(temperature=target, wait_for_temperature=wait)

            # If couldn't reach in the time: Warn the user
            if response.timed_out_waiting_for_temperature:
                if exit_on_timeout:
                    self.logger.log_to_gui(  # type: ignore
                        "base_device.exit_on_timeout_temperature",
                        params=dict(target=target, timeout=timeout, tolerance=tolerance),
                    )
                    sys.exit(0)

                if self.is_promethion:
                    self.logger.log_to_gui(  # type: ignore
                        "base_device.warn_to_reach_temperature",
                        params=dict(timeout=timeout),
                    )
                else:
                    self.logger.log_to_gui(  # type: ignore
                        "base_device.fail_to_reach_temperature",
                        params=dict(target=target, timeout=timeout, tolerance=tolerance),
                    )
        else:
            self.logger.log_to_gui(  # type: ignore
                "base_device.set_temperature",
                params=dict(target=target),
            )

            # And apply
            response = self.connection.device.set_temperature(temperature=target)

    def set_bias_voltage_offset(self, offset: float) -> None:
        """If this nanodeep is called, every time set_bias_voltage is called
        this offset value will be added to the value. Per instance of device

        This will adjust the current bias voltage of the device to be inline

        :param offset: (mV) Adjustment to be used
        """
        voltage = self.get_bias_voltage()
        legacy_base.set_bias_voltage_offset(offset)
        self.set_bias_voltage(voltage)

    def set_bias_voltage(self, bias_voltage: float) -> None:
        """
        Set the bias voltage. (With any applied offsets)

        :param bias_voltage: Value for the bias voltage
        :param wait_for_change: boolean. Wait until the bias voltage to be
            applied before returning. Data acquisition must be running for this
            to work
        """
        offset = self.get_bias_voltage_offset()
        voltage = bias_voltage + offset
        self.logger.info(
            f"setting bias voltage to {voltage}, with a bias voltage of {bias_voltage} and a voltage offset of {offset}"
        )
        self.connection.device.set_bias_voltage(bias_voltage=voltage)

        legacy_base.set_bias_voltage(bias_voltage)

    def set_channel_calibration_status(self, channel_list: list[bool]) -> None:
        """
        Store which channels passed/failed calibration

        :param channel_list: list of booleans assuming 0 index is first channel
        """
        channels = self.get_channel_list()
        chan_dict = {channels[i]: channel_list[i] for i in range(len(channels))}
        legacy_base.set_channel_calibration_status(chan_dict)

    def set_platform_qc_result(self, passed: bool, total_pore_count: int) -> None:
        """Store platform QC result in minknow

        :param passed: Whether the PQC was classed as successful (Usually based on pore_count)
        :param total_pore_count: Number of pores found from a PQC
        """

        self.connection.protocol.set_platform_qc_result(
            protocol_run_id=self.get_protocol_run_id(),
            pqc_result=protocol_service.PlatformQcResult(
                flow_cell_id=self.flow_cell_info.flow_cell_id,
                passed=passed,
                total_pore_count=total_pore_count,
            ),
        )

    def set_mux_scan_metadata(self, metadata: acquisition_service.MuxScanMetadata) -> None:
        """Sets the metadata in MinKNOW. This is used to group/colour mux scan values in reports and in the UI. New from 5.0

        :param metadata acquisition_service.MuxScanMetadata: metadata to set
        """
        bream_info = acquisition_service.BreamInfo(mux_scan_metadata=metadata)
        self.connection.acquisition.set_bream_info(info=bream_info, overwrite_unset_fields=False)

    def append_mux_scan_result(self, counts: dict[str, int]) -> None:
        """Sends the newest mux scan count information to MinKNOW. New from 5.0

        counts example: {"good_single": 5, ...}
        :param counts: A dict of classification names -> chan/mux count
        """
        result = acquisition_service.MuxScanResult(counts=counts, mux_scan_timestamp=self.get_acquisition_duration())
        self.connection.acquisition.append_mux_scan_result(result)

    def update_acquisition_display_targets(
        self,
        q_score_target: Optional[tuple[float, float]] = None,
        temperature_target: Optional[tuple[float, float]] = None,
        translocation_speed_target: Optional[tuple[float, float]] = None,
    ) -> None:
        """Updates the target bands on the UI graphs and reports. New as of MK 5.0.
        If some values are None/not specified the existing values will still hold.

        :param q_score_target tuple[float, float]: lower, upper q score target
        :param temperature_target tuple[float, float]: lower, upper temperature target
        :param translocation_speed_target tuple[float, float]: lower, upper translocation speed target
        """
        bream_info = acquisition_service.BreamInfo()

        if q_score_target is not None:
            bream_info.target_q_score.minimum = q_score_target[0]
            bream_info.target_q_score.maximum = q_score_target[1]
        if temperature_target is not None:
            bream_info.target_temperature.minimum = temperature_target[0]
            bream_info.target_temperature.maximum = temperature_target[1]
        if translocation_speed_target is not None:
            bream_info.target_translocation_speed.minimum = translocation_speed_target[0]
            bream_info.target_translocation_speed.maximum = translocation_speed_target[1]

        self.connection.acquisition.set_bream_info(info=bream_info, overwrite_unset_fields=False)

    def set_analysis_configuration(self, config_dict: dict, merge: bool = True) -> None:
        """Given a nested dict, apply this to be the sole analysis
        configuration - This will overwrite all previous analysis
        configuration, unless merge is specified.  This dict will follow the protobuf style.
        You can pass straight in from configs. See a sequencing config for an example.

        {"read_classification": { "classification_strategy": "modal",
                                { "parameters": {
                                     "rules_in_execution_order": [ "...." ] }}}
         "event_detection": ... }

        :param config_dict: dictionary following specification above
        :param merge: Whether to merge these settings with the existing ones
        :raises RpcError: If any parameters are incorrect
        """

        if merge:
            old_config = self.get_analysis_configuration()
            recursive_merge(old_config, config_dict)
            config_dict = old_config

        # Grab the transformer for json <-> analysis config pb
        analysis_conf_pb = analysis_configuration_service.AnalysisConfiguration()

        # Manipulate dict to json
        json_config = json.dumps(config_dict)

        # Swap to protobuf
        analysis_config = json_format.Parse(json_config, analysis_conf_pb)

        # Send config
        self.connection.analysis_configuration.set_analysis_configuration(analysis_config)

    def _set_saturation_thresholds(
        self,
        general_threshold: Optional[int] = None,
        unblock_threshold: Optional[int] = None,
        user_general_threshold: Optional[int] = None,
        user_unblock_threshold: Optional[int] = None,
    ) -> None:
        """
        Set the saturation thresholds
        * If any of the parameters are None, it will use what is currently registered
        * If any are 0, that means that that threshold will not trigger
        * Note that the saturation checker in MinKNOW only checks every 64th sample

        :param general_threshold: The number of samples in the software saturation (adc) range
            required to trigger saturation.
        :param unblock_threshold: The number of samples in the software saturation (adc) range
            required to trigger saturation. Only for samples in unblock muxes.
        :param user_general_threshold: The number of samples in the user saturation (pA)
            range required to trigger saturation
        :param user_unblock_threshold: The number of samples in the user saturation (pA)
            range required to trigger saturation. Only for samples in unblock muxes.
        """

        old_thresholds = self.connection.device.get_saturation_config().thresholds
        new_thresholds = {}

        def get_new_threshold(old_threshold, new_value):
            if new_value is None:
                return old_threshold
            if new_value % 64 != 0:
                raise RuntimeError(
                    "Found saturation threshold value "
                    + "{}, but thresholds need to be in multiples of 64".format(new_value)
                )
            #  This is divide by 64 as the saturation checker in MinKNOW checks every 64th sample
            return UInt32Value(value=int(new_value / 64.0))

        # Merge values with values currently in there
        new_thresholds["general_threshold"] = get_new_threshold(old_thresholds.general_threshold, general_threshold)

        new_thresholds["unblock_threshold"] = get_new_threshold(old_thresholds.unblock_threshold, unblock_threshold)

        new_thresholds["user_general_threshold"] = get_new_threshold(
            old_thresholds.user_general_threshold, user_general_threshold
        )

        new_thresholds["user_unblock_threshold"] = get_new_threshold(
            old_thresholds.user_unblock_threshold, user_unblock_threshold
        )

        # Propagate to device
        self.connection.device.set_saturation_config(
            thresholds=self.dev_msgs.SaturationConfig.Thresholds(**new_thresholds)
        )

    def set_saturation_adc(
        self,
        min_value: int,
        max_value: int,
        threshold: Optional[int] = None,
        saturation_during_unblock: Optional[bool] = None,
    ) -> None:
        """Specify the min/max ADC values to trigger
        saturation. threshold is the threshold of how many
        samples that need to lie outside of the min/max to trigger
        saturation.

        * If threshold is not specified, it will use the current threshold.
        * If threshold is 0 this will cause an exception -
            Why specify boundaries that will never trigger?
        * The threshold is expected to be in multiples of 64

        :param min_value: int of min value to not saturate (ADC)
        :param max_value: int of max value to not saturate (ADC)
        :param threshold: int of how many samples should trigger saturation
        :param saturation_during_unblock: bool if saturation should also apply to unblock
        :raises Rendezvous: If no threshold now or before has been set
        """

        if saturation_during_unblock is None:
            # Don't pass anything to keep what it was before
            self._set_saturation_thresholds(general_threshold=threshold)
        elif not saturation_during_unblock:
            # If false, pass 0 to disable
            self._set_saturation_thresholds(general_threshold=threshold, unblock_threshold=0)
        else:
            # If its true pass threshold to both
            self._set_saturation_thresholds(general_threshold=threshold, unblock_threshold=threshold)

        self.connection.device.set_saturation_config(
            software_saturation=self.dev_msgs.SaturationConfig.SoftwareSaturation(
                enabled=True,
                software_min_adc=Int32Value(value=min_value),
                software_max_adc=Int32Value(value=max_value),
            )
        )

    def set_saturation_pa(
        self,
        min_value: float,
        max_value: float,
        threshold: Optional[int] = None,
        saturation_during_unblock: Optional[bool] = None,
    ) -> None:
        """Specify the min/max pA values to trigger
        saturation. threshold is the threshold of how many
        samples that need to lie outside of the min/max to trigger
        saturation.

        * If threshold is not specified, it will use the current threshold.
        * If threshold is 0 this will cause an exception -
            Why specify boundaries that will never trigger?
        * The threshold is expected to be in multiples of 64

        :param min_value: int of min value to not saturate (pA)
        :param max_value: int of max value to not saturate (pA)
        :param threshold: int of how many samples should trigger saturation
        :param saturation_during_unblock: bool if saturation should also apply to unblock
        :raises Rendezvous: If no threshold now or before has been set
        """

        if saturation_during_unblock is None:
            self._set_saturation_thresholds(user_general_threshold=threshold)
        elif not saturation_during_unblock:
            self._set_saturation_thresholds(user_general_threshold=threshold, user_unblock_threshold=0)
        else:
            self._set_saturation_thresholds(user_general_threshold=threshold, user_unblock_threshold=threshold)

        self.connection.device.set_saturation_config(
            user_threshold_saturation=self.dev_msgs.SaturationConfig.UserThresholdSaturation(
                enabled=True,
                user_threshold_min_pa=FloatValue(value=min_value),
                user_threshold_max_pa=FloatValue(value=max_value),
            )
        )

    def disable_saturation_control(self) -> None:
        self.connection.device.set_saturation_config(
            user_threshold_saturation=self.dev_msgs.SaturationConfig.UserThresholdSaturation(enabled=False),
            software_saturation=self.dev_msgs.SaturationConfig.SoftwareSaturation(enabled=False),
        )

    @abc.abstractmethod
    def start_waveform(self, waveform_values: Sequence[Union[float, int]], frequency: Optional[float] = None) -> None:
        """
        Method for setting the waveform

        :param waveform_values: Bias voltage values
        :param frequency: float, only used by PromethION
        """
        pass

    @abc.abstractmethod
    def stop_waveform(self) -> None:
        """
        Stop applying the waveform

        """
        pass

    ##################
    # EEPROM Methods #
    ##################

    def write_adapter_id(self, adapter_id: str) -> None:
        """Writes the adapter id to the EEPROM. This will upgrade the eeprom to the v6 schema

        :param str adapter_id: The adapter id to write
        """

        msgs = self.connection.production._pb

        self.logger.info("Pushing EEPROM Version 6 with adapter_id: {}".format(adapter_id))

        self.connection.production.write_flowcell_data(
            v6=msgs.WriteFlowcellDataRequest.Version6(minor_version=0, adapter_id=adapter_id, temperature_offset=32767)
        )

        # Update cached version
        self.flow_cell_info = self.connection.device.get_flow_cell_info()

        # It should have been written
        if self.flow_cell_info.adapter_id != adapter_id:
            self.print_error_to_gui("Adapter ID not written correctly")
            raise RuntimeError("Adapter ID not written correctly")

        self.logger.info("Adapter ID {} successfully written".format(adapter_id))

    def write_product_code(self, product_code: str) -> None:
        """Writes the product code to the EEPROM. If the EEPROM is
        incompatible, upgrade it to Version 5
        (Could be V2 that doesn't support writing of product code)

        :param str product_code: The product code to write
        """

        self.logger.info(
            (
                "Pushing EEPROM Version 5 with wells_per_channel {0} "
                + "flowcell_id {1} product_code {2} temperature_offset {3}"
            ).format(
                self.wells_per_channel,
                self.flow_cell_info.flow_cell_id,
                product_code,
                32767,
            )
        )

        self.connection.production.write_flowcell_data(
            v5=WriteFlowcellDataRequest.Version5(
                minor_version=0,
                wells_per_channel=self.wells_per_channel,
                flowcell_id=self.flow_cell_info.flow_cell_id,
                product_code=product_code,
                temperature_offset=32767,
            )
        )

        # Update cached version
        self.flow_cell_info = self.connection.device.get_flow_cell_info()

        # It should have been written
        if self.flow_cell_info.product_code != product_code:
            self.print_error_to_gui("Product code not written correctly")
            raise RuntimeError("Product code not written correctly")

        self.logger.info("Product code %s successfully written" % (product_code,))

    ###############
    # Get Methods #
    ###############

    def get_bias_voltage_times(self, start: Optional[float] = None, end: Optional[float] = None) -> dict[float, float]:

        """
        returns all bias voltage changes that bream as requested from minknow, they come as a dict of key, value
        where key is the time and value is the voltage set.
        you can optionally pass in a start and end seconds to just get back voltages for those times
        :param start: seconds the search will start from (monotonic)
        :param end: seconds the search will return on (monotonic)
        :return: a dict of time_stamps : voltage, for the given start and end times.
        """

        return legacy_base.get_bias_voltage_times(start, end)

    def get_flow_cell_id(self) -> str:
        """Returns the current flow cell id. If flongle, it will return the user_specified_flow_cell_id.
        If not flongle try to return the ID on eeprom. If that isn't present, return the user specified one

        :returns: flow cell id
        :rtype: str

        """
        if self.is_flongle:
            return self.flow_cell_info.user_specified_flow_cell_id

        flow_cell_id = self.flow_cell_info.flow_cell_id
        if not flow_cell_id:
            flow_cell_id = self.flow_cell_info.user_specified_flow_cell_id
        return flow_cell_id

    def get_read_classification_names(self, include_internal_classifications: bool = True) -> list[str]:

        """
        Gets the read classifications currently set in MinKNOW and parses it to
        get the names of the classifications names. These are then returned along
        with MinKNOWs own internal classifications to the user as str in a list
        :param include_internal_classifications: bool flag, if True the call returns the internal MinKNOW
               classifications in addition to the user defined ones.
        :return: list of the names of the set classifications.
        """

        analysis_config = self.get_analysis_configuration()
        current_classification_names = [
            x.split("=")[0] for x in analysis_config["read_classification"]["parameters"]["rules_in_execution_order"]
        ]
        if include_internal_classifications:
            current_classification_names.extend(INTERNAL_MINKNOW_TAGS)
        return current_classification_names

    def get_writer_configuration(self) -> dict:
        """
        Returns the writer configuration as a dict.

        :return dict: the contents of the read configuration
        """
        config = self.connection.analysis_configuration.get_writer_configuration()
        json_config = json_format.MessageToJson(
            config,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )

        return json.loads(json_config)

    def get_context_tags(self) -> dict[str, str]:
        """
        Returns the context tag buffer as a dict.

        :return dict: the contents of the context tag buffer
        """
        return dict(self.connection.protocol.get_context_info().context_info)

    def get_exp_script_purpose(self) -> str:
        """
        Returns the experiment script purpose

        :return str: The purpose of the script
        """
        return str(self.connection.protocol.get_protocol_purpose().purpose)

    def get_channel_states_watcher(
        self, first_channel: Optional[int] = None, last_channel: Optional[int] = None, heartbeat: Optional[int] = None
    ):
        """Returns a channel state streamer of all channels

        next(get_channel_states_watcher()).channel_states will give a list of channel state data

        :param first_channel: First channel (Defaults to 1)
        :param last_channel: Last channel (Defaults to last channel)
        :param heartbeat: If present, MinKNOW will send an update at least every heartbeart seconds
        :returns: A (blocking) stream of channel states
        :rtype: Channels states streamer (Generator)

        """
        if not first_channel:
            first_channel = 1
        if not last_channel:
            last_channel = self.channel_count

        if heartbeat:
            return self.connection.data.get_channel_states(
                first_channel=first_channel,
                last_channel=last_channel,
                heartbeat=Duration(seconds=heartbeat),
            )

        return self.connection.data.get_channel_states(first_channel=first_channel, last_channel=last_channel)

    def get_channel_configuration(
        self, first_channel: Optional[int] = None, last_channel: Optional[int] = None
    ) -> dict:
        """Return a dict of channel -> configuration
        The configuration.well will give the current well

        :param first_channel: First channel (Defaults to 1)
        :param last_channel: Last channel (Defaults to last channel)
        :returns: channel -> configuration
        :rtype: dict

        """
        if not first_channel:
            first_channel = 1
        if not last_channel:
            last_channel = self.channel_count

        response = self.connection.device.get_channel_configuration(channels=range(first_channel, last_channel + 1))
        return {channel: data for (channel, data) in enumerate(response.channel_configurations, start=first_channel)}

    @abc.abstractmethod
    def get_disconnection_status_for_active_wells(self) -> dict[int, bool]:
        """Returns whether each channel is disconnected or not

        :returns: channel->Disconnected?
        :rtype: dict

        """
        pass

    def get_time_in_each_well(self, wells: list[int]) -> dict[int, dict[int, float]]:
        """Return the amount of time we've asked each channel to be in each well for.

        This is currently only valid on any set_channels_to_well/set_all_channels_to_well
        call you do. If you start_acquisition without setting wells, you will get no
        information from this.

        :param wells: list of which wells to return
        :returns: channel->(well->seconds)

        """
        return legacy_base.channel_well_times(wells)

    def get_acquisition_duration(self) -> int:
        """Returns how long acquisition has been running for.
        If no acquisition is running, it will be the most recent acquisition.

        This will not take into account if the system clock was adjusted mid run.
        If a negative number arises from this, 0 will be returned instead.

        Will raise an Exception if an acquistion hasn't run since MinKNOW started

        :returns: How many seconds since acquisition started
        """

        acq_timestamp = self.connection.acquisition.get_acquisition_info().start_time

        current_timestamp = (datetime.utcnow() - datetime(1970, 1, 1)).total_seconds()
        relative_timestamp = int(current_timestamp - acq_timestamp.seconds)

        if relative_timestamp < 0:
            self.logger.warning(
                "get_acquisition_duration has -ve number, "
                + "most likely due to system clock adjusted mid run. Returning 0."
            )
            relative_timestamp = 0

        return relative_timestamp

    def get_run_id(self) -> str:
        """Returns the run id of the current acquisition
        This is guaranteed to be unique across all protocols/minknows

        :returns: string of ASCII characters
        :rtype: string
        :raises RpcError.FAILED_PRECONDITION: If an acquisition is not running
        """
        return self.connection.acquisition.get_acquisition_info().run_id

    def get_well_list(self) -> list[int]:
        """
        Return the list of the available wells on the device

        :return: list of 1-indexed integers of available wells
        """
        # MinKNOW will now always present it as [1...]
        return list(range(1, self.wells_per_channel + 1))

    def get_channel_list(self) -> list[int]:
        """
        Return the list of the available channels on the device

        :return: list of 1-indexed integers of available channels
        """
        # MinKNOW will now always present it as [1...]
        return list(range(1, self.channel_count + 1))

    def get_logs_directory(self) -> str:
        """
        Find where MinKNOW are outputting their logs

        :return: path to the MinKNOW log directory
        """
        return self.connection.instance.get_output_directories().log

    def get_protocol_output_path(self) -> str:
        """
        Find where MinKNOW output data is stored for the most recent
        protocol run (e.g. where reads are stored)

        :return: path to the MinKNOW output directory
        :raises RpcError: FAILED_PRECONDITION If no protocol has been run
        """

        return self.connection.protocol.get_run_info().output_path

    def get_protocol_run_id(self) -> str:
        """
        Get the unique ID for the protocol that has most recently been started.
        (Most likely by the protocol_selector)

        :return: ID of the most recent run
        :raises RpcError: FAILED_PRECONDITION If no protocol has been run
        """

        return self.connection.protocol.get_run_info().run_id

    def get_sample_rate(self) -> int:
        """
        Get the current sampling frequency of the device

        :return: sampling frequency
        """
        return self.connection.device.get_sample_rate().sample_rate

    def get_sample_id(self) -> str:
        """Retrievies the sample id of the most recently started run.
        This is set from the GUI by the user to define an experiment

        :returns: sample_id
        :rtype: string
        :raises RpcError: FAILED_PRECONDITION If no protocol has been run
        """
        return self.connection.protocol.get_run_info().user_info.sample_id.value

    def get_read_statistics(
        self,
        collection_time: float,
        completed_read: bool = True,
        channels: Optional[list[int]] = None,
        include_chunk_statistics: bool = False,
        include_current_statistics: bool = True,
    ) -> data_service.GetReadStatisticsResponse:
        """Get read statistics from MinKNOW using current analysis statistics.
        By default, it will return completed reads on all channels

        :param collection_time: How long to spend collecting data (in seconds)
        :param completed_read: If set to true, statistics is per
                               completed read otherwise it is per chunk
        :param list channels: list of channels to get statistics for.
                              If not specified, all channels will be used
        :param include_chunk_statistics: Whether to include chunk statistics in the result
        :param include_current_statistics: Whether to include current statistics in the result
        :return: ReadStatisticsMessage
        """
        read_split = self.data_msgs.GetReadStatisticsRequest.COMPLETED_READ
        if not completed_read:
            read_split = self.data_msgs.GetReadStatisticsRequest.CHUNK
        if not channels:
            channels = self.get_channel_list()

        return self.connection.data.get_read_statistics(
            seconds=collection_time,
            read_split=read_split,
            channels=channels,
            no_current_statistics=not include_current_statistics,
            no_chunk_statistics=not include_chunk_statistics,
        )

    def get_analysis_configuration(self) -> dict:
        """Gets the current analysis configuration in nested dict form

        :rtype: dict(dict)
        :return: nested dict of config e.g.
        {"read_classification": {"classification_strategy": "modal",
                                 "parameters":{
                                   "rules_in_execution_order": .... }}}
        """

        config = self.connection.analysis_configuration.get_analysis_configuration()
        json_config = json_format.MessageToJson(
            config,
            preserving_proto_field_name=True,
            including_default_value_fields=True,
        )
        return json.loads(json_config)

    def get_calibration_coefficients(self) -> tuple[list[float], list[float]]:
        """
        Get two arrays of the scaling coefficients for every channel - scaling
        and offset. To calculate the current the equation is:

        array_pA = (array_ADC + offset) * scale

        The returned arrays are of length self.channel_count i.e. data for
        a channel in the array is the 0-indexed channel number

        :return: tuple of (scaling, offset) numpy arrays of length
            self.channel_count, in channel order
        """
        resp = self.connection.device.get_calibration(first_channel=1, last_channel=self.channel_count)
        return (resp.pa_ranges, resp.offsets)

    def get_current_sample_number(self) -> int:
        """
        Get the current sample number. If no acquisition is running, 0 will be returned

        :return: integer sample number
        """
        return self.connection.acquisition.get_progress().raw_per_channel.acquired

    def get_calibration_status_all_channels(self) -> dict[int, bool]:
        """
        Return boolean dict of whether channel passed or failed calibration

        :return: (channel -> passed_calibration)
        """
        return legacy_base.get_calibration_status_all_channels()

    def get_bias_voltage(self) -> float:
        """
        Returns the currently set bias voltage (With any offsets adjusted)

        :return: bias voltage
        """
        return self.connection.device.get_bias_voltage().bias_voltage - legacy_base.get_bias_voltage_offset()

    def get_bias_voltage_offset(self) -> float:
        """
        Return the offset that is applied to set_bias_voltage calls

        :return: bias voltage offset (mV)
        """
        return legacy_base.get_bias_voltage_offset()

    def get_acquisition_status(self) -> int:
        """
        Get the current acquisition status of MinKNOW

        :return: acquisition status code
        """
        return self.connection.acquisition.current_status().status

    @abc.abstractmethod
    def get_minimum_voltage_adjustment(self) -> float:
        """
        Returns the minimum voltage multiplier the hardware can set.
        """
        pass

    @abc.abstractmethod
    def get_minimum_unblock_voltage_multiplier(self) -> float:
        """
        Returns the minimum unblock voltage multiplier the hardware can set.
        """
        pass

    def get_firmware_versions(self) -> dict[str, str]:
        """
        Returns a dict of name->version. This will vary between devices.

        Example keys that may be present:

        Hubblet Board: ...
        Satellite Board: ...
        GridION FPGA: ....
        MinION FPGA: ....
        USB: ...


        """
        lookup = {}

        versions = self.connection.device.get_device_info().firmware_version
        for version in versions:
            lookup[version.component] = version.version

        return lookup

    @abc.abstractmethod
    def get_unblock_voltage(self) -> float:
        """Gets the unblock voltage on the device."""
        pass

    #################
    # Other Methods #
    #################

    def start_acquisition(self, purpose: Optional[str] = None, file_output: Optional[bool] = None) -> None:
        """
        Begin an acquisition period in MinKNOW.

        :param purpose: Which purpose this acquisition is for.
        Can be sequencing/other_purpose/calibration. Defaults to other_purpose if not specified
        :param file_output: Whether to override default settings and ensure/disable file output
        """

        if purpose is None:
            purpose = "other_purpose"

        purposes = {
            "sequencing": self._acq_msgs.SEQUENCING,
            "calibration": self._acq_msgs.CALIBRATION,
            "other_purpose": self._acq_msgs.OTHER_PURPOSE,
        }

        file_outputs = {
            None: self._acq_msgs.AUTO,
            True: self._acq_msgs.FORCE,
            False: self._acq_msgs.DISABLE,
        }

        if purpose not in purposes:
            raise RuntimeError("Acquisition purpose not valid")

        self.logger.debug("Begin data acquisiton period")
        self.connection.acquisition.start(purpose=purposes[purpose], file_output=file_outputs[file_output])

        # We'll attempt to refresh the flow cell info now that we're pretty sure
        # we have one plugged in.
        self.flow_cell_info = self.connection.device.get_flow_cell_info()
        legacy_base.reset_well_timings()

    def stop_acquisition(self, keep_power_on: bool = False) -> None:
        """Stop streaming data through the MinKNOW analysis pipeline

        :param keep_power_on: Whether to keep asic power on or not. Defaults to False

        """
        self.logger.debug("End data acquisition period")

        # Make sure device is in a fully powered down state
        # So disconnecting the flowcell after this nanodeep returns is safe
        self.set_bias_voltage(0)
        self.set_all_channel_inputs_to_disconnected()

        self.connection.acquisition.stop(wait_until_ready=True, keep_power_on=keep_power_on)

        legacy_base.reset_well_timings()

    def preload_basecaller_configuration(self, basecaller_config: dict) -> None:
        """
        Sets the basecaller configuration in MinKNOW to be preloaded. It will not request preloading
        if enable is set to false.

        This is mainly useful for alignment files etc that may take a while to load.
        See set_basecaller_configuration for an explanation of the input.

        :param basecaller_config: dict of which basecall parameters to set

        """
        if basecaller_config.get("enable"):
            self.logger.info("Preloading basecaller configuration")

            # Grab the transformer for json <-> analysis config pb
            basecall_conf_pb = analysis_configuration_service.BasecallerConfiguration()

            # Manipulate dict to json
            json_config = json.dumps(basecaller_config)

            # Swap to protobuf
            basecaller_config_message = json_format.Parse(json_config, basecall_conf_pb)

            # Create request wrapper
            basecaller_config_request = self.analysis_conf_msgs.SetBasecallerConfigurationRequest(
                configs=basecaller_config_message
            )

            self.connection.analysis_configuration.preload_basecaller_configuration(basecaller_config_request)

    def unblock(self, channels: list[int], duration: float) -> None:
        """Speciies to unblock a list of channels, and for how long.

        unblock_voltage/regen_current settings, if available, are used by the device here

        :param channels: list of channels to unblock
        :param float duration: How long to attempt to unblock them for, in seconds
        """
        # unblock request actually takes either duration_in_seconds or duration_in_milliseconds
        # but it has to be an int
        duration = int(1000 * duration)
        self.connection.device.unblock(channels=channels, duration_in_milliseconds=duration)

    def cancel_unblocks(self) -> int:
        """Cancels any current unblocks

        :returns: Number of device.unblock calls stopped (Not the number of channels)
        :rtype: int

        """

        return self.connection.device.cancel_unblocks().cancelled_unblocks

    def reset_channel_states(self) -> None:
        """
        Reset the the state of every channel. In particular this call resets all the channels in a
        state having the the latch flag active. If such a state doesn't have the reset_on_mux_change flag
        active this is the only way the state for that channel can be changed.
        It does not reset locked channel states
        """
        self.connection.data.reset_channel_states()

    def lock_channel_states(self, channels: list[int], state_name: str) -> None:
        """Lock channels into a specified channel state
        For this to work the channel state has to exist and the logic criteria for it
        has to be 'never_evaluated'

        :param channels: list of channels to lock to specified state
        :param state_name: Name to lock to

        """
        self.connection.data.lock_channel_states(channels=channels, state_name=state_name)

    def unlock_channel_states(self, channels: list[int]) -> None:
        """Allows channels to come out of the locked state

        :param channels: list of channels to unlock

        """
        self.connection.data.unlock_channel_states(channels=channels)

    def _print_to_gui(
        self, message: str, severity: int, identifier: Optional[str] = None, extra_data: Optional[dict] = None
    ) -> None:

        if not identifier:
            identifier = ""
        if not extra_data:
            extra_data = {}
        # Everything is expected to be a string
        extra_data = {str(k): str(v) for (k, v) in extra_data.items()}

        self.connection.log.send_user_message(
            severity=severity, user_message=message, identifier=identifier, extra_data=extra_data
        )

    def print_log_to_gui(
        self, message: str, identifier: Optional[str] = None, extra_data: Optional[dict] = None
    ) -> None:
        """
        Passes message onto the MinKNOW to be displayed in the GUI

        :param str message: text message to be displayed in the GUI
        :param str identifier: An identifier for the message that can be used by
        UI for translation
        :param dict extra_data: A dict to pass to the UI to fill in the gaps when
        the identifier is looked up
        """
        self._print_to_gui(message, self.log_msgs.MESSAGE_SEVERITY_INFO, identifier, extra_data)

    def print_warning_to_gui(
        self, message: str, identifier: Optional[str] = None, extra_data: Optional[dict] = None
    ) -> None:
        """
        Passes message onto the MinKNOW to be displayed in the GUI

        :param str message: text message to be displayed in the GUI
        :param str identifier: An identifier for the message that can be used by
        UI for translation
        :param dict extra_data: A dict to pass to the UI to fill in the gaps when
        the identifier is looked up
        """
        self._print_to_gui(message, self.log_msgs.MESSAGE_SEVERITY_WARNING, identifier, extra_data)

    def print_error_to_gui(
        self, message: str, identifier: Optional[str] = None, extra_data: Optional[dict] = None
    ) -> None:
        """
        Passes message onto the MinKNOW to be displayed in the GUI

        :param str message: text message to be displayed in the GUI
        :param str identifier: An identifier for the message that can be used by
        UI for translation
        :param dict extra_data: A dict to pass to the UI to fill in the gaps when
        the identifier is looked up
        """
        self._print_to_gui(message, self.log_msgs.MESSAGE_SEVERITY_ERROR, identifier, extra_data)

    def close_grpc_connection(self) -> None:
        """
        Closes the gRPC connection to MinKNOW
        """
        del self.connection

    def send_ping_data(self, ping_data: dict) -> None:
        """
        Sends the ping to MinKNOW, where the context tags and tracking_id
         will be added to it and then sent on to our servers.

        :param dict ping_data: a dict of the data packet to be sent.
        """

        self.logger.debug("Sending ping data to MinKNOW")
        self.connection.log.send_ping(ping_data=ping_to_str(ping_data))

    def reset(self) -> None:
        # TODO: A device.reset(reset_calibration = True) would be very useful
        """
        Reset configurations to default settings.
        """
        self.logger.debug("Resetting device to default settings")
        self.cancel_unblocks()
        self.connection.device.reset_device_settings()
        self.stop_waveform()
        self.stop_acquisition()
        self.set_writer_configuration({})
        self.reset_saturation_control()
        self.reset_analysis_configuration()
        self.set_context_tags({})
        self.set_bias_voltage_offset(0)
        self.reset_signal_reader()

    def reset_signal_reader(self) -> None:
        """Reset where MinKNOW is retrieving the signal from."""
        device_enum = self.connection.acquisition._pb.SetSignalReaderRequest.DEVICE
        self.connection.acquisition.set_signal_reader(reader=device_enum)

    def reset_analysis_configuration(self) -> None:
        """Ask MinKNOW to reset the analysis configuration"""
        self.connection.analysis_configuration.reset_analysis_configuration()

    @abc.abstractmethod
    def reset_saturation_control(self) -> None:
        """
        Reset the saturation control values to some sensible values.
        """
        pass

    def set_estimated_run_time(self, run_time: int) -> None:
        """
        Sets the estimated time in the keystore to allow the GUI to display an estimate
        :param run_time: How long this protocol will be running for (in seconds)

        """
        # We want time.time() here and not monotonic because we want an end time including
        # timezone
        experiment = Experiment(estimated_end_time=int(time.time() + run_time))
        self.connection.keystore.store(values={"bream.RUN_ESTIMATED_END_TIME": experiment})
