from __future__ import annotations

from typing import NamedTuple, Optional

import grpc

import bream4.legacy.device_interfaces.devices.minion as legacy_minion
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface

Temperature = NamedTuple("Temperature", [("current_temp", float), ("desired_temp", float)])


class MinionGrpcClient(BaseDeviceInterface):
    """
    Minion-specific grpc calls
    """

    def __init__(self, **kwargs):
        super(MinionGrpcClient, self).__init__(**kwargs)

        self.min_msgs = self.connection.minion_device._pb

        self._test_current = self.min_msgs.MinionDeviceSettings.TEST_CURRENT
        self._disconnected = self.min_msgs.MinionDeviceSettings.DISCONNECTED

    ###############
    # Set Methods #
    ###############

    # For more information on any of the set_ methods, see the minknow API:
    # https://github.com/nanoporetech/minknow_api [Developer License]

    def set_integration_capacitor(self, capacitance: float) -> None:
        """
        Set the integration capacitor

        :param capacitance: The capacitance in femtofarads
        :return:
        """
        allowed_values = {
            self.KEEP: self.min_msgs.MinionDeviceSettings.INTCAP_KEEP,
            62.5: self.min_msgs.MinionDeviceSettings.INTCAP_62_5fF,
            250: self.min_msgs.MinionDeviceSettings.INTCAP_250fF,
            1000: self.min_msgs.MinionDeviceSettings.INTCAP_1pF,
        }
        if capacitance not in allowed_values:
            raise ValueError("Allowed capacitance (in femto Farads): {}".format(allowed_values.keys()))

        self.connection.minion_device.change_settings(int_capacitor=allowed_values[capacitance])

    def set_gain(self, gain: int) -> None:
        """
        Set the gain

        :param gain: value to set the gain to
        """
        allowed_values = {
            self.KEEP: self.min_msgs.MinionDeviceSettings.GAIN_KEEP,
            1: self.min_msgs.MinionDeviceSettings.GAIN_1,
            5: self.min_msgs.MinionDeviceSettings.GAIN_5,
        }

        if gain not in allowed_values:
            raise ValueError("Allowed gains: {}".format(allowed_values.keys()))

        self.connection.minion_device.change_settings(th_gain=allowed_values[gain])

    def set_unblock_voltage(self, voltage: float) -> None:
        """Sets the unblock voltage on the MinION. When self.unblock is
        called, this voltage will be used

        :param int voltage: mV to use. -372 to 0 in steps of 12
        """
        if 0 < voltage or voltage < -375 or voltage % 12 != 0:
            raise RuntimeError(
                "unblock voltage {} is out of hardware spec voltage must be a multiple of 12 between 0 and -375".format(
                    voltage
                )
            )

        self.connection.minion_device.change_settings(unblock_voltage=voltage)

    def set_non_overlap_clock(self, clock_setting: int) -> None:
        """
        Set the non overlap clock

        :param clock_setting: setting for the non overlap clock
        """
        allowed_values = {
            self.KEEP: self.min_msgs.MinionDeviceSettings.NOC_KEEP,
            1: self.min_msgs.MinionDeviceSettings.NOC_1_HS_CLOCK,
            2: self.min_msgs.MinionDeviceSettings.NOC_2_HS_CLOCK,
        }

        if clock_setting not in allowed_values:
            raise ValueError("Allowed non overlap clock settings: {}".format(allowed_values.keys()))

        self.connection.minion_device.change_settings(non_overlap_clock=allowed_values[clock_setting])

    def set_low_pass_filter_frequency(self, lpf_frequency: int) -> None:
        """
        Set the low pass filter frequency in kHz

        :param lpf_frequency: frequency for the low pass filter in kHz
        """
        allowed_values = {
            self.KEEP: self.min_msgs.MinionDeviceSettings.LPF_KEEP,
            5: self.min_msgs.MinionDeviceSettings.LPF_5kHz,
            10: self.min_msgs.MinionDeviceSettings.LPF_10kHz,
            20: self.min_msgs.MinionDeviceSettings.LPF_20kHz,
            40: self.min_msgs.MinionDeviceSettings.LPF_40kHz,
            80: self.min_msgs.MinionDeviceSettings.LPF_80kHz,
            "disabled": self.min_msgs.MinionDeviceSettings.LPF_DISABLED,
        }

        if lpf_frequency not in allowed_values:
            raise ValueError("Allowed low pass filter settings: {}".format(allowed_values.keys()))

        self.connection.minion_device.change_settings(low_pass_filter=allowed_values[lpf_frequency])

    def set_sinc_decimation(self, decimation: int) -> None:
        """
        Set sinc decimation

        :param decimation: decimation
        """
        allowed_values = {
            self.KEEP: self.min_msgs.MinionDeviceSettings.DECIMATION_KEEP,
            32: self.min_msgs.MinionDeviceSettings.DECIMATION_32,
            64: self.min_msgs.MinionDeviceSettings.DECIMATION_64,
        }

        if decimation not in allowed_values:
            raise ValueError("Allowed sinc decimation settings: {}".format(allowed_values.keys()))

        self.connection.minion_device.change_settings(sinc_decimation=allowed_values[decimation])

    def set_int_reset_time(self, reset_time: float) -> None:
        """
        Set the integration reset time.

        :param reset_time: Float for int reset time
        """
        self.connection.minion_device.change_settings(int_reset_time=float(reset_time))

    def set_th_sample_time(self, sample_time: float) -> None:
        """
        Set the sample time.

        :param sample_time: Float for the sample time.
        """
        self.connection.minion_device.change_settings(th_sample_time=float(sample_time))

    def set_sinc_delay(self, sinc_delay: int) -> None:
        """
        Set the sinc delay.

        :param sinc_delay: Int for the sinc delay.
        """
        self.connection.minion_device.change_settings(sinc_delay=int(sinc_delay))

    def set_samples_to_reset(self, reset_samples: int) -> None:
        """
        Set the samples to reset.

        :param reset_samples: Int for the samples to reset.
        """
        self.connection.minion_device.change_settings(samples_to_reset=int(reset_samples))

    def set_bias_current(self, bias_current: float) -> None:
        """
        Set the bias current.

        :param bias_current: Int for the bias current - can be a value between
            0 and 15 in steps of 5.
        """
        try:
            self.connection.minion_device.change_settings(bias_current=int(bias_current))
        except grpc.RpcError as e:
            raise e.details()

    def set_compensation_capacitor(self, capacitor_value: int) -> None:
        """
        Set the compensation capacitor.

        :param capacitor_value: Int for the compensation capacitor - can be a
            value between 0 and 49 in steps of 7.
        """
        try:
            self.connection.minion_device.change_settings(compensation_capacitor=int(capacitor_value))
        except grpc.RpcError as e:
            raise e.details()

    def set_overcurrent_limit(self, set_limit: bool) -> None:
        """
        Turn the overcurrent limit on or off.

        :param set_limit: Bool for the overcurrent limit.
        """
        if type(set_limit) is not bool:
            raise ValueError("Set overcurrent limit is either True or False")

        self.connection.minion_device.change_settings(overcurrent_limit=set_limit)

    def set_test_current(self, test_current: int) -> None:
        """
        Set the test current only, but do not set the channels to use it.
        Current is set in picoamps.

        :param test_current: Integer value for the test current in picoamps
        """
        self.logger.info("Set the device to test current {}".format(test_current))
        allowed_values = [0, 50, 100, 150, 200, 250, 300, 350]

        if test_current not in allowed_values:
            raise ValueError("Allowed test current settings: {}".format(allowed_values))
        try:
            self.connection.minion_device.change_settings(test_current=test_current)
        except grpc.RpcError as e:
            raise e.details()

    def set_all_channel_inputs_to_disconnected(self) -> None:
        """
        Set all channels to be disconnected
        """
        self.logger.debug("Set all channels to be disconnected")

        self.connection.minion_device.change_settings(channel_config_default=self._disconnected)

    def start_waveform(self, waveform_values: list[int], frequency: Optional[float] = None) -> None:
        """
        Method for setting the bias voltage lookup values.
        Effectively defines a waveform.

        Will iterate to the next value every 1 ms.

        :param waveform_values: Bias voltage values
        :param frequency: float, only used by PromethION
        """
        self.logger.debug("Set the waveform to {}".format(waveform_values))
        self.logger.debug("Starting the waveform")
        self.connection.minion_device.change_settings(bias_voltage_lookup_table=waveform_values)
        self.connection.minion_device.change_settings(enable_bias_voltage_lookup=True)

    def stop_waveform(self) -> None:
        """
        Stop applying the bias voltages in the device-defined lookup table

        """
        self.logger.debug("Stopping the waveform")
        self.connection.minion_device.change_settings(enable_bias_voltage_lookup=False)

    def set_all_channels_to_well(self, well: int) -> None:
        """
        Set all channels to the physical well

        :param well: 1-indexed well number
        """
        self.set_channels_to_well({channel: well for channel in self.get_channel_list()})
        if self.is_mk1c:
            legacy_minion.mk1c_signal_mitigation()

    def set_channels_to_well(self, channel_config: dict[int, int]) -> None:
        """
        Set channels to the desired physical well. This takes a dictionary of
        1-indexed channels to be set to their corresponding 1-indexed wells

        :param channel_config: dictionary of {channel : well}
        """
        # Move in groups to stop current peaks causing nearby saturation
        legacy_minion.set_channels_to_well(self, channel_config)

    ###############
    # Get Methods #
    ###############

    def get_disconnection_status_for_active_wells(self) -> dict[int, bool]:
        """
        Get a boolean dictionary of whether the channel has disconnected for the active well

        :return: A dict with channel as key and disconnected as the truth value
        """
        stg = self.connection.minion_device.get_settings()
        configs = stg.channel_config.items()

        # Make sure to exclude any non-flongle channels if flongle
        filtered_configs = {channel: well for channel, well in configs if channel <= self.channel_count}

        return {channel: well == self._disconnected for channel, well in filtered_configs.items()}

    def get_temperature(self) -> Temperature:
        """
        Returns a tuple of temperatures

        :return: NamedTuple - current_temp, desired_temp
        """
        response = self.connection.device.get_temperature()

        return Temperature(response.minion.heatsink_temperature.value, response.target_temperature)

    def get_minimum_voltage_adjustment(self) -> float:
        """
        Returns the minimum voltage multiplier the hardware can set.
        """
        return 5

    def get_minimum_unblock_voltage_multiplier(self) -> float:
        """
        Returns the minimum voltage multiplier the hardware can set.
        """
        return 12

    def get_unblock_voltage(self) -> float:
        """Gets the unblock voltage on the MinION. When self.unblock is
        called, this voltage will be used
        """
        stg = self.connection.minion_device.get_settings()
        return stg.unblock_voltage

    #################
    # Other Methods #
    #################

    def reset_saturation_control(self) -> None:
        """
        Reset the saturation control values to the defaults, which is saturation
        on, and some sensible thresholds
        """

        self.disable_saturation_control()
        self._set_saturation_thresholds(0, 0, 0, 0)
        self.set_saturation_adc(-4000, 4000, threshold=640, saturation_during_unblock=False)
