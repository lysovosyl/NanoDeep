from __future__ import annotations

from typing import NamedTuple, Optional, Sequence, Union

import bream4.legacy.device_interfaces.devices.promethion as legacy_promethion
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface

Temperature = NamedTuple("Temperature", [("current_temp", float), ("desired_temp", float)])


class PromethionGrpcClient(BaseDeviceInterface):
    """
    Promethion-specific grpc calls

    The structure of the PromethION chip is different from the MinION chip. The chip
    is divided into 12 sections, called pixel blocks. Each block has 250 channels
    which are called pixels internally. There are two ways to set settings
    for channels:

        #. set them per pixel block to apply the setting to all 250 pixelsat the same
            time. This is suitable for global settings like the unblock voltage
            and the regen V clamp voltage.
        #. set them per pixel, for example the gain, integration capacitor and test
            current.
    The Current API assumes that when setting a pixel block setting,
    the setting should be applied to all blocks. The current API does not currently
    support a block by block setting option.


    """

    def __init__(self, **kwargs):
        super(PromethionGrpcClient, self).__init__(**kwargs)
        self.prom_msgs = self.connection.promethion_device._pb
        self._disconnected = self.prom_msgs.PixelSettings.InputWell.InputConfig.Value("NONE")

    ###############
    # Set Methods #
    ###############

    def set_bias_voltage(self, bias_voltage: float):
        """Set the bias voltage of the device

        :param bias_voltage: Bias voltage to set
        """
        legacy_promethion.step_bias_voltage(self, bias_voltage)
        super().set_bias_voltage(bias_voltage)

    def set_all_channels_to_well(self, well: int) -> None:
        """
        Set all channels to the physical well

        :param well: 1-indexed well number
        """
        legacy_promethion.set_all_channels_to_well(self, well)  # Clears hardware saturation
        self._set_all_channels_to_well(well)

    def set_channels_to_well(self, channel_config: dict[int, int]) -> None:
        """
        Set channels to the desired physical well. This takes a dictionary of
        1-indexed channels to be set to their corresponding 1-indexed wells

        :param channel_config: dictionary of {channel : well}
        """
        legacy_promethion.set_channels_to_well(self, channel_config)  # Clears hardware saturation
        self._set_channels_to_well(channel_config)

    def set_all_channel_inputs_to_disconnected(self) -> None:
        """
        Sets all channels to be disconnected
        """
        self.set_all_channels_to_well(0)

    def set_integration_capacitor(self, capacitance: float) -> None:
        """
        Set the integration capacitor
        :param capacitance: The capacitance in femtofarads
        """
        allowed_values = {
            100: self.prom_msgs.PixelSettings.INTCAP_100fF,
            200: self.prom_msgs.PixelSettings.INTCAP_200fF,
            500: self.prom_msgs.PixelSettings.INTCAP_500fF,
            600: self.prom_msgs.PixelSettings.INTCAP_600fF,
        }

        if capacitance not in allowed_values:
            raise ValueError("Allowed capacitances " "(in femto Farads): {}".format(allowed_values.keys()))

        self.connection.promethion_device.change_pixel_settings(
            pixel_default=self.prom_msgs.PixelSettings(gain_capacitor=allowed_values[int(capacitance)])
        )

    def set_unblock_voltage(self, voltage: float) -> None:
        """Sets the unblock voltage on the PromethION. When self.unblock is
        called, this voltage will be used

        :param float voltage: mV to use. -1000 to 1000
        """
        new_settings = self.prom_msgs.PixelBlockSettings()
        new_settings.unblock_voltage.value = voltage

        self.connection.promethion_device.change_pixel_block_settings(pixel_block_default=new_settings)

    def set_gain(self, gain: int) -> None:
        """
        Set the second stage aka correlated double sample stage gain multiplier

        :param gain:
        """
        allowed_values = {
            2: self.prom_msgs.PixelSettings.INTGAIN_2,
            4: self.prom_msgs.PixelSettings.INTGAIN_4,
        }

        if gain not in allowed_values:
            raise ValueError("Allowed gains: {}".format(allowed_values.keys()))

        self.connection.promethion_device.change_pixel_settings(
            pixel_default=self.prom_msgs.PixelSettings(gain_multiplier=allowed_values[gain])
        )

    def set_ramp_voltage(self, ramp_voltage: float) -> None:
        """
        Set the ramp voltage

        :param ramp_voltage: Value for the ramp voltage
        """
        self.connection.promethion_device.change_device_settings(ramp_voltage=ramp_voltage)

    def set_current_inverted(self, invert: bool) -> None:
        """
        This sets whether to invert the currents polarity
        :param invert: Whether to invert the polarity

        """
        new_settings = self.prom_msgs.PixelSettings()
        new_settings.current_inverted.value = invert

        self.connection.promethion_device.change_pixel_settings(pixel_default=new_settings)

    def set_membrane_simulation_enabled(self, enable: bool) -> None:
        """
        This controls the state of the membrane simulation
        :param enable: Whether to enable membrane simulation

        """
        new_settings = self.prom_msgs.PixelSettings()
        new_settings.membrane_simulation_enabled.value = enable

        self.connection.promethion_device.change_pixel_settings(pixel_default=new_settings)

    def set_regeneration_current_voltage_clamp(self, voltage_clamp: float) -> None:
        """
        Set the regeneration current voltage clamp

        :param voltage_clamp: Current voltage clamp
        """

        new_settings = self.prom_msgs.PixelBlockSettings()
        new_settings.regen_current_voltage_clamp.value = voltage_clamp

        self.connection.promethion_device.change_pixel_block_settings(pixel_block_default=new_settings)

    def set_regeneration_current_test_enabled(self, enabled: bool) -> None:
        """
        This connects the regeneration current to the integration adc circuit and the input well.
        and allows users to read regen current via the channel adc value.


        :param enabled: Whether to enable the regeneration current test

        """
        new_settings = self.prom_msgs.PixelSettings()
        new_settings.regeneration_current_test_enabled.value = enabled

        self.connection.promethion_device.change_pixel_settings(pixel_default=new_settings)

    def set_regeneration_current(self, regen_current: int) -> None:
        """
        Set the regeneration current on all channels

        :param regen_current: regeneration current
        """
        allowed_values = {
            0: self.prom_msgs.PixelSettings.REGEN_0pA,
            50: self.prom_msgs.PixelSettings.REGEN_50pA,
            100: self.prom_msgs.PixelSettings.REGEN_100pA,
            150: self.prom_msgs.PixelSettings.REGEN_150pA,
            400: self.prom_msgs.PixelSettings.REGEN_400pA,
            450: self.prom_msgs.PixelSettings.REGEN_450pA,
            500: self.prom_msgs.PixelSettings.REGEN_500pA,
            550: self.prom_msgs.PixelSettings.REGEN_550pA,
            800: self.prom_msgs.PixelSettings.REGEN_800pA,
            850: self.prom_msgs.PixelSettings.REGEN_850pA,
            900: self.prom_msgs.PixelSettings.REGEN_900pA,
            950: self.prom_msgs.PixelSettings.REGEN_950pA,
            1200: self.prom_msgs.PixelSettings.REGEN_1200pA,
            1250: self.prom_msgs.PixelSettings.REGEN_1250pA,
            1300: self.prom_msgs.PixelSettings.REGEN_1300pA,
            1350: self.prom_msgs.PixelSettings.REGEN_1350pA,
        }

        if regen_current not in allowed_values:
            raise ValueError("Allowed regen_current values " "(in pico-amps (pA) ): {}".format(allowed_values.keys()))

        new_settings = self.prom_msgs.PixelSettings()
        new_settings.regeneration_current = allowed_values[regen_current]

        self.connection.promethion_device.change_pixel_settings(pixel_default=new_settings)

    def set_sample_rate(self, sample_rate: int) -> None:
        """
        Set the asic to the sample rate specified

        :param sample_rate: Integer value to set the sample rate to
        """

        super().set_sample_rate(sample_rate)
        legacy_promethion.set_sample_rate(self, sample_rate)

    def set_overload_mode(self, overload_mode: str, saturation_control_enabled: bool = True) -> None:
        """
        Sets the overload_mode of the device. This determines what happens
        when the current goes above the ADC range.

        mode can be one of:
        * 'set_flag' when hardware saturation is triggered this mode sets a flag
            to say saturation has happened but does not change the mux settings
        * 'latch_off' This will set the saturation flag and turn off the mux
        * 'limit' This limits the amount of current that can be felt by the
            channel but keeps turning it back on if the limit is exceeded
        * 'clear' resets all the flags and mux changes related to saturation

        :param overload_mode: which mode to set the overload to
        :param saturation_control_enabled: Whether saturation controls should trigger.
                                           `None` indicates leave it alone

        """
        allowed_values = {
            "set_flag": self.prom_msgs.PixelSettings.OVERLOAD_SET_FLAG,
            "latch_off": self.prom_msgs.PixelSettings.OVERLOAD_LATCH_OFF,
            "clear": self.prom_msgs.PixelSettings.OVERLOAD_CLEAR,
            "limit": self.prom_msgs.PixelSettings.OVERLOAD_LIMIT,
        }

        if overload_mode not in allowed_values.keys():
            raise RuntimeError(
                "overlod mode {} is not a valid setting. Please choose from one of {}".format(
                    overload_mode, allowed_values.keys()
                )
            )

        if saturation_control_enabled is not None:
            self.connection.promethion_device.change_device_settings(
                saturation_control_enabled=saturation_control_enabled
            )

        new_settings = self.prom_msgs.PixelSettings()
        new_settings.overload_mode = allowed_values[overload_mode]

        self.connection.promethion_device.change_pixel_settings(pixel_default=new_settings)

    def set_bias_current(self, bias_current: str) -> None:
        """
        Sets the bias current for the amplifier - this controls the level of noise of the signal.
        The higher the bias current, the lower the noise, but the bigger the heat and power drawn by
        the amplifier. If it is set to off, no signal readings can be made.

        bias_current can be one of:
        * 'off' - 0 microA
        * 'low' - 390 microA
        * 'nominal' - 586 microA
        * 'high' - 808 microA

        :param bias_current: What to set the bias current to

        """
        allowed_values = {
            "off": self.prom_msgs.PixelSettings.BIAS_OFF,
            "low": self.prom_msgs.PixelSettings.BIAS_LOW,
            "nominal": self.prom_msgs.PixelSettings.BIAS_NOMINAL,
            "high": self.prom_msgs.PixelSettings.BIAS_HIGH,
        }

        if bias_current not in allowed_values.keys():
            raise RuntimeError(
                "bias current {} is not a valid setting. Please choose from one of {}".format(
                    bias_current, allowed_values.keys()
                )
            )

        new_settings = self.prom_msgs.PixelSettings()
        new_settings.bias_current = allowed_values[bias_current]

        self.connection.promethion_device.change_pixel_settings(pixel_default=new_settings)

    def set_low_pass_filter_frequency(self, frequency: int) -> None:
        """
        Sets the signal filter for input adc signal.

        frequency can be one of: 10/20/30/40/50/60/70/80 (kHz)

        :param frequency: What to set lpf frequency to

        """
        allowed_values = {
            10: self.prom_msgs.PixelSettings.LPF_10kHz,
            20: self.prom_msgs.PixelSettings.LPF_20kHz,
            30: self.prom_msgs.PixelSettings.LPF_30kHz,
            40: self.prom_msgs.PixelSettings.LPF_40kHz,
            50: self.prom_msgs.PixelSettings.LPF_50kHz,
            60: self.prom_msgs.PixelSettings.LPF_60kHz,
            70: self.prom_msgs.PixelSettings.LPF_70kHz,
            80: self.prom_msgs.PixelSettings.LPF_80kHz,
        }

        if frequency not in allowed_values.keys():
            raise RuntimeError(
                "low_pass_filter_frequency {} is not a valid setting. Please choose from one of {}".format(
                    frequency, allowed_values.keys()
                )
            )

        new_settings = self.prom_msgs.PixelSettings()
        new_settings.cutoff_frequency = allowed_values[frequency]

        self.connection.promethion_device.change_pixel_settings(pixel_default=new_settings)

    def stop_waveform(self) -> None:
        """
        Stop applying the bias voltages in the device-defined lookup table

        """
        self.connection.promethion_device.change_device_settings(bias_voltage=0)

    def start_waveform(self, waveform_values: Sequence[Union[float, int]], frequency: Optional[float] = None) -> None:
        """
        Method for setting the bias voltage lookup values.
        Effectively defines a waveform.

        MinKNOW will attempt to up/down sample the waveform values depending on the firmware.
        32 is guaranteed to work. 128 and 1024 are dependent on prom firmware

        Will iterate to the next value every 1 ms.

        :param waveform_values: Bias voltage values. Length: 32, 128, 1024
        :param frequency: the frequency of the waveform to be applied
        """
        if len(waveform_values) not in (32, 128, 1024):
            raise RuntimeError(f"Waveform has {len(waveform_values)} values but expecting either 32, 128 or 1024")

        new_settings = self.prom_msgs.WaveformSettings(frequency=frequency, voltages=waveform_values)

        self.connection.promethion_device.change_device_settings(bias_voltage_waveform=new_settings)

    ###############
    # Get Methods #
    ###############

    def get_gain_and_int_capacitor(self) -> tuple[int, int]:
        """
        Get the gain and integration capacitor together

        :return: tuple of gain and integration capacitor
        """
        settings = self.connection.promethion_device.get_pixel_settings(pixels=self.get_channel_list())
        gain_capacitor = set()
        gain_multiplier = set()
        for pixel in settings.pixels:
            gain_capacitor.add(pixel.gain_capacitor)
            gain_multiplier.add(pixel.gain_multiplier)

        if len(gain_multiplier) > 1:
            raise RuntimeError("Multiple gain multipliers set across channels {}".format(gain_multiplier))

        if len(gain_capacitor) > 1:
            raise RuntimeError("multiple gain capacitors set across channels {}".format(gain_capacitor))

        pixel = settings.pixels[0]

        capacitance_name = pixel.GainCapacitor.Name(pixel.gain_capacitor)
        if capacitance_name == "INTCAP_KEEP":
            raise ValueError("Received unexpected value for capacitance: INTCAP_KEEP")
        # capacitance_name something like INTCAP_100fF
        capacitance = int(capacitance_name.split("_")[1][:-2])

        multiplier_name = pixel.GainMultiplier.Name(pixel.gain_multiplier)
        if multiplier_name == "INTGAIN_KEEP":
            raise ValueError("Received unexpected value for multiplier: INTGAIN_KEEP")
        # multiplier name something line INTGAIN_2
        multiplier = int(multiplier_name.split("_")[1])

        return multiplier, capacitance

    def get_overload_mode(self) -> str:
        """
        Gets the current overload_mode of the device. This makes the
        assumption that overload mode is the same for all the
        pixels. If set using the `set_overload_mode` api then this
        assumption will be true.

        mode that will be returned can be one of:
        * 'set_flag' when hardware saturation is triggered this mode sets a flag
            to say saturation has happened but does not change the mux settings
        * 'latch_off' This will set the saturation flag and turn off the mux
        * 'limit' This limits the amount of current that can be felt by the
            channel but keeps turning it back on if the limit is exceeded
        * 'clear' resets all the flags and mux changes related to saturation

        :return: String of current mode

        """
        lookup = {
            self.prom_msgs.PixelSettings.OVERLOAD_SET_FLAG: "set_flag",
            self.prom_msgs.PixelSettings.OVERLOAD_LATCH_OFF: "latch_off",
            self.prom_msgs.PixelSettings.OVERLOAD_CLEAR: "clear",
            self.prom_msgs.PixelSettings.OVERLOAD_LIMIT: "limit",
        }

        settings = self.connection.promethion_device.get_pixel_settings(pixels=[1])
        overload_mode = settings.pixels[0].overload_mode

        return lookup[overload_mode]

    def get_disconnection_status_for_active_wells(self):
        """
        Get a boolean dictionary of whether the channel has disconnected for the active well

        :return: A dict with channel as key and disconnected as the truth value
        """
        channels = self.get_channel_list()

        pixel_response = self.connection.promethion_device.get_pixel_settings(pixels=channels)
        pixels = pixel_response.pixels

        return {k: v.input.input_well == self._disconnected for k, v in zip(channels, pixels)}

    def get_temperature(self) -> Temperature:
        """
        Returns a tuple of temperatures

        :return: Namedtuple - current_temp, desired_temp
        """
        response = self.connection.device.get_temperature()
        return Temperature(response.promethion.flowcell_temperature.value, response.target_temperature)

    def get_minimum_voltage_adjustment(self) -> float:
        """
        Returns the minimum voltage multiplier the hardware can set.
        """
        return 1

    def get_minimum_unblock_voltage_multiplier(self) -> float:
        """
        Returns the minimum voltage multiplier the hardware can set.
        """
        return 1

    def get_regeneration_current(self) -> int:
        """
        Returns the regeneration_current used in the self.unblock call. This assumes that all pixels
        have the same regeneration current
        """
        settings = self.connection.promethion_device.get_pixel_settings(pixels=[1])

        lookup = {
            self.prom_msgs.PixelSettings.REGEN_0pA: 0,
            self.prom_msgs.PixelSettings.REGEN_50pA: 50,
            self.prom_msgs.PixelSettings.REGEN_100pA: 100,
            self.prom_msgs.PixelSettings.REGEN_150pA: 150,
            self.prom_msgs.PixelSettings.REGEN_400pA: 400,
            self.prom_msgs.PixelSettings.REGEN_450pA: 450,
            self.prom_msgs.PixelSettings.REGEN_500pA: 500,
            self.prom_msgs.PixelSettings.REGEN_550pA: 550,
            self.prom_msgs.PixelSettings.REGEN_800pA: 800,
            self.prom_msgs.PixelSettings.REGEN_850pA: 850,
            self.prom_msgs.PixelSettings.REGEN_900pA: 900,
            self.prom_msgs.PixelSettings.REGEN_950pA: 950,
            self.prom_msgs.PixelSettings.REGEN_1200pA: 1200,
            self.prom_msgs.PixelSettings.REGEN_1250pA: 1250,
            self.prom_msgs.PixelSettings.REGEN_1300pA: 1300,
            self.prom_msgs.PixelSettings.REGEN_1350pA: 1350,
        }

        return lookup[settings.pixels[0].regeneration_current]

    def get_regeneration_current_voltage_clamp(self) -> float:
        """Gets the regeneration current voltage clamp on the PromethION. Currently when self.unblock is
        called, this voltage will be used. It assumes all pixel blocks have been set with the same voltage and so only
        returns the value from the first pixel block
        """

        stg = self.connection.promethion_device.get_pixel_block_settings()
        return stg.pixel_blocks[1].regen_current_voltage_clamp.value

    def get_unblock_voltage(self) -> float:
        """Gets the unblock voltage on the PromethION. This value is currently not being used on promethion
        It assumes all pixel blocks have been set with the same voltage and so only
        returns the value from the first pixel block

        """

        stg = self.connection.promethion_device.get_pixel_block_settings()
        return stg.pixel_blocks[1].unblock_voltage.value

    #################
    # Other Methods #
    #################

    def clear_saturation(self, overload_mode: str, saturation_control_enabled: bool = True) -> None:

        """
        Resets the hardware saturation flags for all channels and sets the saturation to
        the overload mode passed into the method
        There are 4 over load methods:

            #. 'set_flag' when hardware saturation is triggered this mode sets a flag
                to say saturation has happened but does not change the mux settings
            #. 'latch_off' This will set the saturation flag and turn off the mux
            #. 'limit' This limits the amount of current that can be felt by the
                channel but keeps turning it back on if the limit is exceeded
            #. 'clear' resets all the flags and mux changes related to saturation

        The saturation_control_enabled bool enables or disables the firmware saturation
        layer, when enabled when the hardware saturation is triggered the
        firmware overrides the data coming back from the chip to a known value
        that is intercepted by MinKNOW with this disabled then the data is
        returned unmodified
        :param overload_mode: The type of saturation that will be set, options
        are:
         * 'limit',
         * 'latch_off',
         * 'clear',
         * 'set_flag'
        :param saturation_control_enabled: Bool for whether saturation control is on
                                      once saturation is cleared
        """

        self.set_overload_mode("clear", saturation_control_enabled=False)
        self.set_overload_mode(overload_mode, saturation_control_enabled=saturation_control_enabled)

    def reset_saturation_control(self) -> None:
        """
        Reset the saturation control values to the defaults, which is saturation
        on, and some sensible thresholds
        """

        self.disable_saturation_control()
        self._set_saturation_thresholds(0, 0, 0, 0)
        self.set_saturation_adc(-5, 1900, threshold=640, saturation_during_unblock=False)

    def enable_test_regeneration_current(self) -> None:

        """
        Set the regeneration current on all channels

        :param regen_current: regeneration current
        """

        new_settings = self.prom_msgs.PixelSettings()
        new_settings.regeneration_current_test_enabled.value = True
        self.connection.promethion_device.change_pixel_settings(pixel_default=new_settings)

    def disable_test_regeneration_current(self) -> None:
        """
        Set the regeneration current on all channels

        :param regen_current: regeneration current
        """
        new_settings = self.prom_msgs.PixelSettings()
        new_settings.regeneration_current_test_enabled.value = False
        self.connection.promethion_device.change_pixel_settings(pixel_default=new_settings)
