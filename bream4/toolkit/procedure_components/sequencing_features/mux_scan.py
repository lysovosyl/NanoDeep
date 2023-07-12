"""
Mux Scan 0 mV calibration version

The purpose of the mux scan is to assess the wells in the array and identify the ones that are viable for sequencing
It does this by collecting data for a short period of time in each well and then using the MinKNOW classifications seen
to determine if a well can be used for sequencing.

The original mux scan had a significant problem when it came to doing this process later in the experiment, determining
an appropriate voltage to run the collection period at. Due to mediator drift the voltage of the scan period had to be
determined externally and fed into the mux scan.

The 0 mV calibration introduced in this new scan addresses this limitation by determining the level of drift in the
system prior to doing the scan. It does this by measuring the current response to 3 or more voltages and using the line
of best fit between them to identify the voltage required to achieve 0 pA of current (the intercept). It then uses
the median intercept to plus a fixed voltage for the voltage scan. so for example if to achieve a median current of 0 pA
a voltage offset of -15 mV was required and the config had a scan voltage of -185 mV the scan would be carried out at
-200 mV.

This change in the mux scan allowed for a revamp of the reserve pore feature. The new feature now takes advantage of
this new information. The reserve pore feature was designed to hold in reserve pores that had depleted in mediator at a
faster rate to the main population. The new version of the reserve pore does this by identifying wells that have an
intercept voltage that is shifted significantly away from the median intercept. This has the advantage of only reserving
wells that are depleted compared to the original that always reserved a fixed amount of wells regardless of the deletion
they had undergone.

"""
from __future__ import annotations

import copy
import itertools
import logging
import math
import os
import pprint
from typing import Any, Optional

import numpy as np
import pandas as pd
from minknow_api.data_service import GetReadStatisticsResponse

import bream4.toolkit.procedure_components.voltage_operations as voltage_operations
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.calibration.calibrate_voltage_offset import calibrate_voltage_offset, limit_voltage_offset
from bream4.toolkit.procedure_components.data_extraction.collect_continuous_data import collect_continuous_data
from bream4.toolkit.procedure_components.dict_utils import recursive_merge
from bream4.toolkit.procedure_components.feature_manager import FeatureManager
from bream4.toolkit.procedure_components.output_locations import get_run_output_path
from bream4.toolkit.procedure_components.sequencing_features.mux_scan_keystore import add_mux_scan_result_to_keystore
from bream4.toolkit.procedure_components.sequencing_features.mux_scan_minknow import add_mux_scan_result_to_minknow
from bream4.toolkit.procedure_components.voltage_operations import calculate_nearest_bias_voltage

DEFAULTS = {
    # ---   classify_well   --- #
    "threshold": 2.5,  # well has to spend at least this time in good states
    "states_to_count": ["pore", "strand", "adapter", "event"],
    # ---   report generation   --- #
    "enable_ping": True,  # Whether to ping the mux scan
    "enable_report": True,  # Whether to output data to the log directory
    "enable_report_unpivot_voltage": False,  # If true, will create a point per channel/well/voltage in the report
    # ---   routine settings   --- #
    "collect_0mv_readings": True,
    "collection_time_per_well": 10,  # How long to look at each well for
    "interval": 5400,  # How often the mux scan is performed (So that the GUI can populate the graph) (seconds)
    "initial_global_flick": True,
    # ---   Well Filtering    --- #
    "enable_reserve_pore": False,  # boolean indicating if percentile_cut_off will be used to filter channels
    "reserve_voltage_offset_threshold": -10,
    "realign_offset_for_group_1": True,
    # ---   Channel Ordering   --- #
    "priorities": ["intercept", "total_good_samples"],
    # A list of the metrics to use to priorities the wells of a channel
    "priorities_sort_ascending": [False, False],
    # A list of booleans, one for each priority, the values will be ranked in ascending order if True, descending if
    # False
    # ---   Offset Calibration --- #
    "offset_calibration": {
        "voltage_steps": [-5, -10, -15],
        "muxes": [1, 2, 3, 4],
        "collection_time": 1,
        "threshold_pa": 1.0,
        "minimum_conductance": 0.6,
        "maximum_conductance": 1.4,
        "maximum_adjustment": 300,
        "pass_fraction": 1.0,
        "min_wells": 20,
        "maximum_confidence": 2,
        "minimum_confidence": -2,
    },
    # ---   Global Flick Config   --- #
    "global_flick": {
        "enabled": True,
        "perform_relative_flick": True,
        "rest_voltage": 0,
        "rest_duration": 1.0,
        "flick_duration": 3.0,
        "voltage_gap": 300,
    },
}
# in addition to these headers the internal dataframe will also have the read classifications as heads, these are added
# dynamically on start as they may vary between experiment runs
COLUMNS = [
    # run data
    "channel",
    "well",
    "flow_cell_id",
    "experiment_tag",
    "seconds_since_start_of_run",
    # mux scan run data
    "initial_0mv_current_pa",
    "initial_0mv_current_std",
    "intercept_adjusted_0mv_current_pa",
    "intercept_adjusted_0mv_current_std",
    "group_1_adjusted_offset_0mv_mean",
    "group_1_adjusted_offset_0mv_std",
    "scan_running_voltage_mv",
    "scan_offset_voltage_mv",
    "reserve_pore_intercept_median_mv",
    "group_1_offset_voltage_mv",
    # well data
    "mux_scan_assessment",
    "total_time_active",
    "total_good_samples",
    "pore_median",
    "pore_sd",
    "strand_range",
    "strand_median",
    "include_well",
    "group_order",
    "conductance",
    "intercept",
    "collinear_confidence",
    "conductance_filtered_data",
]

COLUMNS_TO_PING = [
    "channel",
    "well",
    "mux_scan_assessment",
    "include_well",
    "total_good_samples",
    "pore_median",
    "pore_sd",
    "initial_0mv_current_pa",
    "initial_0mv_current_std",
    "total_time_active",
    "intercept_adjusted_0mv_current_pa",
    "intercept_adjusted_0mv_current_std",
    "conductance",
    "intercept",
    "collinear_confidence",
]


class MuxScan(object):
    def __init__(self, device: BaseDeviceInterface, relative_scan_voltage: int, config: dict):
        """
        Sets up the class instance with the relevant tracking and storage information for the class
        :param device: A minknow device API
        :param relative_scan_voltage: The relative voltage to be applied to the scan, The offset voltage is added to
        this value
        :param config: The configuration for the class.
        """

        self.device = device
        self.relative_scan_voltage = relative_scan_voltage
        self.config = copy.deepcopy(DEFAULTS)
        recursive_merge(self.config, config)

        self.logger = logging.getLogger(__name__)

        self.logger.info("Running mux scan with the following config:")
        self.logger.info(pprint.pformat(self.config))
        self.sample_rate = self.device.get_sample_rate()
        self.wells = self.device.get_well_list()

        self.scan_voltage = 0
        self.previous_offset = 0

        self.current_classification_names = self.device.get_read_classification_names()
        self.report_headers = COLUMNS.copy()
        self._voltage_names = [
            [f"initial_voltage_{x}_mv", f"initial_current_{x}_pa"]
            for x in range(len(self.config["offset_calibration"]["voltage_steps"]))
        ]
        for x in self._voltage_names:
            self.report_headers.extend(x)
        self.report_headers += self.current_classification_names
        self.data = []
        self._groups = {}
        self.mux_scan_number = 0

    def _generate_mux_scan_results_df(self) -> pd.DataFrame:
        """
        Generates a pandas dataframe to store all the data collected during the mux scan. Tt has a row for each well
        with all the data columns pre generated.
        :return: pandas Data frame
        :rtype: pd.DataFrame
        """
        seconds_since_start_of_run = int(self.device.get_acquisition_duration())
        time_in_wells = self.device.get_time_in_each_well(self.wells)
        channel_list = self.device.get_channel_list()
        df = pd.DataFrame(itertools.product(channel_list, self.wells), columns=["channel", "well"])
        df = df.reindex(columns=self.report_headers, fill_value=0.0)
        df["flow_cell_id"] = self.device.get_flow_cell_id()
        df["experiment_tag"] = self.device.get_sample_id()
        df["seconds_since_start_of_run"] = seconds_since_start_of_run
        df["mux_scan_assessment"] = "saturated"
        df["include_well"] = df["include_well"].astype(int)
        df["total_good_samples"] = df["total_good_samples"].astype(int)
        df["total_time_active"] = df.apply(
            lambda row: time_in_wells.get(row["channel"], {}).get(row["well"], 0.0), axis=1
        )
        df = df.set_index(["channel", "well"])
        return df

    # classify and generate groups

    def _classify_well(
        self, classifications: dict[str, GetReadStatisticsResponse.PerClassificationData]
    ) -> tuple[str, int]:
        """Given the data for reads from the scan period, determine a classification

        :param classifications: dict of (classification -> read statistics)
        :returns: (classification, count of good reads)
        :rtype: tuple

        """

        if "multiple" in classifications and classifications["multiple"].samples_duration > 0:
            return "multiple", 0

        count = 0
        for classification in self.config["states_to_count"]:
            if classification in classifications:
                count += classifications[classification].samples_duration

        if count >= self.config["threshold"] * self.sample_rate:
            return "single_pore", count

        unavailable_count = 0
        if "unavailable" in classifications:
            unavailable_count = classifications["unavailable"].samples_duration

        zero_count = 0
        if "zero" in classifications:
            zero_count = classifications["zero"].samples_duration

        if unavailable_count > 0 and unavailable_count >= zero_count:
            return "unavailable", count
        if zero_count > 0 and unavailable_count < zero_count:
            return "zero", count

        return "other", int(count)

    def _update_well_classifications_from_scan_data(
        self,
        results_df: pd.DataFrame,
        statistics: GetReadStatisticsResponse,
        disconnected_status: dict[int, bool],
        current_well: int,
    ) -> None:
        """
        Takes the statistics information from the scan period and uses it to update the well information to reflect what
         has been observed, if a pore or strand has been seen additional information is recoded about the current levels
        :param results_df: The storage data frame to be updated with the results of the method call.
        :param statistics: The statistics return from minknow holding the chunk metric data for each classification for
         each channel
        :param disconnected_status: A dict of the channels that have been disconnected during data collection
        :param current_well: The current well number
        """
        # ---- Classify data from each channel/well ---- #

        # map the statistics data collected to channels
        updates = []
        for read_per_channel, channel in zip(statistics.channels, self.device.get_channel_list()):
            channel_updates: dict[str, Any] = {"channel": channel, "well": current_well}

            for reads_per_channel_conf in read_per_channel.configurations:
                well = reads_per_channel_conf.channel_configuration.well
                unblocking = reads_per_channel_conf.channel_configuration.unblock

                # Is saturated/changed or unblocking. Skip.
                if well != current_well or unblocking:
                    continue

                classifications = reads_per_channel_conf.classifications

                # is it saturated?
                if not disconnected_status[channel]:
                    mux_scan_assessment, total_good_samples = self._classify_well(classifications)
                    channel_updates["mux_scan_assessment"] = mux_scan_assessment
                    channel_updates["total_good_samples"] = int(total_good_samples)
                if "pore" in classifications:
                    pore_stats = classifications["pore"].chunk_statistics
                    channel_updates["pore_median"] = round(pore_stats.median, 2)
                    channel_updates["pore_sd"] = round(pore_stats.median_sd, 2)
                else:
                    channel_updates["pore_median"] = np.nan
                    channel_updates["pore_sd"] = np.nan

                if "strand" in classifications:
                    strand_stats = classifications["strand"].chunk_statistics
                    channel_updates["strand_range"] = round(strand_stats.range, 1)
                    channel_updates["strand_median"] = round(strand_stats.median, 2)

                for classification_name in self.current_classification_names:
                    if classification_name in classifications:
                        channel_updates[classification_name] = classifications[classification_name].samples_duration
            updates.append(channel_updates)
        df2 = pd.DataFrame(updates).set_index(["channel", "well"])
        results_df.update(df2)

    def _filter_and_sort_pores_into_scan_groups(self, current_mux_scan_data: pd.DataFrame):
        """
        Takes the updated information and then groups the well into 5 groups, one for each mux for a channel (four main
        sequencing groups) and a disconnected group. The first four groups are the wells that will be viable for
        sequencing during the next sequencing period and will be made p of active pores. The 5th group is a collection
        of the wells that are not viable for sequencing in the next sequencing period and is made up of saturated,
        membrane only (zero classification) wells as well as multiples and blocked wells. If reserve pore is enabled,
        viable wells that are to be reserved are also moved into this group with a classification of reserved pore.

        The wells of a single channel are ordering into the four groups based on the relative distance of their
        intercepts to the median intercept. with the intercepts that are the most positive being given the highest
        priority.

        :param current_mux_scan_data: The storage data frame to be updated with the results of the method call.
        """

        # ------ Filter out wells ----- #
        if self.config["enable_reserve_pore"]:
            active_wells_with_good_intercepts = current_mux_scan_data[
                (current_mux_scan_data.conductance_filtered_data > 0)
                & (current_mux_scan_data.mux_scan_assessment == "single_pore")
            ]
            reserve_pore_intercept_median = active_wells_with_good_intercepts["intercept"].median()
            current_mux_scan_data["reserve_pore_intercept_median_mv"] = reserve_pore_intercept_median
            channels_to_filter = current_mux_scan_data[
                (
                    current_mux_scan_data.intercept
                    < (reserve_pore_intercept_median + self.config["reserve_voltage_offset_threshold"])
                )
                & (current_mux_scan_data.mux_scan_assessment == "single_pore")
            ]
            current_mux_scan_data.loc[channels_to_filter.index, "mux_scan_assessment"] = "reserved_pore"
        current_mux_scan_data.loc[(current_mux_scan_data.mux_scan_assessment == "single_pore"), "include_well"] = int(1)

        # -------- Order wells --------- #

        df = current_mux_scan_data.reset_index()
        df = df[(df["include_well"] == 1)]
        # Sort outside -> channel then mux
        df = df.sort_values(
            ["channel"] + self.config["priorities"], ascending=[True] + self.config["priorities_sort_ascending"]
        )
        # Numbers each mux in the group by by their order (starting from 0 so +1)
        df["group_order"] = df.groupby("channel")["well"].cumcount() + 1
        groups = df.sort_values("group_order").groupby("channel")["well"].apply(list).to_dict()
        for channel in self.device.get_channel_list():
            if channel not in groups:
                groups[channel] = []

        df.set_index(["channel", "well"], inplace=True)
        current_mux_scan_data["group_order"] = df["group_order"]
        current_mux_scan_data["group_order"].fillna(0, inplace=True)
        self._groups = groups

    def get_pore_group(self) -> dict[int, list[int]]:
        """
        Returns the pore groups calculated by the last mux scan period.
        :return: dict(channel-> [wells])
        :rtype: dict
        """
        return self._groups

    def _reassess_offset_for_pore_population(self, current_mux_scan_data: pd.DataFrame) -> None:
        """
        After well classification and ordering the wells selected for group 1 are used to calculate a sequencing offset
         to be used this group is prioritised based on being the furthest away from the median of all viable wells,
         as a result its possible that this groups voltage median is significantly shifted that a new voltage offset
         is appropriate to run with.
        :param current_mux_scan_data:
        """

        # ---- Re-calculate the offset based on good singles ---- #
        # first filter to the channels that have a single pore

        channels_to_use = current_mux_scan_data[
            (current_mux_scan_data.group_order == 1)
            & (
                current_mux_scan_data.conductance.between(
                    self.config["offset_calibration"]["minimum_conductance"],
                    self.config["offset_calibration"]["maximum_conductance"],
                )
            )
            & (
                current_mux_scan_data.collinear_confidence.between(
                    self.config["offset_calibration"]["minimum_confidence"],
                    self.config["offset_calibration"]["maximum_confidence"],
                )
            )
        ]

        if self.config["realign_offset_for_group_1"] and len(channels_to_use) > 0:
            # then calculate the new median intercept
            median_intercept_pores = [
                channels_to_use["initial_current_0_pa"].median(),
                channels_to_use["initial_current_1_pa"].median(),
                channels_to_use["initial_current_2_pa"].median(),
            ]
            self.logger.debug("passing {},into the np.polyfit call".format(median_intercept_pores))

            new_grad, new_intercept = np.polyfit(
                median_intercept_pores, self.config["offset_calibration"]["voltage_steps"], deg=1
            )  # deg=1 -> straight line fit, returns (gradient, intercept)

            # ---- adjust the voltage for device ---- #
            raw_recalculated_offset = calculate_nearest_bias_voltage(
                self.device.get_minimum_voltage_adjustment(), math.floor(new_intercept)
            )

            current_mux_scan_data["group_1_offset_voltage_mv"] = limit_voltage_offset(
                raw_recalculated_offset, DEFAULTS["offset_calibration"]["maximum_adjustment"]
            )
        else:
            current_mux_scan_data["group_1_offset_voltage_mv"] = current_mux_scan_data["scan_offset_voltage_mv"]
            self.logger.info("no channel found when re-evaluating offset")

    # procedural methods

    def _perform_offset_calibration(self, results_df: pd.DataFrame) -> None:
        """
        The procedural code for carrying out the 0mV offset assessment and updating the current scan database.
        :param results_df: the current scan data base

        """
        initial_raw_offset, initial_adjusted_offset, initial_offset_data = calibrate_voltage_offset(
            self.device,
            output_bulk=False,
            config=self.config["offset_calibration"],
            control_acquisition=False,
            default_offset=self.previous_offset,
        )
        self.previous_offset = initial_adjusted_offset

        for (idx, voltage) in enumerate(self.config["offset_calibration"]["voltage_steps"]):
            results_df[f"initial_voltage_{idx}_mv"] = voltage
            results_df[f"initial_current_{idx}_pa"] = initial_offset_data[voltage]

        results_df["conductance"] = initial_offset_data["conductances_ns"]
        results_df["intercept"] = initial_offset_data["intercept_ns"]
        results_df["collinear_confidence"] = initial_offset_data["collinear_confidence"]
        results_df["conductance_filtered_data"] = initial_offset_data["conductance_filtered_data"]
        results_df["scan_offset_voltage_mv"] = initial_adjusted_offset
        results_df["scan_running_voltage_mv"] = initial_adjusted_offset + self.relative_scan_voltage

    def _procedure_to_assess_wells_for_pores(
        self, current_mux_scan_data: pd.DataFrame, feature_manager: Optional[FeatureManager]
    ) -> None:
        """
        The procedural code for scanning each of the wells and collecting the read classification statistics for them,
        The code collects data for each well at a given voltage, this is then followed by an optional global flick
        :param current_mux_scan_data:  the current scan data base
        :param feature_manager: A FeatureManager class that controls features running in parallel with the procedural
         code such as the progressive flicking.
        """
        for well in self.wells:

            self.device.reset_channel_states()
            self.device.set_bias_voltage(0)
            self.device.set_all_channels_to_well(well)
            self.device.set_bias_voltage(self.relative_scan_voltage)

            # Enable feature manager for progressive unblocks etc in the get_read_stats call
            if feature_manager:
                feature_manager.resume_all()

            statistics = self.device.get_read_statistics(
                self.config["collection_time_per_well"],
                completed_read=False,
                include_current_statistics=False,
                include_chunk_statistics=True,
            )

            global_flick_config = self.config["global_flick"]

            if feature_manager:
                feature_manager.stop_all()
            self.device.set_bias_voltage(0)
            if global_flick_config["enabled"]:
                voltage_operations.global_flick(
                    self.device,
                    current_run_voltage=self.relative_scan_voltage,
                    flick_voltage=global_flick_config.get("flick_voltage"),
                    flick_duration=global_flick_config.get("flick_duration"),
                    voltage_gap=global_flick_config.get("voltage_gap"),
                    rest_voltage=global_flick_config.get("rest_voltage"),
                    rest_duration=global_flick_config.get("rest_duration"),
                    perform_relative_flick=global_flick_config.get("perform_relative_flick"),
                )

            disconnected_status = self.device.get_disconnection_status_for_active_wells()
            self.device.set_bias_voltage(0)

            self._update_well_classifications_from_scan_data(
                current_mux_scan_data, statistics, disconnected_status, well
            )

    def all_wells_global_flick(self) -> None:
        """
        Applied a global flick to every well on the array. This is used at the start of the mux scan to clear the wells
        prior to the intercept voltages being calculated.
        """

        global_flick_config = self.config["global_flick"]

        for well in self.wells:
            self.device.set_bias_voltage(0)
            self.device.set_all_channels_to_well(well)
            voltage_operations.global_flick(
                self.device,
                current_run_voltage=self.relative_scan_voltage,
                flick_voltage=global_flick_config.get("flick_voltage"),
                flick_duration=global_flick_config.get("flick_duration"),
                voltage_gap=global_flick_config.get("voltage_gap"),
                rest_voltage=global_flick_config.get("rest_voltage"),
                rest_duration=global_flick_config.get("rest_duration"),
                perform_relative_flick=global_flick_config.get("perform_relative_flick"),
            )

        self.device.set_bias_voltage(0)

    def _measure_current_offset(
        self, current_mux_scan_data: pd.DataFrame, column_name: str, bias_voltage_offset: int
    ) -> None:
        """
        Measures the current with a bias voltage offset applied and updates the database passed in the the results
        :param current_mux_scan_data: The data base t obe updated
        :param column_name: the name of the column to store the data under in the dataframe
        :param bias_voltage_offset: The voltage offset to apply while measuring the offset.
        """
        self.device.set_bias_voltage_offset(int(bias_voltage_offset))

        # ---- collect some new 0mV data with the new offset ---- #
        new_data_container = collect_continuous_data(
            self.device,
            [0],
            [self.config["offset_calibration"]["collection_time"]],
            ["mean", "std"],
            clean_field_names=True,
        )

        current_mux_scan_data[f"{column_name}_0mv_mean"] = new_data_container["offset_0mv_mean"]
        current_mux_scan_data[f"{column_name}_0mv_std"] = new_data_container["offset_0mv_std"]

    # Report methods

    def _report_results(self, current_mux_scan_data: pd.DataFrame) -> None:
        """
        Sends pings updated the keystore and generates CSV outputs for the mux scan just run.
        :param current_mux_scan_data: The data frame to extract the data to report on
        """

        current_mux_scan_data["scan_number"] = self.mux_scan_number

        # report to keystore - will be deprecated soon
        self._report_results_to_keystore(current_mux_scan_data)

        # report to minknow
        self._report_results_to_minknow(current_mux_scan_data)

        # ping to database
        if self.config["enable_ping"]:
            self._mux_scan_ping(current_mux_scan_data)

        # report to csv log
        if self.config["enable_report"]:
            self.report_results_csv(current_mux_scan_data)

    def _mux_scan_ping(self, data: pd.DataFrame) -> None:
        """Sends a ping about this current mux scan
        :param data: Pandas dataframe of data collected in the mux scan
        """

        ping = {
            "mux_scan_assessment_results": mux_scan_assessment_results(data),
            "mux_scan_channel_metrics": self._mux_scan_ping_channel_metrics(data),
            "mux_scan_summary": self.mux_scan_summary(data),
            "scan_interval": int(self.config["interval"]),
            "scan_number": int(self.mux_scan_number),
            "scan_running_voltage_mv": int(self.scan_voltage),
            "scan_offset_voltage_mv": int(data["scan_offset_voltage_mv"][(1, 1)]),
            "group_1_offset_voltage_mv": int(data["group_1_offset_voltage_mv"][(1, 1)]),
            "experiment_duration_seconds": int(self.device.get_acquisition_duration()),
        }

        # Send the ping
        self.device.send_ping_data(ping)

    def _report_results_to_keystore(self, current_mux_scan_data: pd.DataFrame) -> None:
        """
        Reports results of the scan to MinKNOW's keystore for display on the GUI
        :param current_mux_scan_data: Pandas dataframe dataframe of data collected in the mux scan
        """

        #   To the keystore (Picked up by the gui)
        classification_counts = current_mux_scan_data.mux_scan_assessment.value_counts()
        add_mux_scan_result_to_keystore(
            self.device,
            self.config["interval"],
            single_pore=classification_counts.get("single_pore", 0),
            reserved_pore=classification_counts.get("reserved_pore", 0),
            unavailable=classification_counts.get("unavailable", 0),
            multiple=classification_counts.get("multiple", 0),
            saturated=classification_counts.get("saturated", 0),
            zero=classification_counts.get("zero", 0),
            other=classification_counts.get("other", 0),
        )

    def _report_results_to_minknow(self, current_mux_scan_data: pd.DataFrame) -> None:
        """
        Reports results of the scan to MinKNOW for display on the GUI and report generation
        :param current_mux_scan_data: Pandas dataframe dataframe of data collected in the mux scan
        """

        classification_counts = current_mux_scan_data.mux_scan_assessment.value_counts()
        add_mux_scan_result_to_minknow(
            self.device,
            self.config["interval"],
            single_pore=classification_counts.get("single_pore", 0),
            reserved_pore=classification_counts.get("reserved_pore", 0),
            unavailable=classification_counts.get("unavailable", 0),
            multiple=classification_counts.get("multiple", 0),
            saturated=classification_counts.get("saturated", 0),
            zero=classification_counts.get("zero", 0),
            other=classification_counts.get("other", 0),
        )

    def mux_scan_summary(self, data: pd.DataFrame) -> dict[str, float]:
        """
        Summaries wells selected/open pore statistics for the ping generation
        :param data: Pandas dataframe
        :returns: dict of summary info
        """
        good_data = data[(data.mux_scan_assessment == "single_pore")]

        summary = {
            "run_time_voltage": self.scan_voltage,
            "single_open_pore_mean": round(good_data["pore_median"].mean(), 2),
            "single_open_pore_std": round(good_data["pore_median"].std(), 2),
            "total_wells_selected": data[(data.include_well == 1)].shape[0],
        }

        if np.isnan(summary["single_open_pore_mean"]):
            summary["single_open_pore_mean"] = 0
        if np.isnan(summary["single_open_pore_std"]):
            summary["single_open_pore_std"] = 0

        group = data[(data.mux_scan_assessment == "single_pore")].groupby("channel").size()
        for well in self.device.get_well_list():
            summary["multiplexing_group_" + str(well)] = group[group >= well].size

            # To the UI messages
        group_count = {
            group: len([1 for x in self._groups.values() if len(x) >= group]) for group in self.device.get_well_list()
        }
        reserved_pores = len(data[data.mux_scan_assessment == "reserved_pore"])

        msg = "mux_scan_result"
        if self.device.is_flongle:
            msg = "channel_scan_result"

        self.logger.log_to_gui(  # type: ignore
            msg,
            params={
                "flow_cell_id": self.device.get_flow_cell_id(),
                "total_pores": sum(group_count.values()) + reserved_pores,
                "num_pores": group_count[1],
            },
        )

        return summary

    def report_results_csv(self, current_mux_scan_data: pd.DataFrame) -> pd.DataFrame:

        """
        Write the DataFrame to the reads folder if possible, else the logs directory

        :param current_mux_scan_data: Pandas DataFrame
        :return: Pandas DataFrame of what was output
        """
        report_path = get_run_output_path(self.device, "pore_scan_data.csv")

        # Append to CSV. Only write header if this file doesn't yet exist
        self.logger.info(f"logging data to {report_path}")

        if self.config["enable_report_unpivot_voltage"]:
            # currently the DB has a single line per channel/well, this expands this to the number of voltages used
            # for ease of analysis this is converted into a row for each measurement used in the calculations for
            # the intercept.

            current_mux_scan_data["point"] = list(
                zip(
                    *(
                        zip(current_mux_scan_data[initial_mean], current_mux_scan_data[initial_voltage])
                        for (initial_voltage, initial_mean) in self._voltage_names
                    )
                )
            )
            # Make new row for every entry in point this should mean 3 lines per well.
            current_mux_scan_data = current_mux_scan_data.explode("point")
            current_mux_scan_data["mean"], current_mux_scan_data["voltage"] = zip(*current_mux_scan_data.point)

        current_mux_scan_data.to_csv(report_path, mode="a", header=not os.path.exists(report_path))
        return current_mux_scan_data

    def _mux_scan_ping_channel_metrics(self, data: pd.DataFrame) -> list[dict]:
        """
        Generates the per well component of the ping
        :param data:Pandas dataframe
        """
        ping_data = data.reset_index()
        cols = COLUMNS_TO_PING.copy()
        for x in self._voltage_names:
            cols.extend(x)
        ping_data = ping_data[cols]
        ping_data["total_good_samples"] = ping_data["total_good_samples"].astype(int)
        return [d[1].dropna().to_dict() for d in ping_data.iterrows()]  # type: ignore pandas funny business

    # Main run method
    def run_mux_scan(self, feature_manager: Optional[FeatureManager]) -> None:
        """
        Executes the mux scan procedure, main running nanodeep.
        :param feature_manager: A FeatureManager class that controls features running in parallel with the procedural
         code such as the progressive flicking.
        """
        self.mux_scan_number += 1
        if feature_manager:
            feature_manager.stop_all()

        # do a global flick for each well
        # skip the first scan, this flick should be done at a relative level, the first flick of an experiment has not
        # established the offset and so it is safer to skip this at this point
        if self.config["initial_global_flick"] and self.mux_scan_number > 1:
            self.all_wells_global_flick()

        current_mux_scan_data = self._generate_mux_scan_results_df()

        # reset the bias voltage offset
        self.device.set_bias_voltage_offset(0)

        if self.config["collect_0mv_readings"]:
            # collect 0mV data with no offset applied
            self._measure_current_offset(current_mux_scan_data, "offset", 0)

        # do initial offset calibration
        self._perform_offset_calibration(current_mux_scan_data)

        if self.config["collect_0mv_readings"]:
            # assess the offset with the global pore calculated offset applied
            self._measure_current_offset(
                current_mux_scan_data,
                "intercept_adjusted_0mv_current_pa",
                current_mux_scan_data["scan_offset_voltage_mv"][(1, 1)],
            )

        # scan wells to identify wells with pores
        self._procedure_to_assess_wells_for_pores(current_mux_scan_data, feature_manager)

        # filter, reserve and order the pores into groups
        self._filter_and_sort_pores_into_scan_groups(current_mux_scan_data)

        # recalculate the offset for the wells that have been put into the first group
        self._reassess_offset_for_pore_population(current_mux_scan_data)

        if self.config["collect_0mv_readings"]:
            # assess the 0mV again for the final run voltage
            self._measure_current_offset(
                current_mux_scan_data,
                "group_1_adjusted_offset",
                current_mux_scan_data["group_1_offset_voltage_mv"][(1, 1)],
            )

        self.scan_voltage = int(current_mux_scan_data["group_1_offset_voltage_mv"][(1, 1)]) + self.relative_scan_voltage

        self.device.set_all_channels_to_well(0)
        self.device.reset_channel_states()

        # report the results
        self._report_results(current_mux_scan_data)

        # ---- Clean up/report/return ---- #
        empty_channels = [x for x, y in self._groups.items() if not y]
        self.device.lock_channel_states(empty_channels, "no_pore")

        if feature_manager:
            feature_manager.resume_all()


def mux_scan_assessment_results(data: pd.DataFrame) -> dict[str, int]:
    """Count up the counts of the different assessments

    :param data: Pandas dataframe
    :returns: dict of counts of classifications

    """
    defaults = {
        "multiple": 0,
        "other": 0,
        "saturated": 0,
        "single_pore": 0,
        "reserved_pore": 0,
        "unavailable": 0,
        "zero": 0,
    }

    assessment = data["mux_scan_assessment"].value_counts().to_dict()
    defaults.update(assessment)
    defaults = {k: int(v) for (k, v) in defaults.items()}
    return defaults
