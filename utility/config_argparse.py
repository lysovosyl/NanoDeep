from __future__ import annotations

import argparse
import ast
import logging
import os
import re
import sys
import warnings
from typing import Any, Optional

import minknow_api.basecaller_pb2
import minknow_api.basecaller_service
import toml
from bream4.device_interfaces.device_wrapper import create_grpc_client
from utility.config_apply import config_apply_to_device, filter_config_for_device
from utility.config_file_utils import create_path, find_path, set_path
from utility.config_loader import CONFIG_DIR, ConfigParseException, load_config

try:
    from production.bias_voltage_offset_lookup import get_bias_voltage_offset_from_server
except ImportError:
    pass


KIT_TOML = os.path.join(os.environ.get("ONT_CONFIG_DIR", "configuration"), "sequencing", "kits.toml")

ARTIC_IDENTIFIER = "SYSTEM:post_processing/artic/artic"
ARTIC_CONFIG = os.path.join("post_processing", "artic", "artic.toml")


def split_args(arg_name: str, args: list[str]) -> tuple[Optional[list[str]], list[str]]:
    """Uses arg_name to cleave args into running args and the remaining_args

    If arg_name == test and args == [a, b, --test, c, --test2]
    ([--c], [a, b, --test2]) will be returned

    If arg_name is not present, (None, args) will be returned

    :param arg_name: Which arg to find
    :param args: List of args to process
    :returns: ([matching args], [remaining args])
    """
    matching = []
    non_matching = []
    index = 0

    if "--" + arg_name in args:
        index = args.index("--" + arg_name)
        non_matching = args[:index]

        index += 1

        # Grab all arguments after needed one until the next is a different command
        while index < len(args) and not args[index].startswith("-"):
            # Transform to pass arguments as --
            matching.append("--" + args[index])
            index += 1

        non_matching.extend(args[index:])
        return matching, non_matching

    else:
        return None, args


def update_read_classifiers(config: dict, kit_config: dict, kit: str) -> None:
    read_classifiers = find_path(
        "analysis_configuration.read_classification.parameters.rules_in_execution_order", config
    )

    if read_classifiers and kit:

        # Get all kits
        kits = {kit}
        barcode_kits = find_path("basecaller_configuration.barcoding_configuration.barcoding_kits", config)
        if barcode_kits:
            kits.update(set(barcode_kits))

        # Find out event length
        adapter_lengths = sum([kit_config[kit].get("adapter_length", 0) for kit in kits])
        barcode_lengths = sum([kit_config[kit].get("barcode_length", 0) for kit in kits])
        events = adapter_lengths + barcode_lengths

        # Adjust classifications
        for idx, classification in enumerate(read_classifiers):
            if "adapter" in classification:
                classification = re.sub(r"\(event_count,lt,.*?\)", f"(event_count,lt,{events+1})", classification)
            if "strand" in classification or "strand" in classification:
                classification = re.sub(r"\(event_count,gt,.*?\)", f"(event_count,gt,{events})", classification)
            read_classifiers[idx] = classification


# Wrap parse_args so that it automagically loads and applies the config
def parse_args_decorator(parse_args):
    def wrapper(*args, **kwargs):
        # Make sure to get device before the first logger call
        # Otherwise the log will default to home directory
        device = create_grpc_client()
        logger = logging.getLogger(__name__)

        # Log the args received
        logger.info("Parsing argv: {} args: {} kwargs: {}".format(sys.argv, args, kwargs))
        args = parse_args(*args, **kwargs)
        config: dict[str, Any] = {"meta": {"context_tags": {}, "protocol": {}}}

        kit_config = toml.load(KIT_TOML)
        kit_config.update(kit_config.pop("expansion_kits", {}))

        # If the config exists: get it, maybe apply it, and attach it to args
        if args.config:
            config = filter_config_for_device(load_config(args.config), device)

        if args.bias_voltage_offset_lookup_server and device.is_promethion:
            offset = get_bias_voltage_offset_from_server(device, args.bias_voltage_offset_lookup_server)
            set_path("device.promethion.bias_voltage_offset", config, offset)
            set_path("meta.protocol.bias_voltage_offset_lookup_server", config, args.bias_voltage_offset_lookup_server)

        #  Overwrite config with options passed in. Until MinKNOW deals with config
        if args.sawtooth_url:
            config["meta"]["protocol"]["sawtooth_url"] = args.sawtooth_url

        if args.experiment_type:
            config["meta"]["protocol"]["experiment_type"] = args.experiment_type

        if args.department:
            set_path("meta.context_tags.department", config, args.department)

        if args.keep_power_on:
            set_path("custom_settings.keep_power_on", config, args.keep_power_on)

        # For any of these arguments, we shall assume a sequencing config has been passed in
        if args.kit:
            set_path("meta.context_tags.sequencing_kit", config, args.kit)

        if args.lamp_kit:
            set_path("basecaller_configuration.lamp_configuration.lamp_kit", config, args.lamp_kit)

        if args.experiment_time:
            config["custom_settings"]["run_time"] = int(args.experiment_time * 60 * 60)
            # context_tags expects in minutes
            config["meta"]["context_tags"]["experiment_duration_set"] = int(args.experiment_time * 60)
        else:
            duration = find_path("custom_settings.run_time", config)
            if duration:
                config["meta"]["context_tags"]["experiment_duration_set"] = duration

        if args.start_bias_voltage:
            config["custom_settings"]["start_bias_voltage"] = int(args.start_bias_voltage)

        if args.fast5:
            create_path("writer_configuration.read_fast5", config)

            if args.fast5 == "on":
                set_path("writer_configuration.read_fast5.raw", config, [[1, 3000]])

            elif args.fast5 == "off":
                # If fast5 is off, There are several things we want to do regardless of config:
                # * Always output skipped reads
                # * If batch_count/file_pattern is set make sure they also carry over

                read_fast5_from_file = config["writer_configuration"].get("read_fast5", {})

                # Make sure to just write skipped reads and give a batch_count so files can be split
                write_skipped_reads = {
                    "disable_writing_force_skipped_reads": False,
                    "disable_writing_passed_reads": True,
                    "disable_writing_failed_reads": True,
                    "batch_count": read_fast5_from_file.get("batch_count", 4000),
                }

                if "file_pattern" in read_fast5_from_file:
                    write_skipped_reads["file_pattern"] = read_fast5_from_file["file_pattern"]

                config["writer_configuration"]["read_fast5"] = write_skipped_reads

        if args.base_calling:
            if args.base_calling == "on":
                config["basecaller_configuration"]["enable"] = True
                # Find the correct guppy filename
                if args.guppy_filename:
                    guppy_filename = args.guppy_filename
                else:
                    guppy_filename = config["meta"]["protocol"]["default_basecall_model"]
                config["basecaller_configuration"]["config_filename"] = guppy_filename

            elif args.base_calling == "off":
                if "basecaller_configuration" in config:
                    config["basecaller_configuration"]["enable"] = False
        else:
            # If we weren't told whether to enable basecalling through flags,
            # Check if the basecaller section is enabled.
            # If it is and there isn't a config, pull it from the default_basecall_model if possible.
            # This is roundabout, but it tries to gather a valid basecall filename
            if find_path("basecaller_configuration.enable", config) is True:
                if not find_path("basecaller_configuration.config_filename", config):
                    guppy_filename = find_path("meta.protocol.default_basecall_model", config)
                    if guppy_filename:
                        config["basecaller_configuration"]["config_filename"] = guppy_filename

        if args.barcoding_kits:
            warnings.warn("--barcoding_kits deprecated. See --barcoding argument", DeprecationWarning)
            set_path("basecaller_configuration.barcoding_configuration.barcoding_kits", config, args.barcoding_kits)

        if args.trim_barcodes:
            warnings.warn("--trim_barcodes deprecated. See --barcoding argument", DeprecationWarning)
            config_path = "basecaller_configuration.barcoding_configuration.trim_barcodes"
            if args.trim_barcodes == "on":
                set_path(config_path, config, True)
            else:
                if find_path(config_path, config):
                    set_path(config_path, config, False)

        if args.barcoding:
            config_path = "basecaller_configuration.barcoding_configuration"

            for entry in args.barcoding:
                label, data = entry.split("=")
                new_data = ast.literal_eval(data)

                if new_data == "off":
                    new_data = False
                elif new_data == "on":
                    new_data = True

                set_path(f"{config_path}.{label}", config, new_data)

        if args.alignment:
            config_path = "basecaller_configuration.alignment_configuration"

            for entry in args.alignment:
                label, data = entry.split("=")
                new_data = ast.literal_eval(data)

                set_path(f"{config_path}.{label}", config, new_data)

        if args.read_splitting:
            for entry in args.read_splitting:
                label, data = entry.split("=")
                if label == "enable":
                    if data == "on":
                        set_path("basecaller_configuration.enable_read_splitting", config, True)
                    elif data == "off":
                        set_path("basecaller_configuration.enable_read_splitting", config, False)
                    else:
                        raise argparse.ArgumentTypeError(f"Invalid read_splitting enable value {data}")
                elif label in ["min_score_read_splitting"]:
                    new_data = ast.literal_eval(data)
                    set_path(f"basecaller_configuration.{label}", config, new_data)
                else:
                    raise argparse.ArgumentTypeError(f"Invalid read_splitting parameter {label}")

        # Dynamically update read classifications based on adapter/strand
        # Needs to be done after args.barcoding
        update_read_classifiers(config, kit_config, args.kit)

        # Update context_tags with basecall information which could come from UI or config file
        if "basecaller_configuration" in config:
            bc = config["basecaller_configuration"]
            # Update local basecalling and filename
            if bc.get("enable"):
                config["meta"]["context_tags"]["local_basecalling"] = 1
                config["meta"]["context_tags"]["basecall_config_filename"] = bc.get("config_filename", "")
            else:
                config["meta"]["context_tags"]["local_basecalling"] = 0

            # Update barcoding information
            barcoding_kits = find_path("barcoding_configuration.barcoding_kits", bc)
            if barcoding_kits:
                config["meta"]["context_tags"]["barcoding_enabled"] = 1
                config["meta"]["context_tags"]["barcoding_kits"] = " ".join(barcoding_kits)

                if any([kit_config[kit].get("artic") for kit in barcoding_kits]):

                    # Not everywhere will have artic, if available then set up post processing
                    try:
                        load_config(os.path.join(CONFIG_DIR, ARTIC_CONFIG))
                        process_request = minknow_api.basecaller_pb2.StartPostProcessingProtocolRequest()
                        process_request.identifier = ARTIC_IDENTIFIER
                        start_request = minknow_api.basecaller_service.StartRequest(
                            start_post_processing_protocol_request=process_request
                        )

                        # Schedule artic pipeline
                        device.connection.protocol.associate_post_processing_analysis_for_protocol(
                            run_id=device.get_protocol_run_id(), start_request=start_request
                        )
                    except ConfigParseException as exc:
                        logger.info(exc)
                        logger.info(
                            "Artic kit specified, but Artic config not found/valid. "
                            + "Not scheduling artic post processing."
                        )

                    # Guppy doesn't need to know about these artic kits
                    # May get swapped to analysis kits in future
                    # The voltrax artic kit is _not_ an expansion kit
                    barcoding_kits = [
                        barcoding_kit for barcoding_kit in barcoding_kits if not kit_config[barcoding_kit].get("artic")
                    ]
                    bc["barcoding_configuration"]["barcoding_kits"] = barcoding_kits

            else:
                config["meta"]["context_tags"]["barcoding_enabled"] = 0

        # If barcoding is disabled or basecalling is disabled, remove barcode from the headers and filepath
        if not find_path("basecaller_configuration.enable", config) or not find_path(
            "basecaller_configuration.barcoding_configuration.barcoding_kits", config
        ):
            header_paths = (
                "writer_configuration.read_fast5.fastq_header_pattern",
                "writer_configuration.read_fastq.header_pattern",
            )

            for header_path in header_paths:
                header = find_path(header_path, config)
                if header:
                    header_components = header.split()
                    new_header = " ".join(filter(lambda x: not x.startswith("barcode"), header_components))
                    set_path(header_path, config, new_header)

            pattern_paths = (
                "writer_configuration.read_fast5.file_pattern",
                "writer_configuration.read_fastq.file_pattern",
                "writer_configuration.read_bam.file_pattern",
            )

            for pattern_path in pattern_paths:
                pattern = find_path(pattern_path, config)
                if pattern:
                    new_pattern = pattern.replace("_{barcode_arrangement}", "")
                    new_pattern = new_pattern.replace("{barcode_arrangement}_", "")
                    set_path(pattern_path, config, new_pattern)

        if args.fast5_data:
            for data in args.fast5_data:
                if "compress" in data:
                    if data == "zlib_compress":
                        c_type = "ZlibCompression"
                    else:
                        c_type = "VbzCompression"
                    config["writer_configuration"]["read_fast5"]["compression_type"] = c_type
                else:
                    config["writer_configuration"]["read_fast5"][data] = [[1, 3000]]

        if args.fast5_reads_per_file:
            config["writer_configuration"]["read_fast5"]["batch_count"] = args.fast5_reads_per_file

        if args.bam:
            if "read_bam" not in config["writer_configuration"]:
                config["writer_configuration"]["read_bam"] = {}

            # Either enable all channels on bam or none
            if args.bam == "on":
                config["writer_configuration"]["read_bam"]["enable"] = [[1, 3000]]
            else:
                del config["writer_configuration"]["read_bam"]

        if args.fastq:
            if "read_fastq" not in config["writer_configuration"]:
                config["writer_configuration"]["read_fastq"] = {}

            # Either enable all channels on fastq or none
            if args.fastq == "on":
                config["writer_configuration"]["read_fastq"]["enable"] = [[1, 3000]]
            else:
                del config["writer_configuration"]["read_fastq"]

        if args.fastq_data:
            for data in args.fastq_data:
                if data == "compress":
                    config["writer_configuration"]["read_fastq"]["compression"] = True

                    # Extend the filename with .gz to indicate it's compressed
                    fn = find_path("writer_configuration.read_fastq.file_pattern", config)
                    if fn and not fn.endswith(".gz"):
                        set_path("writer_configuration.read_fastq.file_pattern", config, fn + ".gz")

        if args.fastq_reads_per_file:
            config["writer_configuration"]["read_fastq"]["batch_count"] = args.fastq_reads_per_file

        if args.generate_bulk_file:
            if args.generate_bulk_file == "on":
                if "bulk" not in config["writer_configuration"]:
                    config["writer_configuration"]["bulk"] = {}
                config["writer_configuration"]["bulk"]["device_metadata"] = True
                config["writer_configuration"]["bulk"]["device_commands"] = True
                config["writer_configuration"]["bulk"]["channel_states"] = [[1, 3000]]
                config["writer_configuration"]["bulk"]["multiplex"] = [[1, 3000]]

            else:
                # If off, no contents should have been specified
                if args.bulk_file_content:
                    raise argparse.ArgumentError(
                        args.generate_bulk_file, "generate_bulk_file=off but bulk_file_content specified"
                    )

                if "bulk" in config["writer_configuration"]:
                    del config["writer_configuration"]["bulk"]

        if args.bulk_file_content:
            for entry in args.bulk_file_content:
                label, data = entry.split("=")
                # Grab a list representation of the string passed in
                new_data = ast.literal_eval(data)
                # Update the label
                if label == "read_table":
                    label = "reads"

                config["writer_configuration"]["bulk"][label] = new_data

        if args.read_filtering:
            config_path = "basecaller_configuration.read_filtering"

            for entry in args.read_filtering:
                label, data = entry.split("=")
                set_path(f"{config_path}.{label}", config, ast.literal_eval(data))

        if args.active_channel_selection == "on":
            cs = config["custom_settings"]
            if "progressive_unblock" in cs:
                cs["progressive_unblock"]["change_mux_after_last_tier"] = True
            if "group_manager" in cs:
                cs["group_manager"]["swap_out_disabled_channels"] = True
                if "global_mux_change" in cs["group_manager"]:
                    cs["group_manager"]["global_mux_change"]["enabled"] = False

        elif args.active_channel_selection == "off":
            cs = config["custom_settings"]
            if "progressive_unblock" in cs:
                cs["progressive_unblock"]["change_mux_after_last_tier"] = False
            if "group_manager" in cs:
                cs["group_manager"]["swap_out_disabled_channels"] = False

                if "global_mux_change" in cs["group_manager"]:
                    cs["group_manager"]["global_mux_change"]["enabled"] = True

                    if args.group_change_period:
                        cs["group_manager"]["global_mux_change"]["interval"] = int(args.group_change_period * 60 * 60)

            # Disable muxscan extra
            if "mux_scan" in cs:
                cs["mux_scan"]["interval"] = 0

        if args.min_read_length is not None:
            set_path("custom_settings.min_read_length_base_pairs", config, args.min_read_length)

        if args.mux_scan_period is not None:
            config["custom_settings"]["mux_scan"]["enabled"] = True
            config["custom_settings"]["mux_scan"]["interval"] = int(args.mux_scan_period * 60 * 60)

        if args.pore_reserve == "on":
            config["custom_settings"]["mux_scan"]["enable_reserve_pore"] = True
        elif args.pore_reserve == "off":
            config["custom_settings"]["mux_scan"]["enable_reserve_pore"] = False

        config = config_apply_to_device(config, device=device)

        args.config = config
        return args

    return wrapper


def escape_str(string: str) -> str:
    return string.replace("\\", "\\\\")


def boolify(arg: str) -> bool:
    # This is here for backwards compatibility, otherwise the store_true flag makes much more sense
    # TODO: Maybe remove in the next major version in favor of store_true
    if arg in {"True", "true", "1", "on"}:
        return True
    elif arg in {"False", "false", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid arg {arg}")


def config_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", help="Path to a configuration file to load and apply", required=False)
    parser.add_argument("--sawtooth_url", help="Internal use. Pass sawtooth url to the script", required=False)
    parser.add_argument(
        "--bias_voltage_offset_lookup_server",
        help="Internal use. Look up bias voltage offset from server",
        required=False,
    )
    parser.add_argument("--department", help="Internal use. Pass department to context_tags", required=False)
    parser.add_argument("--experiment_type", help="Internal use. Override experiment_type in script", required=False)
    parser.add_argument("--flow_cell_id", help="DEPRECATED: No effect.", required=False)
    parser.parse_args = parse_args_decorator(parser.parse_args)

    # Sequencing features. This is horrible. MinKNOW config grpc coming soon
    parser.add_argument("--experiment_time", help="Run time in hours", required=False, type=float)
    parser.add_argument(
        "--start_bias_voltage", help="Bias voltage to start the sequencing with", required=False, type=int
    )

    # Basecall/guppy features
    parser.add_argument(
        "--base_calling", help="Whether to turn on local basecalling", required=False, choices=["on", "off"]
    )
    parser.add_argument(
        "--barcoding_kits",
        help="DEPRECATED: See barcoding. Which barcoding kits to pass to the basecaller",
        required=False,
        nargs="+",
        type=str,
    )
    parser.add_argument(
        "--trim_barcodes",
        help="DEPRECATED: See barcoding. Whether basecaller should trim barcodes",
        required=False,
        choices=["on", "off"],
    )
    parser.add_argument(
        "--barcoding",
        help="What barcoding values to pass to guppy. See barcoding_configuration "
        + "protobuf documentation for allowed values",
        required=False,
        nargs="+",
    )
    parser.add_argument(
        "--alignment",
        help="What alignment parameters to pass to guppy. See alignment_configuration "
        + "protobuf documentation for allowed values",
        required=False,
        nargs="+",
        type=escape_str,
    )
    parser.add_argument(
        "--read_splitting",
        help="What read_splitting parameters to pass to guppy. See basecaller_configuration "
        + "protobuf documentation for allowed values",
        required=False,
        nargs="+",
    )

    # Modify output formats
    parser.add_argument("--bam", help="Whether to output bam", required=False, choices=["on", "off"])
    parser.add_argument("--fast5", help="Whether to output fast5", required=False, choices=["on", "off"])
    parser.add_argument(
        "--fast5_data",
        nargs="+",
        help="What fast5 options to otutput",
        choices=["fastq", "raw", "trace_table", "move_table", "zlib_compress", "vbz_compress"],
        required=False,
    )
    parser.add_argument(
        "--fastq_data", nargs="+", help="What fastq options to otutput", choices=["compress"], required=False
    )
    parser.add_argument("--fast5_reads_per_file", help="How many reads per file", type=int, required=False)
    parser.add_argument("--fast5_reads_per_folder", help="DEPRECATED: No effect.", type=int, required=False)
    parser.add_argument("--fastq", help="Whether to output fastq", required=False, choices=["on", "off"])
    parser.add_argument("--fastq_reads_per_file", help="How many reads per file", type=int, required=False)
    parser.add_argument(
        "--generate_bulk_file", help="Whether to output bulk files", required=False, choices=["on", "off"]
    )
    parser.add_argument("--bulk_file_content", nargs="+", help="What fast5 data to otutput", required=False)
    parser.add_argument(
        "--read_filtering", nargs="+", help="Which filters to cause a read to be in the pass folder", required=False
    )

    # Modify custom settings
    parser.add_argument(
        "--min_read_length",
        type=int,
        help="Pass suggested min read length (in base pairs) to the config",
        required=False,
        choices=[20, 200, 1000],
    )
    parser.add_argument(
        "--mux_scan_period", type=float, help="Run time between mux scans. 0 for disable mux scan", required=False
    )
    parser.add_argument(
        "--active_channel_selection",
        help="Whether to turn on or off active channel selection",
        required=False,
        choices=["on", "off"],
    )
    parser.add_argument("--group_change_period", type=float, help="How often to change groups(hours)", required=False)
    parser.add_argument("--guppy_filename", type=str, help="Guppy filename to use for basecalling", required=False)
    parser.add_argument("--kit", help="Which kit is being run.", type=str, required=False)
    parser.add_argument("--lamp_kit", help="Which lamp kit is being run.", type=str, required=False)
    parser.add_argument(
        "--pore_reserve",
        help="Whether to turn on or off reserve pore feature",
        required=False,
        choices=["on", "off"],
    )
    parser.add_argument(
        "--keep_power_on", help="Internal use only. Allows elec1/2 to keep asic power on", type=boolify, required=False
    )
    return parser
