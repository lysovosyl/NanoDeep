from __future__ import annotations
import argparse
import ast
import json
import logging
import os
import pprint
import signal
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import pandas as pd
from bream4.device_interfaces import device_wrapper
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface, get_env_grpc_port
from bream4.toolkit.procedure_components.output_locations import get_run_output_path
from google.protobuf import json_format
from grpc import RpcError
from minknow_api import protocol_service
from read_until import AccumulatingCache, ReadUntilClient
from read_until.read_until_utils import WatchAcquisitionStatus, wait_for_sequencing_to_start
from typing_extensions import TypeAlias
from utility.config_argparse import boolify
from utility.config_file_utils import expand_grouped_list
from read_deep.rt import rt_deep
from importlib import import_module
import yaml
import torch


def check_mem(cuda_device):
    devices_info = os.popen(
        '"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split(
        "\n")
    total, used = devices_info[int(cuda_device[-1])].split(',')
    return total, used


def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = 2560
    used = int(used)
    max_mem = int(total * 0.4)
    block_mem = max_mem - used
    x = torch.cuda.FloatTensor(256, 1024, block_mem)
    del x


"""
Basic Read-Until script.

Note that because this uses a bream device_wrapper you can do log_to_gui etc.
The logs will also appear in the relevant device position logs.

If any log output is generated, due to how log rotation works on windows vs unix,
 * a separate log file (bream-0) will be created on unix
 * the same bream log as the main process will be used on Windows.

Due to the fact there is a bream process already interacting with the device,
please alter this at your own peril.

The current implementation of bream does no flicking of strand so we should be
safe to do this.
"""

GUPPY_PARAMS_RENAME = {
    # minknow -> guppy
    "barcoding_kits": "barcode_kits",
    "min_score": "min_score_barcode_front",
    "min_score_rear": "min_score_barcode_rear",
    "min_score_mid": "min_score_barcode_mid",
    "min_score_mask": "min_score_barcode_mask",
}

ReadType: TypeAlias = "dict[str, Any]"  # The read that comes back from pyguppy


class ReadUntil(threading.Thread):
    """This class is the meat of Read Until. This is the class that hooks in to the read until api
    and basecalls reads and determines whether to flick or accept the reads based on configuration passed in.

    * run is the main method that is on a separate thread and does all the lifting.
    * _enrich/_deplete get called depending on config to give a response of what to do
        - unblock/unblock_hit_outside_bed/stop_receiving/stop_receiving_hit_outside_bed/no_decision can be returned

    """

    def __init__(
            self,
            config: dict[str, Any],
            device: BaseDeviceInterface,
            read_until_client: ReadUntilClient,
            nanodeepclient: rt_deep,
            barcode_counts: Optional[dict[str, int]] = None,
    ):
        self.logger = logging.getLogger(__name__)

        self.keep_going = True
        self.read_until_client = read_until_client
        self.nanodeepclient = nanodeepclient
        self.flick_time = (
                time.monotonic() + time.monotonic()
        )

        self.config = config
        if config["filter_type"] not in ["enrich", "deplete"]:
            raise ValueError(f"{config['filter_type']} not enrich, deplete, or barcode_balance")
        self.mode = config["filter_type"]
        self.filter_which = int(config["filter_which"])

        self.report_read_path = config["report_read"]
        if self.report_read_path:
            self.report_read_path = get_run_output_path(device, "adaptive_sampling.csv")
            self.exists = os.path.exists(self.report_read_path)

        super().__init__()

    def _enrich(self, result) -> str:
        """Figures out what to do with a read if trying to enrich from the ref/bed file.

        It accepts a read from guppy - This assumes it is a dict with metadata as a key
        where the relevant information can be accessed.

        Returns a string of the decision (unblock*/stop_receiving*/no_decision)
        """
        if result[self.filter_which] > 0.5:
            return "no_decision"

        return "unblock"

    def _deplete(self, result) -> str:
        """Figures out what to do with a read if trying to enrich from the ref/bed file.

        It accepts a read from guppy - This assumes it is a dict with metadata as a key
        where the relevant information can be accessed.

        Returns a string of the decision (unblock*/stop_receiving*/no_decision)
        """
        if result[self.filter_which] > 0.5:
            return "unblock"

        return "no_decision"

    def run(self) -> None:
        """Constantly:

        1. Pulls the latest reads from the read until api client.
        2. Basecalls them
        3. For each read, figures out whether to flick/accept/leave
        4. Potentially write reports

        To trigger a stop, call self.stop()

        """
        accept_num = 0
        reject_num = 0
        read_basecalled_count = 0
        read_chunk_count = 0
        while self.keep_going:

            batch_time = time.time()  # Used to track acquisition timings in CSV - Needs absolute time
            start_time = time.monotonic()  # Everything else tracked as monotonic to ignore clock changes

            # get the most recent read chunks from the client
            read_chunks = self.read_until_client.get_read_chunks(batch_size=self.config["batch_size"], last=True)

            # Move any "bad" chunks into a separate list so not basecalled
            bad_chunks = []

            read_chunk_count += len(read_chunks)
            chunk_time = time.monotonic()

            called_batch = []
            raw_data_batch = []
            for channel, read in read_chunks:
                reads = {}
                reads['channel'] = channel
                reads['number'] = read.number
                reads['read_id'] = read.id
                called_batch.append(reads)
                signal = np.frombuffer(read.raw_data, self.read_until_client.signal_dtype)
                signal = (signal - np.average(signal)) / np.var(signal)
                raw_data_batch.append(signal)

            if len(raw_data_batch) == 0:
                time.sleep(0.05)
                continue
            result = self.nanodeepclient.signal_classification(raw_data_batch)
            for index in range(len(result)):
                called_batch[index]['result'] = result[index]

            read_until_data = []
            unblock_reads = []
            stop_receiving_reads = []

            for reads in called_batch:
                result = reads['result']
                channel = reads['channel']
                read_number = reads['number']

                read_basecalled_count += 1
                if channel > 256:
                    continue

                if self.mode == "enrich":
                    decision = self._enrich(result)
                elif self.mode == "deplete":
                    decision = self._deplete(result)
                else:
                    assert False  # nosec Type checking helper

                if decision.startswith("unblock"):
                    unblock_reads.append((channel, read_number))
                    reject_num += 1
                else:
                    accept_num += 1

                if self.report_read_path:
                    read_until_data.append(
                        [
                            0,
                            read_number,
                            channel,
                            0,
                            reads["read_id"],
                            0,
                            decision,
                        ]
                    )

            print('\r', 'accpet num:', accept_num, 'reject num:', reject_num, end=' ')
            time_to_make_decisions = time.monotonic()

            if unblock_reads:
                self.read_until_client.unblock_read_batch(unblock_reads, duration=self.config["unblock_duration"])
            if stop_receiving_reads:
                self.read_until_client.stop_receiving_batch(stop_receiving_reads)
            time_to_send_decisions = time.monotonic()

            if read_until_data:
                columns = [
                    "batch_time",
                    "read_number",
                    "channel",
                    "num_samples",
                    "read_id",
                    "sequence_length",
                    "decision",
                ]

                df = pd.DataFrame(read_until_data, columns=columns)
                df["time_to_send_decisions"] = round(
                    (time_to_send_decisions - time_to_make_decisions) / len(read_until_data), 3)
                df.to_csv(self.report_read_path, mode="a", header=not self.exists, index=False)
                self.exists = True

    def stop(self) -> None:
        """Signals to the thread (run method) to stop.

        This can take a while to actually stop because basecalling for that round
        will need to be finished before the stop signal will take effect
        """
        self.keep_going = False


class ReadUntilManager:
    """Class used to start/stop ReadUntil process as necessary"""

    def __init__(self, device: BaseDeviceInterface, custom_settings: dict[str, Any]):
        self.device = device
        self.custom_settings = custom_settings
        self.read_until_process = None
        self.logger = logging.getLogger(__name__)
        self.keep_going = True
        self.watch_acquisition_status = None
        self.watch_protocol_stream = None
        self.barcode_counts = defaultdict(int)  # Used to cache between read until starts
        self._lock = threading.RLock()
        self.read_until_client = None
        self.custom_settings = custom_settings

        if self.custom_settings["model_config"] == None:
            model_args = getattr(import_module('read_deep.model_config.defaultconfig'),
                                 self.custom_settings["model_name"])
        else:
            ymlfile = open(self.custom_settings["model_config"], 'r', encoding='utf-8')
            model_args = yaml.load(ymlfile, Loader=yaml.SafeLoader)

        deepmodel = getattr(import_module('read_deep.model.' + self.custom_settings["model_name"]), 'model')
        self.weight_path = self.custom_settings["weight_path"]
        model = deepmodel(**model_args)
        self.nanodeep_client = rt_deep(
            model,
            self.custom_settings["signal_length"],
            device=custom_settings['device']
        )
        self.nanodeep_client.load_the_model_weights(self.weight_path)

    def _get_barcoding_connection_options(self) -> dict[str, Any]:
        # Gets the barcoding parameters to pass to guppy
        # Retrieves the options from MinKNOW
        basecall_info = self.device.connection.analysis_configuration.get_basecaller_configuration()
        basecall_config = json.loads(
            json_format.MessageToJson(
                basecall_info, preserving_proto_field_name=True, including_default_value_fields=True
            )
        )

        opts = basecall_config.get("barcoding_configuration", {})
        if opts:
            for (key, new_key) in GUPPY_PARAMS_RENAME.items():
                if key in opts:
                    opts[new_key] = opts.pop(key)

            if "trim_barcodes" in opts:
                # Guppy negated option
                opts["disable_trim_barcodes"] = not opts.pop("trim_barcodes")

            if "min_score_barcode_front" in opts and "min_score_barcode_rear" not in opts:
                opts["min_score_barcode_rear"] = opts["min_score_barcode_front"]

        if "barcoding_kits" in self.custom_settings:
            opts["barcode_kits"] = self.custom_settings["barcoding_kits"]

        return opts

    def _exit(self, *args) -> None:
        # Can get called from multiple places as it's a registered handler
        with self._lock:
            self.logger.info("Received exit signal")
            self.keep_going = False

            if self.watch_protocol_stream:
                self.watch_protocol_stream.cancel()
            if self.watch_acquisition_status:
                self.watch_acquisition_status.stop()

            self._stop_read_until()

    def _stop_read_until(self) -> None:
        # To be safe also wrap the one this calls in a lock (Rlock so can reenter for free)
        with self._lock:
            if self.read_until_process:
                self.logger.info("Stopping read until process")
                self.read_until_process.stop()

                if self.read_until_process:
                    self.logger.info("Joining read until process")
                    self.read_until_process.join()

                if self.read_until_client:
                    self.logger.info("Resetting read until client")
                    self.read_until_client.reset()
                    self.read_until_client = None

                self.logger.info("Read until process ended")
                self.read_until_process = None

    def _start_read_until(self) -> None:
        with self._lock:
            if self.read_until_process is None:
                # Make sure any reads waiting in basecaller are drained as we don't need them
                self.logger.info("Draining any straggling basecaller reads")

                self.read_until_client = ReadUntilClient(
                    mk_host="localhost",
                    mk_port=get_env_grpc_port(),
                    cache_type=AccumulatingCache,
                    one_chunk=False,
                    filter_strands=True,
                    calibrated_signal=False,
                )
                self.read_until_client.run(
                    first_channel=self.custom_settings.get("first_channel", 1),
                    last_channel=self.custom_settings.get("last_channel", self.device.channel_count),
                    max_unblock_read_length_samples=self.custom_settings.get("max_unblock_read_length_samples", 0),
                    accepted_first_chunk_classifications=self.custom_settings.get(
                        "accepted_first_chunk_classifications", []
                    ),
                )

                self.logger.info("Starting read until process")
                self.read_until_process = ReadUntil(
                    self.custom_settings,
                    self.device,
                    self.read_until_client,
                    self.nanodeep_client,
                    barcode_counts=self.barcode_counts,
                )
                self.read_until_process.start()

    def run(self) -> None:
        """This coordinates the start/stop of the ReadUntil class.

        It will ensure sequencing has started (Wait for a sequencing process to start and be acquiring data)
        And then follow that run.

        There are several things that can cause this class/method to exit:
        1. Hook into sigterm to allow read until processes to exit
        2. Terminates if acquisition status of the run changes from RUNNING
        3. Receive a CANCELLED/ABORTED

        There are 2 modes to running:
        * config["run_in_mux_scan"] = True -> Just run read until until one of the exit methods above get triggered
        * = False, Start ReadUntil only outside of mux scans (Watches the keystore to figure this out)
        """
        # Attach triggers to stop
        signal.signal(signal.SIGTERM, self._exit)
        print('waitting sequencing to start')
        wait_for_sequencing_to_start(self.device)

        # Move this to after waiting for sequencing to start to ensure all config options have been applied
        self.watch_acquisition_status = WatchAcquisitionStatus(self.device, self._exit)

        self.watch_acquisition_status.start()

        while self.keep_going:

            if not self.custom_settings["run_in_mux_scan"]:

                # Now watch the phase to start/stop the read until feature
                try:
                    self.watch_protocol_stream = self.device.connection.protocol.watch_current_protocol_run()
                    for msg in self.watch_protocol_stream:
                        self.logger.info("Received state %s" % (protocol_service.ProtocolPhase.Name(msg.phase),))

                        if msg.phase in {
                            protocol_service.PHASE_MUX_SCAN,
                            protocol_service.PHASE_PREPARING_FOR_MUX_SCAN,
                        }:
                            self.logger.info("Stopping Read Until as entering a mux scan")
                            self._stop_read_until()

                        elif msg.phase == protocol_service.PHASE_SEQUENCING:
                            self.logger.info("Starting Read Until as starting sequencing")
                            print('start sequencing to start')
                            self._start_read_until()

                except RpcError as exception:
                    self.logger.info(f"Received exception {exception} on stream")

                    code = exception.code()

                    # Check if it wasn't a blip
                    if code == code.CANCELLED or code == code.ABORTED:
                        self._stop_read_until()
                        return

            else:
                self.logger.info("Starting Read Until")
                print('start sequencing to start')
                self._start_read_until()
                self.read_until_process.join()  # type: ignore
                self._stop_read_until()
                return


def main(device: Optional[BaseDeviceInterface] = None) -> None:
    args = parse_args(sys.argv[1:])
    config = vars(args)
    if device is None:
        device = device_wrapper.create_grpc_client()

    if config["last_channel"] is None:
        config["last_channel"] = device.channel_count

    logger = logging.getLogger(__name__)
    logger.info(pprint.pformat(config))

    read_until_manager = ReadUntilManager(device, config)
    read_until_manager.run()


def eval_wrapper(string: str) -> Any:
    # Literal eval will gobble up an escaped sequence, so make sure to double escape
    return ast.literal_eval(string.replace("\\", "\\\\"))


def parse_barcode_list(string: str) -> list[str]:
    # Given: "[[1, 3], [5, 5], 7]" return [barcode01, barcode02, barcode03, barcode05, barcode07]"
    group_list = eval_wrapper(string)
    expanded_group_list = expand_grouped_list(group_list)

    return [f"barcode{x:02}" for x in expanded_group_list]


def parse_args(args: list) -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--filter_type", default="enrich", choices=["enrich", "deplete"])
    parser.add_argument("--filter_which", required=True)
    parser.add_argument("--weight_path", required=True, type=str)
    parser.add_argument('--model_config', default=None)
    parser.add_argument('--model_name', default='nanodeep', help='It should be nanodeep')
    parser.add_argument("--batch_size", default=512, help="How many new reads to grab from read until client", type=int)
    parser.add_argument("--report_read", default=True, type=boolify, help="Whether to output read decisions")
    parser.add_argument("--run_in_mux_scan", default=False, type=boolify, help="Whether to run read until in mux scans")
    parser.add_argument("--device", default='cuda:0', type=str, help="use which device")
    parser.add_argument("--signal_length", default=5000, type=int, help="use which device")
    parser.add_argument(
        "--max_unblock_read_length_samples",
        type=int,
        default=0,
        help="Maximum read length MinKNOW will attempt to unblock (in samples)",
    )
    parser.add_argument(
        "--accepted_first_chunk_classifications",
        nargs="*",
        default=["strand", "adapter"],
        help="RU will only stream reads that start with one of these classifications. "
             + "All others will be _accepted_ + not streamed",
    )
    parser.add_argument(
        "--first_channel",
        type=int,
        default=1,
        help="Start of the range of channels that read until will work with. Defaults to all channels",
    )
    parser.add_argument(
        "--last_channel",
        type=int,
        help="End of the range of channels that read until will work with. Defaults to all channels",
    )
    parser.add_argument("--unblock_duration", type=float, default=0.1, help="How long to unblock reads for")
    return parser.parse_args(args)


if __name__ == "__main__":
    main()
