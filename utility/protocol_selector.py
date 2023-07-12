from __future__ import annotations

import argparse
import logging
import os
import subprocess  # nosec - needed
import sys
from itertools import product
from typing import Optional, Union

from bream4.device_interfaces import device_wrapper
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.procedure_components.command.keystore import set_state, wait_for_protocol_trigger
from minknow.paths import minknow_base_dir
from minknow.protocols import ProtocolHandler, build_protocol_handler_argparser
from utility.config_apply import filter_config_for_device
from utility.config_argparse import split_args
from utility.config_loader import ConfigParseException, load_config, load_raw_toml

CONFIG_DIR = str(os.environ.get("ONT_CONFIG_DIR", "configuration"))
USER_CONFIG_DIR = str(os.environ.get("ONT_USER_DIR", ""))

CONFIG_TYPES_TO_ADVERTISE = {"protocol", "workflow"}


def compatible_flow_cell(device: BaseDeviceInterface, flow_cell_config: dict, flow_cell: str) -> bool:
    """Given a device and a flow cell name, return whether that flow cell
    is compatible with the device

    :param device: MinKNOW device wrapper
    :param flow_cell_config: map of flow cell IDs to config settings (flow_cells.toml)
    :param flow_cell: str of flowcell id
    :returns: if flowcell is valid for device
    :rtype: bool

    """

    logger = logging.getLogger(__name__)

    if flow_cell == "INTERNAL":
        # HACK - this is a magic value used by one production script to make sure it's always
        # visible
        return True

    try:
        config = flow_cell_config[flow_cell]
    except KeyError:
        logger.warning("Unknown flow cell product code %s", flow_cell)
        return False

    try:
        connector = config["connector"].lower()
    except KeyError:
        logger.warning("Missing connector type for flow cell %s", flow_cell)
        return False

    if connector == "promethion":
        return device.is_promethion
    if connector == "flongle":
        return device.is_flongle
    if connector == "minion_mk1":
        return device.is_minion_like and not device.is_flongle
    if connector == "any":
        return True

    logger.warning("Unknown connector type %s for flow cell %s", connector, flow_cell)
    return False


class ProtocolSelector(ProtocolHandler):
    """The protocol selector path is invoked by minknow to inspect the bream scripts
    Protocol selector lists the scripts within a particular location,
    generates a name, a guid and tags for the script
    Protocol selector exposes to options
    list returns the {name,id and tags} for each script it controls

    """

    def _load_and_advertise_config(self, config_path: str, base: str) -> None:
        try:
            config = load_config(config_path, base=base)
            config = filter_config_for_device(config, device=self.device)

            if "type" in config and config["type"] not in CONFIG_TYPES_TO_ADVERTISE:
                return

            flow_cells = config["meta"]["protocol"].pop("flow_cells", [])
            kits = config["meta"]["protocol"].pop("kits", [])

            config["meta"]["protocol"]["config path"] = config_path
            config["meta"]["protocol"]["base import"] = base

            # Get a name to advertise that could appear in the UI
            # Get relative from the import path and remove the .toml extenison
            advertise_name = os.path.relpath(config_path, base)[:-5].replace("\\", "/")

            # If no kits, it's a custom script.
            # Don't advertise for all combinations
            if not kits:
                if flow_cells:
                    # For every flowcell, if it's possible, advertise it
                    for flow_cell in flow_cells:
                        if compatible_flow_cell(self.device, self.flow_cell_config, flow_cell):
                            # Update to this flowcell and generate tags/hashes
                            config["meta"]["protocol"]["flow_cell"] = flow_cell
                            tags = self._transform_tags(config["meta"]["protocol"])
                            config_id = "{}:{}".format(advertise_name, flow_cell)

                            self._add_protocol(name=advertise_name, identifier=config_id, tags=tags)
                else:
                    config_id = advertise_name
                    tags = self._transform_tags(config["meta"]["protocol"])
                    self._add_protocol(name=advertise_name, identifier=config_id, tags=tags)
                return

            for (flow_cell, kit) in product(flow_cells, kits):
                if compatible_flow_cell(self.device, self.flow_cell_config, flow_cell):
                    if kit in self.kit_config:
                        kit_info = self.kit_config[kit]
                        config["meta"]["protocol"]["flow_cell"] = flow_cell
                        config["meta"]["protocol"]["kit"] = kit
                        config["meta"]["protocol"]["lamp_kit"] = kit_info.get("lamp_kit", False)
                        config["meta"]["protocol"]["kit_category"] = kit_info["filters"]
                        config["meta"]["protocol"]["barcoding"] = kit_info["barcoding"]
                        config["meta"]["protocol"]["barcoding_kits"] = kit_info.get("expansion_kits", [])
                        config_id = "{}:{}:{}".format(advertise_name, flow_cell, kit)

                        self._add_protocol(
                            name=advertise_name,
                            identifier=config_id,
                            tags=self._transform_tags(config["meta"]["protocol"]),
                        )
                    else:
                        self.logger.info(f"Kit {kit} in {advertise_name} not present in kits.toml")

        except Exception as exc:
            self.logger.log_to_gui("protocol_selector.parse_error", params=dict(filename=config_path, exc=exc))

    def __init__(self):
        """the constructor lists the scripts within a path and builds a list with {name, id,
        tags} for each script

        """
        super(ProtocolSelector, self).__init__()

        # Grab flow cell types so we can filter on connector type
        try:
            self.flow_cell_config = load_raw_toml(os.path.join(CONFIG_DIR, "flow_cells.toml"))
        except ConfigParseException:
            self.flow_cell_config = {}

        # Grab kit categories to advertise to GUI
        # If the file doesn't exist, just return an empty dictionary
        try:
            self.kit_config = load_raw_toml(os.path.join(CONFIG_DIR, "sequencing", "kits.toml"))
        except ConfigParseException:
            self.kit_config = {}

        self.device = device_wrapper.create_grpc_client()
        self.logger = logging.getLogger(__name__)

        # Firstly, do the recipes folder
        recipe_path = os.path.join(minknow_base_dir(), "python", "recipes")
        if os.path.isdir(recipe_path):
            for filename in os.listdir(recipe_path):
                name, extension = os.path.splitext(filename)

                if extension != ".py":
                    continue

                self._add_protocol(
                    name="recipe/{}".format(name),
                    identifier=filename,
                    tags={"config path": os.path.join(recipe_path, filename)},
                )

        # Then do configs recursively from the config dir
        for (root, directory, files) in os.walk(CONFIG_DIR):
            for filename in files:
                name, extension = os.path.splitext(filename)
                relative_path = os.path.join(root, filename)

                # Only want to process files that are configuration settings
                if extension != ".toml" or name == "":
                    continue

                self._load_and_advertise_config(relative_path, CONFIG_DIR)

        # Then do configs recursively from the user dir
        if USER_CONFIG_DIR:
            for (root, directory, files) in os.walk(USER_CONFIG_DIR):
                for filename in files:
                    name, extension = os.path.splitext(filename)
                    relative_path = os.path.join(root, filename)

                    # Only want to process files that are configuration settings
                    if extension != ".toml" or name == "":
                        continue

                    self._load_and_advertise_config(relative_path, USER_CONFIG_DIR)

    @classmethod
    def _transform_tags(cls, tag_dict: dict) -> dict:
        ret = {}
        for (key, value) in tag_dict.items():
            if key != "kit_category":
                ret[key.replace("_", " ")] = value
            else:
                ret[key] = value
        return ret

    def _run_config(self, config_path: str, base_import: str, extra_args: list[str]) -> int:
        """Runs a python script with a given config and any extra args
        This waits for the python process to finish.

        If there are any processes specified with ::
        [custom_processes.blah]
        script = "blah.py"

        They will all get run in parallel to the script
        """

        # Load config and find associated script
        config = load_config(config_path)
        config = filter_config_for_device(config, device=self.device)

        script = config["script"]

        # If script not present (Could be from user dir) try with base dir
        if not os.path.exists(os.path.join(base_import, script)):
            script = os.path.join(CONFIG_DIR, script)

        # Spawn off any background processes specified
        background_ps, extra_args = self._run_python_sections(
            config.get("custom_processes", {}), base_import, extra_args, parallel=True
        )

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join([base_import, env.get("PYTHONPATH", "")])

            full_args = [sys.executable, "-s", script, "--config", config_path]
            full_args.extend(extra_args)

            wanda_process = subprocess.Popen(args=full_args, cwd=base_import, env=env)  # nosec

        except Exception as exc:
            self.logger.error("protocol_selector.failed_launch", params={"filename": config_path, "exc": exc})

            for p in background_ps:
                p.terminate()

            raise

        # Interact with process: Send data to stdin. Read data from stdout and stderr, until
        # end-of-file is reached. Wait for process to terminate.
        output, error_output = wanda_process.communicate()

        if wanda_process.returncode != 0:
            if output:
                self.logger.info(f"Output from process: {output}")
            if error_output:
                self.logger.info(f"Error output from process: {error_output}")

        if background_ps:
            for background_p in background_ps:
                background_p.terminate()

        return wanda_process.returncode

    def _run_hybrid_config(self, config_path: str, base_import: str, extra_args: list[str]) -> int:
        """Meta-hybrid script is of the form:
        [meta.stages]
        run_stages_in_sequence = [ "stage_1", ... ]

        [meta.stages.stage_1]
        message_before = None / "Do something"
        config = "membrane_qc.toml"

        [meta.protocol]
        experiment_type = "blah"
        """

        # This nanodeep is assuming that the config has already been verified
        # Otherwise it wouldn't have appeared in the list
        contents = load_config(config_path)
        contents = filter_config_for_device(contents, device=self.device)
        stages = contents["meta"]["stages"]["run_stages_in_sequence"]

        # Propagate sawtooth_url & co until minknow handles config
        if "arguments" in contents["meta"]["protocol"]:
            args = contents["meta"]["protocol"]["arguments"]
            for (arg, val) in args.items():
                if isinstance(val, list):
                    extra_args += ["--" + arg] + [str(item) for item in val]
                else:
                    extra_args += ["--" + arg, str(val)]

        for stage_name in stages:
            stage_config = contents["meta"]["stages"][stage_name]
            config_path = os.path.join(base_import, stage_config["config"])

            if stage_config["message_before"]:
                # If there is a message, then synchrnoise up
                # here. If a stage doesn't have a
                # message_before then it shouldn't synchronize
                set_state(self.device, "PAUSED")
                wait_for_protocol_trigger(self.device, "resume")
                set_state(self.device, "RUNNING_SEQUENCING", clear_trigger=True)

            self.device.logger.log_to_gui("protocol_selector.stage.start", params=dict(stage_name=stage_name))
            return_code = self._run_config(config_path, base_import, extra_args)

            if return_code != 0:
                self.device.logger.log_to_gui("protocol_selector.stage.failed", params=dict(stage_name=stage_name))
                return return_code

        # All stages must have succeeded so return a bash success code
        return 0

    def get_protocol(self, identifier: str) -> dict:
        """Find protocol matching identifier. Fails if there isn't exactly one protocol associated
        with that identifier.

        Returns dict of name/tags/identifier

        :param identifier: String of identifier to match
        :rtype: dict

        """
        matching = [config for config in self._protocols if config["identifier"] == identifier]

        if len(matching) != 1:
            raise RuntimeError(f"Found {len(matching)} protocols with identifier {identifier}")

        return matching[0]

    def start(self, arguments: argparse.Namespace, extra_args: list[str]) -> int:
        """
        Launch a particular script

        :param arguments: a valid namespace with .identifier
        :param extra_args: other parameters to edit the bream configuration
        :return: process return code
        """
        protocol_info = self.get_protocol(arguments.identifier)

        if extra_args is None:
            extra_args = []

        if "kit" in protocol_info["tags"]:
            kit = protocol_info["tags"]["kit"]
            extra_args.extend(["--kit", kit])

            # If it's also a lamp kit make sure lamp_kit config is set
            if protocol_info["tags"].get("lamp kit"):
                barcoding_kits = [arg for arg in extra_args if arg.startswith("barcoding_kits")]
                if barcoding_kits and kit in barcoding_kits[0]:
                    extra_args.extend(["--lamp_kit", kit])

        if ".py" in protocol_info["tags"]["config path"]:
            # Makes this a recipe. Just run the python script
            full_args = [sys.executable, "-s", protocol_info["tags"]["config path"]]
            p = subprocess.Popen(args=full_args)  # nosec

            # Interact with process: Send data to stdin. Read data from stdout and stderr, until
            # end-of-file is reached. Wait for process to terminate.
            p.communicate()

            return p.returncode
        else:
            # If there were config errors, it would have been raised in the __init__
            config = load_config(protocol_info["tags"]["config path"])

            if "stages" in config["meta"]:
                return self._run_hybrid_config(
                    protocol_info["tags"]["config path"], protocol_info["tags"]["base import"], extra_args
                )

            return self._run_config(
                protocol_info["tags"]["config path"], protocol_info["tags"]["base import"], extra_args
            )

    def report(self, arguments: argparse.Namespace, extra_args: list[str]):
        """
        Launch report generation for a specific identifier.
        If no reports are specified in the config, this will do nothing.

        This is specified in the configs with the structure: ::

        [custom_reports.some_report]
        script = "/some/script/report.py"
        arguments = ["--test"] # Add extra args to run

        [[custom_reports.something_else]]
        script = "other/other_report.py"
        forward_arguments = ["--protocol-id"] # Forward arguments given to PS

        :param arguments: valid namesapce (arguments.idenfier for config ID)
        :param extra_args: Any other parameters to pass to the script
        """

        protocol_info = self.get_protocol(arguments.identifier)

        config = load_config(protocol_info["tags"]["config path"])
        config = filter_config_for_device(config, device=self.device)

        base_import = protocol_info["tags"]["base import"]

        self._run_python_sections(config.get("custom_reports", {}), base_import, extra_args, parallel=False)

    def _run_python_sections(
        self, python_sections: Union[list[dict], dict], base_import: str, extra_args: list[str], parallel: bool = False
    ) -> tuple[list[subprocess.Popen], list[str]]:
        """Launch all python processes specified in the list coming in.

        Sections is assumed to be of the form:
        {"section1": {"script": ..., ["enabled": true], arguments=[...args...]}},,,

        Any extra args passed will go through the following process:
        --some_name [...args...] a=True b=5 --some_name_1

        Will start some_name with --a=True --b=5
        And start some_name_1 [...args...] (If enabled was set to True)

        If parallel is False - Each one will be run in series waiting for each to be finished
           - If any return codes are not 0 the rest will not get run. RuntimeError raised.
        If parallel is True - All processes will be started and a handle returned for each

        :param python_sections: Dict of info for python processes
        :param base_import: List of imports
        :param extra_args: List of arguments
        :param parallel: Whether to run them in series or not

        :returns: ([process handles still running], [args not used])

        """
        processes_running = []

        # Transform legacy type into non legacy
        if isinstance(python_sections, list):
            new_sections = {}
            for section in python_sections:
                name = section.pop("name")
                new_sections[name] = section
            python_sections = new_sections

        for name, section in python_sections.items():
            p, extra_args = self._run_python_section(name, section, base_import, extra_args)

            if not parallel and p:
                stdout, stderr = p.communicate()  # Wait to finish

                if p.returncode != 0:
                    self.logger.error("protocol_selector.section.error", params={"name": name})
                    self.logger.info(stdout)
                    self.logger.info(stderr)
                    raise RuntimeError(f"Error running script under section {name}")
            elif p:
                processes_running.append(p)

        return processes_running, extra_args

    def _run_python_section(
        self, name: str, python_section: dict, base_import: str, extra_args: list[str]
    ) -> tuple[Optional[subprocess.Popen], list[str]]:
        """Run a given python section. Expected to be of the form:
        {"script": "a.py", ["enabled": true], arguments=[...args...]}

        If extra_args were: ["--test", "z=False", "a=5"] then:
        `a.py [...args...] --z False --a=5` would get run

        :param python_section: Dict of script info
        :param base_import: Where to try to import the script
        :param extra_args: Any args to use within launching
        :returns: (Process handle, [args not used])

        """
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join([base_import, env.get("PYTHONPATH", "")])

        script = python_section["script"]

        all_locations = [
            script,  # In case it's an absolute path
            os.path.join(base_import, script),
            os.path.join(CONFIG_DIR, script),
        ]

        valid_locations = [path for path in all_locations if os.path.exists(path)]
        if not valid_locations:
            self.logger.error("protocol_selector.script.not_found", params={"filename": name, "script": script})
            raise RuntimeError("Script not found")

        matching, non_matching = split_args(name, extra_args)

        # If enabled=True or we get extra arguments matching, run the process
        if matching is not None or python_section.get("enabled", True):
            full_args = [sys.executable, "-s", valid_locations[0]]
            full_args.extend(python_section.get("arguments", []))

            if matching:
                full_args.extend(matching)

            # Forward any arguments requested from the protocol selector
            ps_args_include = python_section.get("forward_arguments", [])
            # sys.argv: test.py report --a 1 --b --c 1 2 3 identifier
            sys_args = " ".join(sys.argv[2:-1]).split("--")
            # sys_args: a 1, b, c 1 2 3
            for sys_arg in sys_args:
                # Check --arg in requested
                if sys_arg and "--" + (sys_arg.split()[0].strip()) in ps_args_include:
                    full_args.extend(("--" + sys_arg).strip().split())

            p = subprocess.Popen(args=full_args, cwd=base_import, env=env)  # nosec
            return p, non_matching

        return None, non_matching


if __name__ == "__main__":
    ps = ProtocolSelector()

    parser = build_protocol_handler_argparser()

    args, unknown = parser.parse_known_args()
    exit(ps.run_command(args, unknown))
