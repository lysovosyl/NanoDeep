from __future__ import annotations

"""
----------------------------------------------------------------------------------------------------
                Utility for parsing configuration files and merging them together.
----------------------------------------------------------------------------------------------------

Simply call load_config('filename.toml') which will return the parsed/loaded config file or...
...This can raise the ConfigParseException.

#. Could not find config file...
#. Could not parse config file...
#. Expecting to see all of REQUIRED_FIELDS
#. Type mismatch (base device x has type int but you tried to override with type string)
#. Imported files have conflicting attributes...
#. Multi level imports not supported:...
#. Bream/MinKNOW version incompatibility

"""

import os
from typing import Optional

import toml

# Check compatibility
from packaging import version
from typing_extensions import TypeAlias
from utility.config_file_utils import create_path, find_path, recursive_dict_iter, recursive_merge

REQUIRED_FIELDS = {
    "post_processing_protocol": ["script", "compatibility.minknow_core"],
    "protocol": [
        "device",
        "script",
        "compatibility.minknow_core",
        "meta.protocol.experiment_type",
        "meta.exp_script_purpose",
    ],
}
CONFIG_DIR = os.environ.get("ONT_CONFIG_DIR", "configuration")
USER_CONFIG_DIR = os.environ.get("ONT_USER_DIR", "")


class ConfigParseException(Exception):
    pass


Imports: TypeAlias = "list[tuple[str, str]]"


def load_config(filename: str, base: Optional[str] = None) -> dict:
    """Loads a config file, merging any imports listed. This can raise a ConfigParseExcetion
    if files not found/not parseable/type mismatch/missing required fields/mutli-level imports

    :param string filename: Path of toml config file to load
    :param string base: Base path of toml, used to resolve imports

    :returns: Dict of attributes->Values loaded from the active config files
    :rtype: dict
    :raises ConfigParseException: if the config cannot be loaded/parsed. See module description
    """
    contents = load_raw_toml(filename)

    if not base:
        # If base isn't specified, try to figure it out
        if filename.startswith(CONFIG_DIR):
            base = CONFIG_DIR
        else:
            base = USER_CONFIG_DIR

    # If the file is a special hybrid, make sure it's valid
    if find_path("meta.stages", contents):
        return verify_hybrid_config(filename, base=base)

    # Lookup any imports. Then do the imports. This prevents triggering imports from an imported
    # file
    imports = find_imports(contents, delete_imports=True)
    base_config = do_imports(imports, base=base)

    # If there are any device independent settings in the contents, remove
    # any matching ones from the base. This is so that if you open
    # a config that imports other things it should be readable
    remove_inherited_dependent(contents, base_config)

    # Then extend the base config with ours (Not the other way round as we'd lose top level updates)
    recursive_merge(base_config, contents)

    # If there is still 'import' as a key, that means we have multi imports which we should fail on,
    # as this will not take effect and the user needs to be aware
    remaining = [value for (key, value) in recursive_dict_iter(base_config) if "import" in key]
    if remaining:
        raise ConfigParseException("Multi-level imports not supported: %s" % (remaining,))

    # Check all required fields are there - Look up 'type' in the required fields
    required_fields = REQUIRED_FIELDS.get(base_config.get("type", "protocol"), [])
    if any([find_path(x, base_config) is None for x in required_fields]):
        raise ConfigParseException("Expecting to see all of the following sections: %s" % (required_fields,))

    check_compatibility(base_config)

    return base_config


def check_compatibility(config: dict) -> None:
    """Checks compatibility fields in a config dict (bream/minknow versions).
    Will raise a ConfigParseException if there is a version incompatibility.

    The actual version will be shortened to the same major/minor/patch as the config
    e.g.
    If the minknow version in the config is 3.6
    and minknows reported version is 3.6.0+12
    3.6 and 3.6 will be compared

    :param config: Dict of config

    """

    if "compatibility" in config:
        if "minknow_core" in config["compatibility"]:
            import minknow

            actual_version = minknow.__version__
            compare_version = str(config["compatibility"]["minknow_core"])

            if "dev" not in actual_version:
                # Only compare as much as specified (major/minor/patch~rc etc)
                dot_len = compare_version.count(".")
                actual_version = ".".join(actual_version.split(".")[: dot_len + 1])
                if version.parse(compare_version) > version.parse(actual_version):
                    raise ConfigParseException("MinKNOW version incompatibility")

        if "bream" in config["compatibility"]:
            import bream4

            actual_version = bream4.__version__
            compare_version = str(config["compatibility"]["bream"])

            if not bream4.__version__.startswith("0.0."):
                # Only compare as much as specified (major/minor/patch~rc etc)
                dot_len = compare_version.count(".")
                actual_version = ".".join(actual_version.split(".")[: dot_len + 1])

                if version.parse(compare_version) > version.parse(actual_version):
                    raise ConfigParseException("Bream version incompatibility")

                # Compare the major version of the config with bream. Invalidate if different
                # due to breaking changes in major versions.
                if int(compare_version.split(".")[0]) != int(bream4.__version__.split(".")[0]):
                    raise ConfigParseException("Bream major version incompatibility")


def verify_hybrid_config(filename: str, base: Optional[str] = None) -> dict:
    """Verifies a hybrid-config.
    This should be of the form: ::

      [meta.stages]
      run_stages_in_sequence = [ "stage_1", ... ]

      [meta.stages.stage_1]
      message_before = "" / "Do something"
      config = "membrane_qc.toml"

      [meta.protocol]
      experiment_type = "blah"

    where all the configs are valid. This will return a dict of the hybrid toml

    :param filename: Filename of config to verify
    :param base: Base of file path to resolve imports
    :returns: contents of config
    :rtype: dict
    :raises ConfigParseException: if the hybrid-config cannot be loaded/parsed.
    """

    contents = load_raw_toml(filename)

    if not base:
        base = CONFIG_DIR

    if not find_path("meta.stages", contents):
        raise ConfigParseException("Hybrid-Config invalid. 'meta' and 'meta.stages' required")

    if not set(contents.keys()).issubset(["compatibility", "meta", "type"]):
        raise ConfigParseException("Hybrid-Config invalid. Only meta & compatibility " + "sections are allowed")

    # Grab the flowcells/kits
    flowcells = set()
    kits = set()
    if "compatibility" in contents:
        flowcells = set(contents["compatibility"].get("flowcells", []))
        kits = set(contents["compatibility"].get("kits", []))

    if "run_stages_in_sequence" not in contents["meta"]["stages"]:
        raise ConfigParseException("Hybrid-Config invalid. No run_stages_in_sequence specified")

    stages = contents["meta"]["stages"]["run_stages_in_sequence"]
    if not set(stages).issubset(contents["meta"]["stages"].keys()):
        raise ConfigParseException("Hybrid-Config invalid. All stages in " + "run_stages_in_sequence not specified")

    for stage in stages:
        stage_config = contents["meta"]["stages"][stage]

        # Check that it has a message_before/config and that the config can be loaded
        if "message_before" not in stage_config.keys():
            raise ConfigParseException(
                "Hybrid-Config invalid. " + "No message_before specified in stage {}".format(stage)
            )

        if "config" not in stage_config.keys():
            raise ConfigParseException("Hybrid-Config invalid. " + "No config specified in stage {}".format(stage))

        stage_config_path = os.path.join(base, stage_config["config"])
        # Check recurison
        if stage_config_path == filename:
            raise ConfigParseException(
                "Hybrid-Config invalid. " + "Recursive stages detected in stage {}".format(stage)
            )
        # Load_config will raise exception if can't be parsed. Also try the base directory
        # Specify base as dont want system workflows pulling stages from user space
        try:
            stage_loaded_config = load_config(stage_config_path, base=base)
        except ConfigParseException:
            stage_loaded_config = load_config(stage_config_path, base=CONFIG_DIR)

        if find_path("meta.stages", stage_loaded_config):
            raise ConfigParseException(
                "Hybrid-Config invalid. " + "Recursive stages detected in stage {}".format(stage)
            )

        # Check flowcell/kit compatibilities
        stage_flowcells = find_path("compatibility.flowcells", stage_loaded_config)
        if stage_flowcells:
            if not flowcells.issubset(stage_flowcells) or not flowcells:
                raise ConfigParseException(
                    "Hybrid-Config invalid. Flowcell incompatibility" + " in stage {}".format(stage)
                )

        stage_kits = find_path("compatibility.kits", stage_loaded_config)
        if stage_kits:
            if not kits.issubset(stage_kits) or not kits:
                raise ConfigParseException("Hybrid-Config invalid. Kit incompatibility" + " in stage {}".format(stage))

    return contents


def remove_inherited_dependent(top: dict, imports: dict) -> None:
    """If there are any device independent settings in the top, either
    from device of custom_settings then remove any of those from
    device independent settings in imports

    :param dict top: config dict of main file
    :param list of dict imports: configs from imported files

    """
    _remove_inherited_dependent(top, imports, attr="device")
    _remove_inherited_dependent(top, imports, attr="custom_settings")


def _remove_inherited_dependent(top: dict, imports: dict, attr: str = "device") -> None:
    if attr not in top:
        return

    top_level_device_independent_settings = []

    for (attribute, value) in top[attr].items():
        if not isinstance(value, dict):
            # Found an attribute of the device
            top_level_device_independent_settings.append(attribute)

    # Now remove these from devices in imports
    for import_config in imports:
        if attr not in import_config:
            continue

        for (attribute, value) in imports[attr].items():
            if isinstance(value, dict):
                # Found a device specific section
                # Remove any device independent settings
                for setting in top_level_device_independent_settings:
                    value.pop(setting, None)


def find_imports(config: dict, delete_imports: bool = True, current_path: str = "") -> Imports:
    """Find a list of imports contained in the given config. Optionally also delete them

    :param dict config: dict of config settings
    :param bool delete_imports: whether to also delete the imports from the config dict
    :param string current_path: Current location we are looking
    :returns: list of (filepaths to import, respective section)
    :rtype: list of (strings, strings)

    """

    imports = []

    for key, value in config.items():
        if key == "import":
            imports.append((config["import"], current_path))

        if isinstance(value, dict):
            if not current_path:
                new_path = key
            else:
                new_path = "{}.{}".format(current_path, key)
            imports.extend(find_imports(value, delete_imports=delete_imports, current_path=new_path))

    # Delete outside loop as bad practice to delete whilst iterating through a dict
    if delete_imports and "import" in config:
        del config["import"]

    return imports


def do_imports(imports: Imports, base: Optional[str] = None) -> dict:
    """Import list of import files and merge them together in a dictionary. If the imports conflict
    in their settings i.e. 2 different files try to set device.frequency then a ConfigParseException
    will be raised.

    :param list imports: List of (toml config filename, path) to load and merge
    :param str base: Path to add to beginning of all the imports
    :returns: dictionary of attributes from merged config files
    :rtype: dict
    :raises ConfigParseException: If files try and set the same attribute

    """

    merged = {}
    for (filename, import_section) in imports:
        if base:
            filepath = os.path.join(base, filename)
            # Special case: Import from the system directory if can't be found
            if not os.path.exists(filepath):
                filepath = os.path.join(CONFIG_DIR, filename)
        else:
            filepath = filename

        # This would be a top level import
        if not import_section:
            contents = load_raw_toml(filepath)
        else:
            # Sectional import

            # Create a contents with a path to the insert
            contents = create_path(import_section, {})

            # Find where to insert the section
            insert_point = ".".join(import_section.split(".")[:-1])
            insert_key = import_section.split(".")[-1]

            # Load the specific section requested
            section = find_path(import_section, load_raw_toml(filepath))
            # Insert in the correct place
            find_path(insert_point, contents)[insert_key] = section  # type: ignore - path made with create_path

        # These imports should never affect one and other. i.e. they should not define anything that
        # the other one does
        keys_merged = {k for (k, v) in recursive_dict_iter(merged)}
        keys_file = {k for (k, v) in recursive_dict_iter(contents)}
        if keys_merged.intersection(keys_file) != set():
            raise ConfigParseException(
                "Imported files have conflicting attributes %s" % keys_merged.intersection(keys_file)
            )
        recursive_merge(merged, contents)
    return merged


def load_raw_toml(filename: str) -> dict:
    """Load a toml config file. ConfigParseException can be raised from this nanodeep if
       the file can't be found or if toml cannot parse the file

    :param str filename: toml config file to load, without resolving imports
    :returns: dictionary of attributes from config file
    :rtype: dict
    :raises ConfigParseException: If file not found, or can't be parsed

    """
    try:
        with open(filename, "r", encoding="utf-8") as toml_file:
            return toml.load(toml_file)
    except IOError:
        raise ConfigParseException("Could not find config file: %s" % (filename,))
    except (toml.TomlDecodeError, IndexError) as exc:
        raise ConfigParseException("Could not parse config file: %s (%s)" % (filename, str(exc)))
