from __future__ import annotations

import hashlib
import os
from typing import Optional


def hash_recursive(folder: str, extensions: list[str]) -> str:
    """Recursively walk a folder grabbing any files that have an
    extension in extensions.

    Hash all the contents together with the paths and return an overall hash

    :param folder: Path to start
    :param extensions: List of extensions

    :returns: Hash of file contents
    :rtype: string

    """
    sha = hashlib.sha256()

    for (root, directory, files) in os.walk(folder):
        for filename in files:
            name, extension = os.path.splitext(filename)
            if extension in extensions:
                relative_path = os.path.join(root, filename)
                with open(relative_path, "r") as f:
                    data = f.read()
                    # so we get the same results in windows
                    data = data.replace("\r\n", "\n")
                    # Also add the filepath to ensure nothing is moved
                    sha.update("{}:{}".format(relative_path, data).encode("utf-8"))
    return sha.hexdigest()


def get_hash_configs(folder: Optional[str] = None) -> str:
    """Get a hash of all tomls in a given folder, or the default config directory

    :param folder: Path to start
    :returns: sha of contents
    :rtype: string

    """

    if not folder:
        folder = str(os.environ.get("ONT_CONFIG_DIR", "configuration"))
    return hash_recursive(folder, [".toml"])


def get_hash_scripts(folder: Optional[str] = None) -> str:
    """Get a hash of all pys in a given folder, or the default config directory

    :param folder: Path to start
    :returns: sha of contents
    :rtype: string

    """

    if not folder:
        folder = str(os.environ.get("ONT_CONFIG_DIR", "configuration"))
    return hash_recursive(folder, [".py"])


if __name__ == "__main__":
    print("Config Hash: {}".format(get_hash_configs()))
    print("Script Hash: {}".format(get_hash_scripts()))
