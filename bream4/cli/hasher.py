from __future__ import annotations

__doc__ = """
Use get_hash() to generate a hash of the .toml and .py that are part of a package
"""

import hashlib
import logging
from collections.abc import Iterable

from _hashlib import HASH
from pkg_resources import resource_isdir, resource_listdir, resource_stream

from bream4 import __package__ as bream4_package


def hash_package(package: str, py_sha: HASH, extensions: Iterable[str]) -> set[str]:
    """

    :param package: where to look for. This is usually  the __package__ variable of the package to inspect
    :param py_sha: the hashlib object to update the hash
    :param extensions: file extensions included in the hashing
    :return: Set of files found
    """

    files_found = set()

    bream_resources = resource_listdir(package, "")
    bream_resources.sort()  # hashing is not commutative

    for file_or_subdir in bream_resources:
        if resource_isdir(package, file_or_subdir):
            # it tries to access the destination folder before getting into it.
            # if it is not a python package, gives up on hashing the folder  and continues to the next file or dir
            try:
                resource_listdir("{}.{}".format(package, file_or_subdir), "")
            except (ImportError, TypeError):
                logging.debug("{} is not a package".format(file_or_subdir))
                continue
            else:
                files_found.update(
                    hash_package(
                        package="{}.{}".format(package, file_or_subdir),
                        py_sha=py_sha,
                        extensions=extensions,
                    )
                )

        elif any((file_or_subdir.endswith(ex) for ex in extensions)):
            files_found.add(file_or_subdir)
            with resource_stream(package, file_or_subdir) as resource_stream_:
                data = resource_stream_.read()
                data = data.replace(b"\r\n", b"\n")  # so we get the same results in windows
                py_sha.update(data)

    return files_found


def get_hash() -> HASH:
    py_sha = hashlib.sha256()
    hash_package(package=bream4_package, py_sha=py_sha, extensions=(".py", ".toml"))
    return py_sha.hexdigest()


def execute():
    print(get_hash())


if __name__ == "__main__":
    exit(execute())
