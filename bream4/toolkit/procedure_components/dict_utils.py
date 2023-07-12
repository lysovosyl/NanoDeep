from __future__ import annotations

from collections.abc import Generator
from functools import reduce
from typing import Any, Optional


def find_path(path: str, dictionary: dict) -> Optional[Any]:
    """Lookup a path e.g. 'a.b' in dict: {'a': {'b': 2}} will return 2.
    If the path doesn't exist, None is returned

    :param str path: Path to lookup in a hierarchical dictionary
    :param dict dictionary: dictionary to look up path in
    :returns: The item stored at that path in the dict
    :rtype: item
    :raises AttributeError: If the path is not there
    """
    try:
        if not path:
            return dictionary
        return reduce(lambda d, k: d.get(k), path.split("."), dictionary)  # type: ignore (AttributeError)
    except AttributeError:
        return None


def recursive_dict_iter(dictionary: dict, path: str = "") -> Generator[Any, None, None]:
    """Allows iteration over all key, value pairs recursively in dictionaries.
    It will yield (path, value) where path gives the path to that item.
    E.g. {a: 1, b: {c: 2, d: 3} would give (a, 1) then (b.c, 2) then (b.d, 3)

    :param dict dictionary: dictionary to recursively iterate over
    :param str path: prefix to add to keys
    :returns: An iterable to loop over all (key, value)
    :rtype: generator

    """
    for key, value in dictionary.items():
        new_path = key
        if path != "":
            new_path = path + "." + key

        if not isinstance(value, dict):
            yield (new_path, value)
        else:
            yield from recursive_dict_iter(value, path=new_path)


def recursive_merge(dict1: dict, dict2: dict) -> None:
    """Does an iterative version of dict1.update(dict2).

    :param dict dict1: dictionary that will be altered
    :param dict dict2: dictionary that items will be pulled from
    """
    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(dict1[key], dict):
            # If dictionary, recurse
            recursive_merge(dict1[key], value)
        else:
            dict1[key] = value
