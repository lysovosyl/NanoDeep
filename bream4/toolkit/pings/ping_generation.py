from __future__ import annotations

from typing import Optional, TypeVar

import pandas as pd


def generate_channelwise_ping(
    dataframe: pd.DataFrame, required_fields: list[str], optional_fields: Optional[list[str]] = None
):
    """
    Transform the dataframe to a generic channelwise ping.

    :param required_fields: The fields that will be used to construct the ping.
    Will give a warning if any of these fields are not present in the dataframe
    :param dataframe: A dataframe with metrics and/or assessment information
    :param optional_fields: A list of optional fields. If set to nan/None they are not copied
    :return: The channel-wise ping
    """

    if not optional_fields:
        optional_fields = []

    filtered = dataframe.reset_index().filter(required_fields + optional_fields)
    assert isinstance(filtered, pd.DataFrame)  # nosec B101 Needed to help type checker

    filtered_columns = filtered.columns.values
    if not set(required_fields).issubset(set(filtered_columns)):
        missing_columns = set(required_fields).difference(filtered_columns)
        raise Warning(
            "Missing: {}. "
            "Available fields {} do not contain the required field(s): {}".format(
                missing_columns, dataframe.columns.values, required_fields
            )
        )

    ping_dict = filtered.astype(object).to_dict("records")
    assert isinstance(ping_dict, list)  # nosec B101 Needed to help type checker

    # Don't want to ping None/NaN if they're optional
    for item in ping_dict:
        for optional_field in optional_fields:
            if pd.isnull(item[optional_field]):
                item.pop(optional_field)

    return ping_dict


def generate_summary_ping(dataframe: pd.DataFrame, field_to_summarise: str) -> dict[str, int]:
    """
    Transform the dataframe to a generic summary ping.

    :param field_to_summarise: The field to summarise
    :param dataframe: A dataframe with metrics and/or assessment information
    :return: The channel-wise ping
    """
    # We want to expose the id indices to allow for conversion
    ping_dict = dataframe[field_to_summarise].value_counts().astype(object).to_dict()
    return ping_dict


T = TypeVar("T", "list[dict]", dict)


def translate_keys(translatable: T, decode_dict: dict) -> T:
    """
    Exchanging the keys of a dictionary using another dictionary

    :param translatable: The data dictionary
    :param decode_dict: The translation dictionary
    :return: The translated dictionary
    """
    if isinstance(translatable, list):
        return [{decode_dict.get(k, k): v for k, v in item.items()} for item in translatable]
    elif isinstance(translatable, dict):
        return {decode_dict.get(k, k): v for k, v in translatable.items()}
    else:
        raise RuntimeError("Invalid type")
