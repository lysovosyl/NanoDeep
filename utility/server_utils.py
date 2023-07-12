from __future__ import annotations

import logging
import socket
import time
import uuid
from collections.abc import Iterable
from typing import Any, Optional

import bream4
import ont_certifi
import requests
from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.toolkit.pings.ping_dumping import ping_to_str
from requests.adapters import HTTPAdapter
from requests.exceptions import BaseHTTPError, RequestException  # type: ignore

logger = logging.getLogger(__name__)

RETRY_DELAY = [0.1, 0.2, 0.5, 1.5]
TIMEOUT = 5


def get_telemetry(url: str, success_codes: Optional[Iterable] = None) -> Optional[Any]:
    """
    Get json data from url

    :param url: url to get data from
    :param success_codes: What codes to class as success. If None, assumes 2xx is success
    :return: jsonified data or None if operation couldn't be completed
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Attempting get at {url}")

    adapter = HTTPAdapter()
    with requests.Session() as session:
        session.mount(url, adapter=adapter)

        for retry, delay in enumerate(RETRY_DELAY):
            try:
                response = session.get(url=url, verify=ont_certifi.where(), timeout=TIMEOUT)
                response_code = response.status_code
                logger.info(f"Attempt {retry} with return_code {response_code} and response {response.content}")

                if success_codes is None:
                    if response_code >= 200 and response_code < 300:
                        return response.json()
                elif response.status_code in success_codes:
                    return response.json()

                time.sleep(delay)

            except (RequestException, BaseHTTPError) as http_err:
                logger.info(f"Attempt {retry} raised {http_err}")

    return None


def post_telemetry(url: str, data: dict, success_codes: Optional[Iterable] = None) -> int:
    """
    Send data to url

    :param url: url to post data to
    :param data: the actual data
    :param success_codes: Codes to class as a success. If None, assumes 2xx is success
    :return: Status code
    """

    logger = logging.getLogger(__name__)
    logger.info(f"Attempting post at {url} with {data}")

    adapter = HTTPAdapter()

    response_code = None

    with requests.Session() as session:
        session.mount(url, adapter=adapter)

        for retry, delay in enumerate(RETRY_DELAY):
            try:
                response = session.post(
                    url=url,
                    data=ping_to_str(data),
                    verify=ont_certifi.where(),
                    timeout=TIMEOUT,
                    headers={"content-type": "application/json"},
                )
                response_code = response.status_code
                response_value = response.content
                logger.info(f"Attempt {retry} with return_code {response_code} and response {response_value}")

                if success_codes is None:
                    if response_code >= 200 and response_code < 300:
                        break
                elif response_code in success_codes:
                    break

                time.sleep(delay)

            except (RequestException, BaseHTTPError) as http_err:
                logger.info(f"Attempt {retry} raised {http_err}")
                response_code = -1

    return response_code  # type: ignore This looks like it will always be set


def get_common_telemetry(device: BaseDeviceInterface) -> dict:
    base = dict(
        msg_id=str(uuid.uuid4())[::-1],
        asic_id_eeprom=device.flow_cell_info.asic_id_str,
        flow_cell_id=device.get_flow_cell_id(),
        device_id=device.device_info.device_id,
        hostname=socket.gethostname(),
        protocols_version=bream4.__version__,
        version=device.connection.instance.get_version_info().minknow.full,
        exp_start_time=str(device.connection.acquisition.get_acquisition_info().start_time.ToDatetime()),
        # script information - also used with analysis linking information and session identification
        department=device.get_context_tags().get("department", ""),
        exp_script_purpose=device.get_exp_script_purpose(),
        # analysis linking information
        run_id=device.get_run_id(),
        protocol_run_id=device.connection.protocol.get_run_info().run_id,
    )
    if device.is_flongle:
        base["adapter_id"] = device.flow_cell_info.adapter_id

    return base
