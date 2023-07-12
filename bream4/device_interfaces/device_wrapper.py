from typing import Optional, Union

from minknow_api import Connection

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface, get_env_grpc_port
from bream4.device_interfaces.devices.minion import MinionGrpcClient
from bream4.device_interfaces.devices.promethion import PromethionGrpcClient


def create_grpc_client(
    grpc_port: Optional[int] = None, reset_device: bool = False
) -> Union[MinionGrpcClient, PromethionGrpcClient]:
    """
    Create a concrete gRPC client able to talk to MinKNOW. This depends on the
    device MinKNOW is configured to work with.

    :param grpc_port: if the gRPC port is known it can be passed in
    :param reset_device: Whether to reset the device to sensible defaults
    :return: An instance of the grpc client able to communicate with MinKNOW
    """
    if grpc_port is None:
        grpc_port = get_env_grpc_port()


    try:
        connection = Connection(port=grpc_port, use_tls=True)
    except TypeError:
        # TODO: On major version bump, just use the except case
        connection = Connection(port=grpc_port)

    device_info = connection.device.get_device_info()

    grpc_client: Optional[BaseDeviceInterface] = None
    if device_info.device_type in [
        device_info.MINION,
        device_info.MINION_MK1C,
        device_info.GRIDION,
    ]:
        grpc_client = MinionGrpcClient(connection=connection)
    elif device_info.device_type == device_info.PROMETHION:
        grpc_client = PromethionGrpcClient(connection=connection)
    else:
        raise RuntimeError(f"Device type not recognised: {device_info.device_type}")

    if reset_device:
        grpc_client.reset()
    return grpc_client
