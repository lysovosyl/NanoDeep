import os

import h5py

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface


def setup_playback(device: BaseDeviceInterface, filename: str) -> None:
    """Configure bream/minknow to playback a raw fast5 file
    with extracting calibration data

    :param device: MinKNOW device wrapper
    :param filename: Path of fast5 file

    """
    filename = os.path.join(os.getcwd(), filename)

    try:
        with h5py.File(filename, "r") as f:
            # Grab calibrated data for each channel
            # Stored as Channel_1 etc
            calibration = {
                int(k.split("_")[1]): (
                    f["IntermediateData/" + k + "/Meta"].attrs["offset"],
                    f["IntermediateData/" + k + "/Meta"].attrs["range"],
                )
                for k in f["IntermediateData"]
            }

    except:  # noqa: E722
        device.logger.info("Failed to retrieve calibration data from {}".format(filename))
        calibration = {channel: (0, 2048) for channel in device.get_channel_list()}

    offsets = [calibration[key][0] for key in sorted(calibration.keys())]
    ranges = [calibration[key][1] for key in sorted(calibration.keys())]

    hdf5_mode = device.connection.acquisition._pb.SetSignalReaderRequest.HDF5
    loop_mode = device.connection.acquisition._pb.SetSignalReaderRequest.LOOP

    # Set the fast5 on loop
    device.connection.acquisition.set_signal_reader(
        reader=hdf5_mode,
        hdf_mode=loop_mode,
        hdf_source=filename,
        sample_rate_scale_factor=1.0,
    )

    # And set the calibration from the file
    device.connection.device.set_calibration(
        first_channel=min(calibration.keys()),
        last_channel=max(calibration.keys()),
        offsets=offsets,
        pa_ranges=ranges,
    )

    # Assume all channels calibrated successfully for playback
    device.set_channel_calibration_status([True] * device.channel_count)
