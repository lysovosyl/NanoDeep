from __future__ import annotations

import math

import numpy as np
import pandas as pd

from bream4.device_interfaces.devices.base_device import BaseDeviceInterface
from bream4.legacy.toolkit.analysis.metrics.utils.signal import sawtooth as sawtooth_wave
from bream4.legacy.toolkit.analysis.metrics.utils.signal import square as square_wave

MEAS_CAP = "meas_cap"
MEAS_RES = "meas_res"
PROB_SQ = "prob_sq"
PROB_TRI = "prob_tri"
TRIANGLE_RATIO = "triangle_ratio"
SD_CAP = "sd_cap"

DV_DT_MINION = 0.12 / 0.075
DV_DT_PROMETHION = 0.128 / 0.075


def filter_data(raw: np.ndarray) -> np.ndarray:
    """
    filters the raw data
    :param raw: raw data points
    :return:
    """
    rft = np.fft.rfft(raw)
    rft[10:] = 0
    y_smooth = np.fft.irfft(rft)
    return y_smooth


def get_data_delay(data: np.ndarray) -> float:
    """
    gets the delay in the data
    :param data: to be defined
    :return:
    """
    len_data = len(data)
    the_max = 0.0
    previous = data[len_data - 1]  # Preload last data point
    delay = 0
    for i in range(len_data):
        current = data[i]
        delta = current - previous
        if delta > the_max:
            the_max = delta
            delay = i
        previous = current
    return delay


def bin_raw_data(samples_per_wave: int, raw_data: np.ndarray) -> np.ndarray:
    """
    breaks up the raw data into manageable bins
    :param samples_per_wave: to be defined
    :param raw_data: to be defined
    :return:
    """

    if len(raw_data) < samples_per_wave:
        raise RuntimeError("Number of data points too small to make a meaningful calculation")
    np_data = np.array(raw_data)
    # get the bins, don't start at zero
    bins = np.arange(samples_per_wave, len(np_data), samples_per_wave)
    # split the data into lists of length samples per wave; ignore last one as
    # it's highly unlikely to be a whole wave
    all_waves = np.split(np_data, bins)[:-1]
    # get the average wave
    average_wave = np.mean(all_waves, 0, dtype=np.float32)
    return average_wave


def measure_cr(
    clip_begin: int, clip_end: int, v_ptp: int, t_period: float, t_sample: float, curr_raw: np.ndarray
) -> tuple[float, float, float, float]:
    """
    measured something
    :param clip_begin: to be defined
    :param clip_end: to be defined
    :param v_ptp: to be defined
    :param t_period: to be defined
    :param t_sample: to be defined
    :param curr_raw: to be defined
    :return:
    """

    # Derive the number of samples in a single wave
    samples_per_wave = int(math.floor(0.5 + t_period / t_sample))

    # Filter the raw data by superimposing the waves
    curr_binned = bin_raw_data(samples_per_wave, curr_raw)

    # Smooth the wave
    curr_filter = filter_data(curr_binned)

    # Get start of waveform
    delay = get_data_delay(curr_filter)
    # then shift data by delay
    curr_delay = np.roll(curr_filter, samples_per_wave - delay)
    # curr_delay = curr_filter[delay:samples_per_wave] + curr_filter[0:delay]

    length = np.linspace(0, len(curr_delay), len(curr_delay))
    curr_delay2 = curr_delay - (curr_delay[0])

    # Define the reference waves - Square wave has smooth-ish turns.
    square = (square_wave((2 * np.pi * 1 * length))) * (np.ptp(curr_delay2) / 2) + np.mean(
        curr_delay2, dtype=np.float32
    )
    square_max = np.max(square)
    square_min = np.min(square)
    square[0:30] = 0
    square[-30:] = 0
    square = savgol_filter(square, 129, 2)
    square[square > square_max] = square_max
    square[square < square_min] = square_min
    # Triangles are easy..
    triangle = (sawtooth_wave((2 * np.pi * 1 * length), width=0.5)) * (np.ptp(curr_delay2) / 2) + np.mean(
        curr_delay2, dtype=np.float32
    )
    # Now compare them at different phases
    best_fit_tri = []
    best_fit_squ = []
    for loop in range(0, (len(curr_delay2) + 1), int(len(curr_delay2) / 20)):
        square2 = np.roll(square, loop)
        triangle2 = np.roll(triangle, loop)
        square_d_obs = np.cumsum(abs(curr_delay2 - square2))[-1]
        triangle_d_obs = np.cumsum(abs(curr_delay2 - triangle2))[-1]
        best_fit_squ.append(square_d_obs)
        best_fit_tri.append(triangle_d_obs)

    triangle_d_obs = min(best_fit_tri)
    square_d_obs = min(best_fit_squ)

    # Fit straight lines to the filtered data
    step, slope = get_measurements(clip_begin, clip_end, curr_delay)

    # Calculate Capacitance from step and Resistance from slope
    meas_cap, meas_res = calculate_cr(step, slope, v_ptp, t_period, t_sample)
    return meas_cap, meas_res, square_d_obs, triangle_d_obs


# TODO MK-3729 Factor out this code
def savgol_filter(y: np.ndarray, window_size: int, order: int, deriv: int = 0) -> np.ndarray:
    """
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.

    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    This code has been taken from http://www.scipy.org/Cookbook/SavitzkyGolay

    :param array_like y: array_like, shape (N,)
        the values of the time history of the signal.
    :param int window_size:
        the length of the window. Must be an odd integer number.
    :param int order:
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    :param int deriv:
        the order of the derivative to compute (default = 0 means only smoothing)

    :return:
        `ys`, the smoothed signal (or it's n-th derivative).
    :rtype: ndarray, shape (N)

    **Notes**

    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    **Examples**

    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.savefig('images/golay.png')
    #plt.show()

    **References**

    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    try:
        window_size = np.abs(np.int(window_size))  # type: ignore
        order = np.abs(np.int(order))  # type: ignore
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.array([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])

    try:
        # Due to Win 10 v. 2004 update, this breaks certain calculations in numpy
        # A retry here is added as a work around until that is fixed.
        # See: https://github.com/numpy/numpy/issues/16744 To follow why
        m = np.linalg.pinv(b)[deriv]
    except np.linalg.LinAlgError:
        m = np.linalg.pinv(b)[deriv]

    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m, y, mode="valid")


def fit_line_to_data(begin: int, end: int, data: np.ndarray) -> tuple[float, float]:
    """
    finds a line of be  fit for the data passed into it
    :param begin: to be defined
    :param end: to be defined
    :param data: to be defined
    :return:
    """
    length = end - begin
    # Calculate average
    sums = 0.0
    for i in range(length):
        sums = sums + data[begin + i]
    the_average = sums / length
    # Calculate slope
    slope_sum = 0.0
    half_length = length / 2
    # Treat odd & even number of data points differently
    # But divisor is basically length^2 / 4
    if 2 * half_length == length:
        odd = False
        slope_div = half_length * half_length
    else:
        odd = True
        slope_div = half_length * (half_length + 1)
    # Run through the data accumulating the residuals (deviation from average)
    for i in range(length):
        residual = data[begin + i] - the_average
        if i < half_length:  # First half of data
            slope_sum -= residual
        elif odd and (i == half_length):  # Exact middle (only for odd number of data points)
            pass
        else:  # Second half of data
            slope_sum += residual
    slope = slope_sum / slope_div
    return the_average, slope


def get_measurements(clip_begin: int, clip_end: int, data: np.ndarray) -> tuple[float, float]:
    """
    Calculate indices for start & stop of high section & start & stop of low section
    :param clip_begin: to be defined
    :param clip_end: to be defined
    :param data: to be defined
    :return: to be defined
    """

    wave_length = len(data)
    hi_start = clip_begin
    hi_stop = int(wave_length / 2) - clip_end
    lo_start = hi_stop + clip_begin + clip_end
    lo_stop = wave_length - clip_end
    # Measure the straight lines that go through the flat top & flat bottom
    hi_average, hi_slope = fit_line_to_data(hi_start, hi_stop, data)
    lo_average, lo_slope = fit_line_to_data(lo_start, lo_stop, data)
    # Combine the measurements from the 2 lines by averaging
    step = (hi_average - lo_average) / 2.0
    slope = (hi_slope - lo_slope) / 2.0
    # Build up (x,y) coordinates for plotting
    return step, slope


def calculate_cr(step: float, slope: float, v_ptp: int, t_period: float, t_sample: float) -> tuple[float, float]:
    """
    calculates wave form capacitance and resistance
    :param step:  to be defined
    :param slope: to be defined
    :param v_ptp: to be defined
    :param t_period: to be defined
    :param t_sample: to be defined
    :return: the mean capacitance and mean resistance of a wave form
    """
    v_wave = 2 * v_ptp  # Total voltage change in 1 wave-period
    volt_per_second = (v_wave * 0.001) / t_period  # Ramp rate in volts per second
    volt_per_sample = v_wave * (t_sample / t_period)  # Ramp rate in volts per sample-period

    try:
        meas_cap = step / volt_per_second  # C = I / (dV/dt)
        if not np.isfinite(meas_cap):
            meas_cap = 0
    except:  # noqa: E722
        # TODO exception clause is too broad
        meas_cap = 0
    try:
        meas_res = volt_per_sample / slope  # R = (dV/dt) / (dI/dt)
        if not np.isfinite(meas_res):
            meas_res = 0
    except:  # noqa: E722
        # TODO exception clause is too broad
        meas_res = 0

    return meas_cap, meas_res


def waveform_triangle(
    sample_rate: int, triangle_waveform_data: np.ndarray, device: BaseDeviceInterface, ndigits: int = 2
) -> pd.Series:
    """
    :param sample_rate: The sample rate
    :param triangle_waveform_data: a numpy array or raw data points
    :param ndigits: approximate the results to this number of digits
    :type device: The device that is being used
    """
    if device.is_promethion:
        return _waveform_triangle_promethion(
            sample_rate=sample_rate,
            triangle_waveform_data=triangle_waveform_data,
            ndigits=ndigits,
        )
    elif device.is_minion_like:
        return _waveform_triangle_minion(
            sample_rate=sample_rate,
            triangle_waveform_data=triangle_waveform_data,
            ndigits=ndigits,
        )
    else:
        raise RuntimeError("Device type not valid")


def _waveform_triangle_minion(sample_rate: int, triangle_waveform_data: np.ndarray, ndigits: int = 2) -> pd.Series:
    """
    calculates the capacitance, resistance and the probability of th waveform being square or triangle
    :param sample_rate: The sample rate
    :param triangle_waveform_data: a numpy array or raw data points
    :param ndigits: approximate the results to this number of digits
    :type triangle_waveform_data: list|ndarray
    :return dict: with capacitance, resistance and the probability of th waveform being square or triangle
    """

    # Set parameters for any data
    t_period = 0.075  # Period of waveform (must be <= 0.075s) in seconds
    v_max = +60  # Maximum waveform generator voltage in milli-volts
    v_min = 0  # Minimum waveform generator voltage in milli-volts
    v_ptp = v_max - v_min

    # Set parameters for the measurement algorithm
    clip_begin = 40  # Ignore data points between a an edge and a flat section
    clip_end = 60  # Ignore data points between a flat section and an edge

    t_sample = 1.0 / sample_rate
    meas_cap, meas_res, prob_sq, prob_tri = measure_cr(
        clip_begin, clip_end, v_ptp, t_period, t_sample, triangle_waveform_data
    )
    # if len(triangle_waveform_data) <= 2000:
    #     raise RuntimeError("Not enough waveform data, Need > 2000 samples")
    sd_cap = np.std(triangle_waveform_data[1000:-1000], dtype=np.float32) / DV_DT_MINION

    if np.all(np.isnan(triangle_waveform_data)):
        meas_cap, meas_res, prob_sq, prob_tri, prob_sq, sd_cap = [np.nan] * 6

    return pd.Series(
        [
            round(meas_cap, ndigits),
            round(meas_res, ndigits),
            round(prob_sq, ndigits),
            round(prob_tri, ndigits),
            round(prob_sq / prob_tri, ndigits),
            round(sd_cap, ndigits),
        ],
        index=[MEAS_CAP, MEAS_RES, PROB_SQ, PROB_TRI, TRIANGLE_RATIO, SD_CAP],
    )


def _waveform_triangle_promethion(sample_rate: int, triangle_waveform_data: np.ndarray, ndigits: int = 2) -> pd.Series:
    """
    calculates the capacitance, resistance and the probability of th waveform
    being square or triangle
    :param sample_rate: The sample rate
    :param triangle_waveform_data: a numpy array or raw data points
    :param ndigits: approximate the results to this number of digits
    :type triangle_waveform_data: list|ndarray
    :return dict: with capacitance, resistance and the probability of th
    waveform being square or triangle
    """

    # Set parameters for any data
    t_period = 0.074  # Period of waveform (must be <= 0.075s) in seconds
    v_max = +64  # Maximum waveform generator voltage in milli-volts
    v_min = 0  # Minimum waveform generator voltage in milli-volts
    v_ptp = v_max - v_min
    t_sample = 1.0 / sample_rate

    # Derive the number of samples in a single wave
    samples_per_wave = int(math.floor(0.5 + t_period / t_sample))
    # Set parameters for the measurement algorithm
    clip_begin = int(math.floor(0.2 * (samples_per_wave / 2)))
    clip_end = int(math.floor(0.2 * (samples_per_wave / 2)))

    meas_cap, meas_res, prob_sq, prob_tri = measure_cr(
        clip_begin, clip_end, v_ptp, t_period, t_sample, triangle_waveform_data
    )
    # if len(triangle_waveform_data) <= 2000:
    #     raise RuntimeError("Not enough waveform data, Need > 2000 samples")
    sd_cap = np.std(triangle_waveform_data[1000:-1000], dtype=np.float32) / DV_DT_PROMETHION

    if np.all(np.isnan(triangle_waveform_data)):
        meas_cap, meas_res, prob_sq, prob_tri, prob_sq, sd_cap = [np.nan] * 6

    return pd.Series(
        [
            round(meas_cap, ndigits),
            round(meas_res, ndigits),
            round(prob_sq, ndigits),
            round(prob_tri, ndigits),
            round(prob_sq / prob_tri, ndigits),
            round(sd_cap, ndigits),
        ],
        index=[MEAS_CAP, MEAS_RES, PROB_SQ, PROB_TRI, TRIANGLE_RATIO, SD_CAP],
    )


#
# def capture_current_and_calculate_waveform_data(device, wave_time):
#     """
#     Captures raw data, time depending on sample frequency and calculated the waveform capacitance and resistance
#     from the raw data
#
#     :param float wave_time: time in seconds to run the wave form for
#     :return: a dict where keys are channels and values are a dict of the 4 results calculated for each channel
#     """
#     # todo needs refactoring to use the defacto methods
#     sample_rate = int(device.get_minknow_state('real_sample_rate'))
#     logger.info('waveform time is {}'.format(wave_time))
#     data = {}
#
#     raw_data_converter = RawDataConverter(
#         calibration_params_dict=device.engine.get_calibration(),
#         digitisation=digitisation)
#
#     raw_data_collection_time = min(wave_time, 10)
#
#     logger.info(
#         'capturing {} seconds of raw data'.format(raw_data_collection_time))
#     start = time.time()
#     required_samples = int(sample_rate * raw_data_collection_time)
#     missing_samples = required_samples
#
#     time.sleep(
#         1.5)  # make "sure" the waveform is applied when the raw data points are collected
#     # todo use the command hyperstream watcher and align the raw data collected with the waveform
#     with DataStream(engine=engine, dtype="raw",
#                     skip_to_end=True) as raw_data_ds:
#         while missing_samples > 0:
#             time.sleep(missing_samples / sample_rate)
#             raw_data_ds.read()
#             samples = required_samples
#             for channel in range(0, raw_data_ds.channels):
#                 samples = min(samples, raw_data_ds.data[channel].size)
#             missing_samples = required_samples - samples
#
#         logger.info(
#             'getting raw data took {} seconds'.format((time.time() - start)))
#
#         logger.info('calculating membrane measurements')
#         start = time.time()
#         raw_data = raw_data_ds.data
#         for channel in range(0, raw_data_ds.channels):
#             channel_raw_data = raw_data[channel][0:required_samples]["value"]
#             x0, x1 = raw_data_converter.get_calib_coeffs(channel)
#             channel_raw_data_pa = x0 + channel_raw_data * x1
#             result = waveform_triangle(channel_raw_data_pa)
#             result['start_raw_index'] = raw_data_ds.start_indices[channel]
#             data[str(channel + 1)] = result
#
#         logger.info('calculating membrane measurements took {} seconds'.format(
#             (time.time() - start)))
#         return data
