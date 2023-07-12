from collections import namedtuple

import numpy as np
import pandas as pd

ALL_PBS = range(1, 13)
Q1, Q3 = 0.25, 0.75

PB = "pb"
MUX = "well"

PIXEL_BLOCK_SIZE = 250


def add_pb_columns(data: pd.DataFrame) -> pd.DataFrame:

    if "channel" not in data.columns:
        if len(data.index) and "channel" in data.index.names:
            reset = data.reset_index()
            channels = reset.channel
        else:
            raise KeyError("Channel column not in supplied dataframe")
    else:
        channels = data.channel

    data["pb"] = (((channels - 1) / PIXEL_BLOCK_SIZE) + 1).astype(int).values
    return data


KsTestResult = namedtuple("KsTestResult", ("statistic", "pvalue"))


def ks_test(sample1: np.ndarray, sample2: np.ndarray) -> KsTestResult:
    """
    Two-sample Kolmogorov-Smirnov test implemented without SciPy to avoid bundling SciPy install
    https://en.wikipedia.org/wiki/Kolmogorov-Smirnov_test
    Tested equivalent to SciPy nanodeep
    :param sample1: 1-d array
    :param sample2: 1-d array
    :return: The K-S statistic and significance value
    """

    _eps1 = 0.001
    _eps2 = 1.0e-8

    def _ks_significance(x, n_iters=100):

        fac = 2.0
        s = 0.0
        term_bf = 0.0

        a2 = -2.0 * x ** 2
        for i in range(1, n_iters):
            term = fac * np.exp(a2 * i ** 2)
            s = s + term
            if abs(term) <= _eps1 * term_bf or abs(term) <= _eps2 * s:
                return s
            fac = -fac
            term_bf = abs(term)

        # Fails to converge:
        return 1.0

    data1 = np.sort(sample1)
    data2 = np.sort(sample2)

    n1 = data1.shape[0]
    n2 = data2.shape[0]
    data_all = np.concatenate([data1, data2])

    cdf1 = np.searchsorted(data1, data_all, side="right") / (1.0 * n1)
    cdf2 = np.searchsorted(data2, data_all, side="right") / (1.0 * n2)

    statistic = np.max(np.absolute(cdf1 - cdf2))

    en = np.sqrt(n1 * n2 / (n1 + n2))

    significance = _ks_significance(statistic * (en + 0.12 + (0.11 / en)), n_iters=100)

    return KsTestResult(statistic, significance)


def find_outlier_pixel_blocks(data: pd.DataFrame, metric_to_use: str) -> pd.DataFrame:
    """
    For each (pixel block, mux) subset, runs a Kolmogorov-Smirnov test on that subset and the overall dataset
    If the correlation coefficient between that subset and the whole dataset is below the 'outlier' threshold for
    at least one mux subset, flag that pixel block as being an outlier
    Note: This method is flawed when using wet chip metrics, as pixel blocks 1 and 12 are likely to look different, but
    works well on dry wiggle amplitude data
    :param data: DataFrame, must contain channel, mux and the supplied metric to analyse
    :param metric_to_use: column to use as sample values
    :return: DataFrame, pixel block and outlier flag
    """

    results = []

    whole_dataset = data[metric_to_use].to_numpy()

    for (pb, mux), pb_data in data.groupby([PB, MUX]):
        this_dataset = pb_data[metric_to_use].fillna(0).values

        correlation = ks_test(whole_dataset, this_dataset)
        results.append(
            {
                "pb": pb,
                "mux": mux,
                "ks_stat": correlation.statistic,
                "ks_pvalue": correlation.pvalue,
                "val_med": np.median(this_dataset[~np.isnan(this_dataset)]),
                "val_mean": np.mean(this_dataset[~np.isnan(this_dataset)]),
            }
        )

    results = pd.DataFrame(results)

    upper = results.ks_stat.quantile(Q3)
    iqr = upper - results.ks_stat.quantile(Q1)
    threshold = upper + (iqr * 3)
    outliers = results[results.ks_stat > threshold].pb.unique().tolist()

    pb_results = map(lambda pixel_block: (pixel_block, pixel_block in outliers), ALL_PBS)

    return pd.DataFrame(pb_results, columns=[PB, "outlier"])
