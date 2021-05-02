import numpy as np
import pandas as pd

# https://en.wikipedia.org/wiki/Hurst_exponent
__all__ = ['hurst', 'get_hurst_df']


def hurst(norm_spread):
    """
    Calculates Hurst exponent.
    https://en.wikipedia.org/wiki/Hurst_exponent
    :param norm_spread: An array like object used to calculate half-life.
    """
    # Create the range of lag values
    lags = range(2, 100)

    # Calculate the array of the variances of the lagged differences
    diffs = [np.subtract(norm_spread[l:], norm_spread[:-l]) for l in lags]
    tau = [np.sqrt(np.std(diff)) for diff in diffs]

    # Use a linear fit to estimate the Hurst Exponent
    poly = np.polyfit(np.log(lags), np.log(tau), 1)

    # Return the Hurst exponent from the polyfit output
    H = poly[0] * 2.0

    return H

def get_hurst_df(df):
    """
    :param df: dataframe, 후보들의 spread가 있어야 함.
    :return: hurst of each spread
    """
    hurst_df = pd.DataFrame(index=['hurst'])
    for i in df.columns:
        h = hurst(df[i].values)
        hurst_df[i] = h
    return hurst_df.T.hurst