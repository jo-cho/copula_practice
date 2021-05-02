import pandas as pd
import matplotlib.pyplot as mpl


def count_crosses(norm_spread, mean = 0.0):
    """
    Calculates the number of times a time series crosses its mean.
    :param norm_spread: An array like object used to calculate half-life.
    :param mean: A float to denote mean of norm_spread.
        Default value is 0.0.
    """

    curr_period = norm_spread
    next_period = norm_spread.shift(-1)
    count = (
        ((curr_period >= mean) & (next_period < mean)) |  # Over to under
        ((curr_period < mean) & (next_period >= mean)) |  # Under to over
        (curr_period == mean)
        ).sum()
    return count

def get_cr_idx(X, threshold=12):

    cols = []

    for i in X.columns:

        cc = count_crosses(X[i])

        if cc >= threshold:
            cols.append(i)

    return cols

