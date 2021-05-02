import numpy as np
import pandas as pd
from pairs_selection import crossing, hurst, optics, PCA, cointegration, hl


def pca_reduce(data: pd.DataFrame, n_components, loading=True):
    data = PCA.normalize(data)
    if loading is False:
        pca_ret = PCA.pca_score(data.T, n_components)
    else:
        pca_ret = PCA.pca_loading(data, n_components)

    return pca_ret

# Cointegration Part

def coint(data: pd.DataFrame, clusters, p_value):
    """
       :param data: Time Series Data
       :param clusters: Clusteres
       :param p_value: p-value for adf test
       :return: Time seires data of selected cointegrated series
   """
    dt, c, p = data, clusters, p_value
    co = cointegration

    coint = co.cointegration(dt, c, p)
    return co.get_spread(dt, coint)


# Hurst Exponent Part
def Hurst(data: pd.DataFrame, threshold=0.5):
    """
   :param data: spreads of each pairs
   :param threshold: hurst exponents threshold
   :return: spreads of each pairs fillter by the hurst exponents
   """


    dt = data
    hurst_series = hurst.get_hurst_df(dt)
    fil_hurst = dt.loc[:,hurst_series.loc[hurst_series < threshold].index]
    return fil_hurst


# Halflife
def HalfLife(data: pd.DataFrame, threshold: int):
    """
    :param data: spreads of each pairs
    :param threshold: half-life threshold
    :return: spreads of each pairs fillter by the half-life
    """

    cols = hl.get_hl_idx(data, threshold)
    after_hl = data[cols]

    return after_hl

# Crossings Part


def Crossing(data: pd.DataFrame, thresholds: int):
    """
    :param data: timeseries of spreads
    :param thresholds: threshold for mean
    :return: filtered timeseires
    """
    cols = crossing.get_cr_idx(data, thresholds)
    after_cc = data[cols]

    return after_cc


def main(data, n_components: int, min_samples: int,
         max_eps=np.inf, cluster_method='xi', crossing_thresholds=12, halflife_threshold=252, pvalue_threshold=0.05, hurst_threshold=0.5):

    dt = data
    nc = n_components
    ms, me, cm = min_samples, max_eps, cluster_method
    co_t, h_t, hl_t, cr_t = pvalue_threshold, hurst_threshold, halflife_threshold, crossing_thresholds

    # PCA reducing
    reduced = pca_reduce(dt, nc)

    # Creating clusters
    clus = optics.optics(reduced, ms, me, cm)

    # Retrieving spreads
    spread = coint(dt, clus, co_t)

    # Hurst
    fil_hurst = Hurst(spread, h_t)

    # Halflife 부분 필요
    fil_hl = HalfLife(fil_hurst, hl_t)

    # Crossing
    fil_cross = Crossing(fil_hl, cr_t)

    filtered_pairs = fil_cross

    return filtered_pairs