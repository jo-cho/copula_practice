import statsmodels.api as sm
import numpy as np

__all__ = ['get_hl_idx']

def half_life(norm_spread):
        """
        Calculates time series half-life.
        https://en.wikipedia.org/wiki/Half-life
        :param norm_spread: An array like object used to calculate half-life.
        """
        lag = norm_spread.shift(1)
        lag[0] = lag[1]

        ret = norm_spread - lag
        lag = sm.add_constant(lag)

        model = sm.OLS(ret, lag)
        result = model.fit()
        half_life = -np.log(2)/result.params.iloc[1]

        return half_life
    
def get_hl_idx(X, threshold=50):

    cols = []

    for i in X.columns:

        hl = half_life(X[i])

        if hl <= threshold:
            cols.append(i)

    return cols

