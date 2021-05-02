import statsmodels.api as sm
import statsmodels.tsa.stattools as ts
from itertools import combinations
import pandas as pd
import numpy as np

__all__ = ['cointegration', 'get_spread']

def cointegration(data, clustered, p_value=0.05):
    """

    :param data: 주가 데이터(normalize를 해야할까 말아야할까)
    :param clustered: list of lists 안 쪽 리스트는 각 cluster에 속한 기업들
    :param p_value: cointegration test를 할 때, error term의 adf test의 p-value. 얘보다 작으면 H0: error-term이 unit-root이다. 를 기각.
    :return: list of lists
            각 list 에는 4개의 값이 포함되어 있다.
            1. 종속변수(기업명)
            2. 독립변수(기업명)
            3. 계수
            4. p-value
    """

    passed = []
    for group in clustered:
        for y, x in combinations(group, 2):
            x_series = data.loc[:, x]
            y_series = data.loc[:, y]

            # y on x
            ols_yx = sm.OLS(y_series, x_series).fit()
            ols_yx_coef = ols_yx.params[0]
            ad_yx = ts.adfuller(ols_yx.resid)

            # x on y
            ols_xy = sm.OLS(x_series, y_series).fit()
            ols_xy_coef = ols_xy.params[0]
            ad_xy = ts.adfuller(ols_xy.resid)

            if ad_yx[0] <= ad_xy[0]:
                if ad_yx[1] < 0.05:
                    passed.append([y, x, ols_yx_coef, ad_yx[1]])
            else:
                if ad_xy[1] < 0.05:
                    passed.append([x, y, ols_xy_coef, ad_xy[1]])
    return passed


def get_spread(data,cointeg_list):
    """

    :param data: 주가 데이터
    :param cointeg_list: 위의 cointegration 함수의 output. list of list
    :return: DataFrame
        - row는 날짜
        - 각 column명:종속변수&독립변수 순으로 기업이름이 나옴.
        - 각 column 값은 해당 spread-series.
    """

    df = pd.DataFrame(index=data.index)
    for coint in cointeg_list:
        name = coint[0]+'&'+coint[1]
        b = coint[2]
        df[name] = data.loc[:, coint[0]] - b * data.loc[:, coint[1]]

    return df

