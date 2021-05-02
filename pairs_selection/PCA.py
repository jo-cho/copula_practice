import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

__all__ = ['normalize', 'pca_score', 'pca_loading']

def normalize(df):
    standardized_df = df.sub(df.mean(), axis=1).div(df.std(), axis=1)
    return standardized_df

def pca_score(df, n_components):
    """
    T x T correlation matrix를 사용하는 거라 이해
    input:
        - df: DataFrame of nxT normalized return series, where n is # of companies.
        - n_components: PCA n_components
    output:
        - n x k matrix
        - A*v
        - Sklearn의 fit_trasnform
    """
    pca = PCA(n_components=n_components)
    pca_ret = pca.fit_transform(df)
    pca_ret = pd.DataFrame(pca_ret, index=df.index)

    return pca_ret


def pca_loading(df, n_components):
    """
    n x n correlation matrix를 사용하고, loading matrix를 구하는 것
    input:
        - df: DataFrame of Txn normalized return series
        - n_components: PCA n_components
    output:
        - n x k matrix(where a_ij = Cov(company i's return series, jth eigen vector))
    """
    pca = PCA(n_components=n_components)
    pca.fit(df)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loading_matrix = pd.DataFrame(loadings, index=df.columns)
    return loading_matrix
