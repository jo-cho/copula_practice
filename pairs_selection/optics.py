from sklearn.cluster import OPTICS
import pandas as pd
import numpy as np

__all__ = ['optics']

def optics(df, min_samples, max_eps=np.inf, cluster_method='xi'):
    """
    :param df: n x k 형태의 normalized dataframe. n: 기업
    :param min_samples: minPts
    :param max_eps: epsilon

    :return: list of lists. 내부의 리스트는 각 cluster 에 속한 기업들을 담고 있음.
    """
    list_of_list = []
    optic = OPTICS(min_samples=min_samples, max_eps=max_eps, cluster_method=cluster_method).fit(df)
    labels = optic.labels_
    no_clusters = len(np.unique(labels))
    df_labels = pd.DataFrame({'label': labels}, index=df.index)
    for i in range(0, no_clusters-1):
        clustered_companies = list(df_labels.loc[df_labels.label == i].index.values)
        list_of_list.append(clustered_companies)

    return list_of_list
