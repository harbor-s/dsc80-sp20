from glob import glob
import pandas as pd
import numpy as np

from scipy.stats import linregress


def rmse(datasets):
    """
    Return the RMSE of each of the datasets.

    >>> datasets = {k:pd.read_csv('data/dataset_%d.csv' % k) for k in range(7)}    
    >>> out = rmse(datasets)
    >>> len(out) == 7
    True
    >>> isinstance(out, pd.Series)
    True
    """

    rmse = []
    for k,v in datasets.items():
        reg = linregress(v.X,v.Y)
        pred = []
        
        for i in range(v.shape[0]):
            pred.append((v.loc[i].X*reg.slope)+reg.intercept)
        pred = pd.Series(pred)
        
        rmse.append(np.sqrt(np.mean((pred-v.Y)**2)))
        
    return pd.Series(rmse)


def heteroskedasticity(datasets):
    """
    Return a boolean series giving whether a dataset is
    likely heteroskedastic.

    >>> datasets = {k:pd.read_csv('data/dataset_%d.csv' % k) for k in range(7)}    
    >>> out = heteroskedasticity(datasets)
    >>> len(out) == 7
    True
    >>> isinstance(out, pd.Series)
    True
    """

    return [False if linregress(v.X,v.Y).pvalue > 0.05 else True for k,v in datasets.items()]
