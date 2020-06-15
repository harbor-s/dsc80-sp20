import pandas as pd
import numpy as np
import seaborn as sns
import os


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def simple_pipeline(data):
    '''
    simple_pipeline takes in a dataframe like data and returns a tuple 
    consisting of the pipeline and the predictions your model makes 
    on data (as trained on data).

    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = simple_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], FunctionTransformer)
    True
    >>> preds.shape[0] == data.shape[0]
    True
    '''
    
    pl = Pipeline([
        ('ft', FunctionTransformer(np.log)),
        ('lr', LinearRegression())
    ])

    pl.fit(data[['c2']], data[['y']])
    predictions = [x[0] for x in pl.predict(data[['c2']])]
    
    return (pl, np.array(predictions))

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def multi_type_pipeline(data):
    '''
    multi_type_pipeline that takes in a dataframe like data and 
    returns a tuple consisting of the pipeline and the predictions 
    your model makes on data (as trained on data).

    :Example:
    >>> fp = os.path.join('data', 'toy.csv')
    >>> data = pd.read_csv(fp)
    >>> pl, preds = multi_type_pipeline(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> isinstance(pl.steps[-1][1], LinearRegression)
    True
    >>> isinstance(pl.steps[0][1], ColumnTransformer)
    True
    >>> data.shape[0] == preds.shape[0]
    True
    '''

    ct = ColumnTransformer([
        ('asis', FunctionTransformer(lambda x: x), ['c1']),
        ('log', FunctionTransformer(np.log), ['c2']),
        ('1hot', OneHotEncoder(sparse=False), ['group'])
    ])
    pl = Pipeline([
        ('colt', ct),
        ('lr', LinearRegression())
    ])

    pl.fit(data[['c1','c2','group']], data[['y']])
    predictions = [x[0] for x in pl.predict(data[['c1','c2','group']])]
    return (pl, np.array(predictions))

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin


class StdScalerByGroup(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass

    def fit(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 2, 2], 'c2': [3, 1, 2, 0]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> std.grps_ is not None
        True
        """
        # X may not be a pandas dataframe (e.g. a np.array)
        df = pd.DataFrame(X)
        
        # A dictionary of means/standard-deviations for each column, for each group.
        g = df.columns[0]
        self.grps_ = {}
        
        for gp in df[g]:
            for col in df.columns[1:]:
                vals = df[df[g] == gp][col]
                self.grps_.update({f'{gp}_{col}':[vals.mean(), vals.std()]})        

        return self

    def transform(self, X, y=None):
        """
        :Example:
        >>> cols = {'g': ['A', 'A', 'B', 'B'], 'c1': [1, 2, 3, 4], 'c2': [1, 2, 3, 4]}
        >>> X = pd.DataFrame(cols)
        >>> std = StdScalerByGroup().fit(X)
        >>> out = std.transform(X)
        >>> out.shape == (4, 2)
        True
        >>> np.isclose(out.abs(), 0.707107, atol=0.001).all().all()
        True
        """

        try:
            getattr(self, "grps_")
        except AttributeError:
            raise RuntimeError("You must fit the transformer before tranforming the data!")
        

        # Define a helper function here?
        def zscore(gp,col):
            ms = self.grps_[f'{gp}_{col}']
            try:
                return(gp_df[col].apply(lambda x: (x-ms[0])/ms[1]))
            except:
                return None


        # X may not be a dataframe (e.g. np.array)
        df = pd.DataFrame(X)
        new_df = pd.DataFrame()
        g = df.columns[0]
        for gp in df[g].unique():
            gp_df = (df[df[g] == gp])
            for col in df.columns[1:]:
                gp_df.loc[:,col] = zscore(gp,col)
            new_df = pd.concat([new_df,gp_df])

        return new_df.drop(g, axis=1)


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def eval_toy_model():
    """
    hardcoded answers to question 4

    :Example:
    >>> out = eval_toy_model()
    >>> len(out) == 3
    True
    """

    return [(2.7551086974518118,0.40068025802737284),(2.3148336164355277,0.5589841023411486),(2.3518910318553576,0.5449753026887407)]


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def tree_reg_perf(galton):
    """

    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = tree_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    >>> out['train_err'].iloc[-1] < out['test_err'].iloc[-1]
    True
    """

    X = galton.drop(['childHeight'], axis=1)
    y = galton.childHeight
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    l = []
    k = []
    for i in range(1,21):
        d = DecisionTreeRegressor(max_depth=i).fit(X_train, y_train)
        pred = d.predict(X_test)
        l.append(np.sqrt(np.mean((pred-y_test.to_numpy())**2)))
        pred = d.predict(X_train)
        k.append(np.sqrt(np.mean((pred-y_train.to_numpy())**2)))
    
    return pd.DataFrame({'train_err':k, 'test_err':l}) 


def knn_reg_perf(galton):
    """
    :Example:
    >>> galton_fp = os.path.join('data', 'galton.csv')
    >>> galton = pd.read_csv(galton_fp)
    >>> out = knn_reg_perf(galton)
    >>> out.columns.tolist() == ['train_err', 'test_err']
    True
    """

    X = galton.drop(['childHeight'], axis=1)
    y = galton.childHeight
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    l = []
    k = []
    for i in range(1,21):
        d = KNeighborsRegressor(n_neighbors=i).fit(X_train, y_train)
        pred = d.predict(X_test)
        l.append(np.sqrt(np.mean((pred-y_test.to_numpy())**2)))
        pred = d.predict(X_train)
        k.append(np.sqrt(np.mean((pred-y_train.to_numpy())**2)))
    
    return pd.DataFrame({'train_err':k, 'test_err':l})

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def titanic_model(titanic):
    """
    :Example:
    >>> fp = os.path.join('data', 'titanic.csv')
    >>> data = pd.read_csv(fp)
    >>> pl = titanic_model(data)
    >>> isinstance(pl, Pipeline)
    True
    >>> from sklearn.base import BaseEstimator
    >>> isinstance(pl.steps[-1][-1], BaseEstimator)
    True
    >>> preds = pl.predict(data.drop('Survived', axis=1))
    >>> ((preds == 0)|(preds == 1)).all()
    True
    """

    return ...

# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


def json_reader(file, iterations):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> isinstance(reviews, list)
    True
    >>> isinstance(labels, list)
    True
    >>> len(labels) == len(reviews)
    True
    """

    return ...


def create_classifier_multi(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
    True
    """
    
    return ...


def to_binary(labels):
    """
    :Example
    >>> lst = [1, 2, 3, 4, 5]
    >>> to_binary(lst)
    >>> print(lst)
    [0, 0, 0, 1, 1]
    """
    
    return ...


def create_classifier_binary(X, y):
    """
    :Example
    >>> fp = os.path.join('data', 'reviews.json')
    >>> reviews, labels = json_reader(fp, 5000)
    >>> to_binary(labels)
    >>> trial = create_classifier_multi(reviews, labels)
    >>> isinstance(trial, Pipeline)
    True
    """

    return ...


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['simple_pipeline'],
    'q02': ['multi_type_pipeline'],
    'q03': ['StdScalerByGroup'],
    'q04': ['eval_toy_model'],
    'q05': ['tree_reg_perf', 'knn_reg_perf'],
    'q06': ['titanic_model']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True