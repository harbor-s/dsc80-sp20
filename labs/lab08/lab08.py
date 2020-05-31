import pandas as pd
import numpy as np
import seaborn as sns
import itertools as iter

from sklearn.preprocessing import Binarizer, QuantileTransformer, FunctionTransformer

# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


def best_transformation():
    """
    Returns an integer corresponding to the correct option.

    :Example:
    >>> best_transformation() in [1,2,3,4]
    True
    """

    # take log and square root of the dataset
    # look at the fit of the regression line (and R^2)

    return 1

# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------


def create_ordinal_col(col,key):
    ord_col = col.apply(lambda x: key.index(x)).values
    return ord_col

def create_ordinal(df):
    """
    create_ordinal takes in diamonds and returns a dataframe of ordinal
    features with names ordinal_<col> where <col> is the original
    categorical column name.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_ordinal(diamonds)
    >>> set(out.columns) == {'ordinal_cut', 'ordinal_clarity', 'ordinal_color'}
    True
    >>> np.unique(out['ordinal_cut']).tolist() == [0, 1, 2, 3, 4]
    True
    """
    
    ord_df_dict = { 'ordinal_cut' : create_ordinal_col(df.cut, ['Fair','Good','Very Good','Premium','Ideal']),
    'ordinal_color' : create_ordinal_col(df.color, ['J','I','H','G','F','E','D']),
    'ordinal_clarity' : create_ordinal_col(df.clarity, ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])}

    return pd.DataFrame(ord_df_dict)

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------


def one_hot_cols(colname,col,vals):
    d = dict()
    for val in vals:
        d.update({f'one_hot_{colname}_{val}':[1 if x == val else 0 for x in col.values.tolist()]})
    return d

def create_one_hot(df):
    """
    create_one_hot takes in diamonds and returns a dataframe of one-hot 
    encoded features with names one_hot_<col>_<val> where <col> is the 
    original categorical column name, and <val> is the value found in 
    the categorical column <col>.

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_one_hot(diamonds)
    >>> out.shape == (53940, 20)
    True
    >>> out.columns.str.startswith('one_hot').all()
    True
    >>> out.isin([0,1]).all().all()
    True
    """
    
    one_hot_dict = dict()

    one_hot_dict.update(one_hot_cols('cut', df.cut, ['Fair','Good','Very Good','Premium','Ideal']))
    one_hot_dict.update(one_hot_cols('color', df.color, ['J','I','H','G','F','E','D']))
    one_hot_dict.update(one_hot_cols('clarity', df.clarity, ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']))
    
    return pd.DataFrame(one_hot_dict)


def create_prop_col(col,vals):
    key = dict(zip(vals,[col[col==val].shape[0]/col.shape[0] for val in vals]))
    return col.replace(key)

def create_proportions(df):
    """
    create_proportions takes in diamonds and returns a 
    dataframe of proportion-encoded features with names 
    proportion_<col> where <col> is the original 
    categorical column name.

    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_proportions(diamonds)
    >>> out.shape[1] == 3
    True
    >>> out.columns.str.startswith('proportion_').all()
    True
    >>> ((out >= 0) & (out <= 1)).all().all()
    True
    """

    prop_df_dict = { 'proportion_cut' : create_prop_col(df.cut, ['Fair','Good','Very Good','Premium','Ideal']),
    'proportion_color' : create_prop_col(df.color, ['J','I','H','G','F','E','D']),
    'proportion_clarity' : create_prop_col(df.clarity, ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])}

    return pd.DataFrame(prop_df_dict)

# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------


def create_quadratics(df):
    """
    create_quadratics that takes in diamonds and returns a dataframe 
    of quadratic-encoded features <col1> * <col2> where <col1> and <col2> 
    are the original quantitative columns 
    (col1 and col2 should be distinct columns).

    :Example:
    >>> diamonds = sns.load_dataset('diamonds')
    >>> out = create_quadratics(diamonds)
    >>> out.columns.str.contains(' * ').all()
    True
    >>> ('x * z' in out.columns) or ('z * x' in out.columns)
    True
    >>> out.shape[1] == 15
    True
    """
        
    quad_df = pd.DataFrame()
    cols = ['carat', 'depth', 'table', 'x', 'y', 'z']
    for pair in list(iter.combinations(cols,r=2)):
        quad_df[f'{pair[0]} * {pair[1]}'] = df[pair[0]] * df[pair[1]]
    return quad_df


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def comparing_performance():
    """
    Hard coded answers to comparing_performance.

    :Example:
    >>> out = comparing_performance()
    >>> len(out) == 6
    True
    >>> import numbers
    >>> isinstance(out[0], numbers.Real)
    True
    >>> all(isinstance(x, str) for x in out[2:-1])
    True
    >>> 0 <= out[-1] <= 1
    True
    """

    # create a model per variable => (variable, R^2, RMSE) table

    return [0.8493, 1548.533193, 'x', 'carat * x', 'color', 0.013]

# ---------------------------------------------------------------------
# Question # 6, 7, 8
# ---------------------------------------------------------------------


class TransformDiamonds(object):
    
    def __init__(self, diamonds):
        self.data = diamonds
        
    def transformCarat(self, data):
        """
        transformCarat takes in a dataframe like diamonds 
        and returns a binarized carat column (an np.ndarray).

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transformCarat(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> transformed[172, 0] == 1
        True
        >>> transformed[0, 0] == 0
        True
        """

        return Binarizer(threshold = 1).transform(self.data[['carat']])
    
    def transform_to_quantile(self, data):
        """
        transform_to_quantiles takes in a dataframe like diamonds 
        and returns an np.ndarray of quantiles of the weight 
        (i.e. carats) of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds')
        >>> out = TransformDiamonds(diamonds.head(10))
        >>> transformed = out.transform_to_quantile(diamonds)
        >>> isinstance(transformed, np.ndarray)
        True
        >>> 0.2 <= transformed[0,0] <= 0.5
        True
        >>> np.isclose(transformed[1,0], 0, atol=1e-06)
        True
        """

        return QuantileTransformer().fit(self.data[['carat']]).transform(self.data[['carat']])
    
    def transform_to_depth_pct(self, data):
        """
        transform_to_volume takes in a dataframe like diamonds 
        and returns an np.ndarray consisting of the approximate 
        depth percentage of each diamond.

        :Example:
        >>> diamonds = sns.load_dataset('diamonds').drop(columns='depth')
        >>> out = TransformDiamonds(diamonds)
        >>> transformed = out.transform_to_depth_pct(diamonds)
        >>> len(transformed.shape) == 1
        True
        >>> np.isclose(transformed[0], 61.286, atol=0.0001)
        True
        """

        def get_depth_pct(l):
            x = l[0]
            y = l[1]
            z = l[2]
            return ((z/((x+y)/2)) * 100)

        return FunctionTransformer(func=get_depth_pct).transform(np.array([data.x,data.y,data.z]))


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['best_transformation'],
    'q02': ['create_ordinal'],
    'q03': ['create_one_hot', 'create_proportions'],
    'q04': ['create_quadratics'],
    'q05': ['comparing_performance'],
    'q06,7,8': ['TransformDiamonds']
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
