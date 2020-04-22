
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------

def car_null_hypoth():
    """
    Returns a list of valid null hypotheses.
    
    :Example:
    >>> set(car_null_hypoth()) <= set(range(1,11))
    True
    """
    return [3,8]


def car_alt_hypoth():
    """
    Returns a list of valid alternative hypotheses.
    
    :Example:
    >>> set(car_alt_hypoth()) <= set(range(1,11))
    True
    """
    return [2,4]


def car_test_stat():
    """
    Returns a list of valid test statistics.
    
    :Example:
    >>> set(car_test_stat()) <= set(range(1,5))
    True
    """
    return [2,4]


def car_p_value():
    """
    Returns an integer corresponding to the correct explanation.
    
    :Example:
    >>> car_p_value() in [1,2,3,4,5]
    True
    """
    return 3


# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------

def convert_kbyte(str):
    if str[-1] == 'k':
        return float(str.strip('k'))
    elif str[-1] == 'M':
        return float(str.strip('M')) * 0.001

def clean_apps(df):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> len(cleaned) == len(df)
    True
    >>> cleaned.Reviews.dtype == int
    True
    '''

    cleaned = df.copy()
    cleaned['Size'] = df['Size'].apply(convert_kbyte)
    cleaned['Installs'] = df['Installs'].str.strip('+').str.replace(',','').astype(int)
    cleaned['Type'] = df['Type'].replace({'Free': 1, 'Paid': 0})
    cleaned['Price'] = df['Price'].str.strip('$').astype(np.float64)
    cleaned['Last Updated'] = df['Last Updated'].str[-4:].astype(int)

    return cleaned


def store_info(cleaned):
    '''
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> df = pd.read_csv(fp)
    >>> cleaned = clean_apps(df)
    >>> info = store_info(cleaned)
    >>> len(info)
    4
    >>> info[2] in cleaned.Category.unique()
    True
    '''

    count = cleaned.groupby('Last Updated').count()
    years = count.loc[(count['App']) >= 100].index.values.tolist()
    first = cleaned[cleaned['Last Updated'].isin(years)].groupby('Last Updated')['Installs'].median().idxmax()

    second = cleaned.groupby('Content Rating')['Rating'].min().idxmax()

    third = cleaned.groupby('Category')['Price'].mean().idxmax()

    fourth = cleaned.loc[cleaned['Reviews'] >= 1000].groupby('Category')['Rating'].mean().idxmin()

    return [first, second, third, fourth]

# ---------------------------------------------------------------------
# Question 3
# ---------------------------------------------------------------------

def std_reviews_by_app_cat(cleaned):
    """
    >>> fp = os.path.join('data', 'googleplaystore.csv')
    >>> play = pd.read_csv(fp)
    >>> clean_play = clean_apps(play)
    >>> out = std_reviews_by_app_cat(clean_play)
    >>> set(out.columns) == set(['Category', 'Reviews'])
    True
    >>> np.all(abs(out.select_dtypes(include='number').mean()) < 10**-7)  # standard units should average to 0!
    True
    """

    df = pd.DataFrame()
    df['Category'] = cleaned['Category']
    df['Reviews'] = cleaned.groupby('Category')['Reviews'].transform(lambda x: (x - x.mean()) / x.std())

    return df


def su_and_spread():
    """
    >>> out = su_and_spread()
    >>> len(out) == 2
    True
    >>> out[0].lower() in ['medical', 'family', 'equal']
    True
    >>> out[1] in ['ART_AND_DESIGN', 'AUTO_AND_VEHICLES', 'BEAUTY',\
       'BOOKS_AND_REFERENCE', 'BUSINESS', 'COMICS', 'COMMUNICATION',\
       'DATING', 'EDUCATION', 'ENTERTAINMENT', 'EVENTS', 'FINANCE',\
       'FOOD_AND_DRINK', 'HEALTH_AND_FITNESS', 'HOUSE_AND_HOME',\
       'LIBRARIES_AND_DEMO', 'LIFESTYLE', 'GAME', 'FAMILY', 'MEDICAL',\
       'SOCIAL', 'SHOPPING', 'PHOTOGRAPHY', 'SPORTS', 'TRAVEL_AND_LOCAL',\
       'TOOLS', 'PERSONALIZATION', 'PRODUCTIVITY', 'PARENTING', 'WEATHER',\
       'VIDEO_PLAYERS', 'NEWS_AND_MAGAZINES', 'MAPS_AND_NAVIGATION']
    True
    """
    return ['equal', 'GAME']


# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


def read_survey(dirname):
    """
    read_survey combines all the survey*.csv files into a singular DataFrame
    :param dirname: directory name where the survey*.csv files are
    :returns: a DataFrame containing the combined survey data
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> out = read_survey(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> len(out)
    5000
    >>> read_survey('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """

    dfs = []

    for filename in os.listdir(dirname):
        if filename[:6] == 'survey':
            df = pd.read_csv(os.path.join(dirname, filename))
            df.columns = (df.columns.str.lower().str.replace('_', ' '))
            dfs.append(df)

    fin = pd.concat(dfs, sort=False)
    return fin[['first name', 'last name', 'current company', 'job title', 'email', 'university']]


def com_stats(df):
    """
    com_stats 
    :param df: a DataFrame containing the combined survey data
    :returns: a list containing the most common first name, job held, 
    university attended, and current company
    :Example:
    >>> dirname = os.path.join('data', 'responses')
    >>> df = read_survey(dirname)
    >>> out = com_stats(df)
    >>> len(out)
    4
    >>> all([isinstance(x, str) for x in out])
    True
    """

    df = df.loc[df['email'].str[-4:] == '.com'][['first name', 'job title', 'university', 'current company']]
    results = []

    for col in df:
        results.append(df[col].mode().max())

    return results


# ---------------------------------------------------------------------
# Question #5
# ---------------------------------------------------------------------

def combine_surveys(dirname):
    """
    combine_surveys takes in a directory path 
    (containing files favorite*.csv) and combines 
    all of the survey data into one DataFrame, 
    indexed by student ID (a value 0 - 1000).

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> out = combine_surveys(dirname)
    >>> isinstance(out, pd.DataFrame)
    True
    >>> out.shape
    (1000, 6)
    >>> combine_surveys('nonexistentfile') # doctest: +ELLIPSIS
    Traceback (most recent call last):
    ...
    FileNotFoundError: ... 'nonexistentfile'
    """

    dfs = []
    for filename in os.listdir(dirname):
        if (filename[:8] == 'favorite'):
            df = pd.read_csv(os.path.join(dirname, filename))
            df = df.set_index('id')
            dfs.append(df)

    return pd.concat(dfs, axis=1, sort=False)


def check_credit(df):
    """
    check_credit takes in a DataFrame with the 
    combined survey data and outputs a DataFrame 
    of the names of students and how many extra credit 
    points they would receive, indexed by their ID (a value 0-1000)

    :Example:
    >>> dirname = os.path.join('data', 'extra-credit-surveys')
    >>> df = combine_surveys(dirname)
    >>> out = check_credit(df)
    >>> out.shape
    (1000, 2)
    """

    cdf = df.copy()
    answers = df.drop('name', axis=1)
    credit = answers.count(axis=1).apply(lambda x: 5 if (x > (0.75 * answers.shape[1])) else 0)
    cdf['credit'] = credit

    ninety = False

    for col in answers:
        if (answers[col].count() / answers.shape[0]) > 0.9:
            ninety = True

    if ninety == True:
        cdf['credit'] = cdf['credit'] + 1

    return cdf[['name', 'credit']]

# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------


def at_least_once(pets, procedure_history):
    """
    How many pets have procedure performed at this clinic at least once.

    :Example:
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = at_least_once(pets, procedure_history)
    >>> out < len(pets)
    True
    """
    ids = procedure_history.groupby('PetID').count().index.values.tolist()
    return pets.loc[pets['PetID'].isin(ids)].shape[0]


def pet_name_by_owner(owners, pets):
    """
    pet names by owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> out = pet_name_by_owner(owners, pets)
    >>> len(out) == len(owners)
    True
    >>> 'Sarah' in out.index
    True
    >>> 'Cookie' in out.values
    True
    """
    return ...


def total_cost_per_owner(owners, pets, procedure_history, procedure_detail):
    """
    total cost per owner

    :Example:
    >>> owners_fp = os.path.join('data', 'pets', 'Owners.csv')
    >>> pets_fp = os.path.join('data', 'pets', 'Pets.csv')
    >>> procedure_detail_fp = os.path.join('data', 'pets', 'ProceduresDetails.csv')
    >>> procedure_history_fp = os.path.join('data', 'pets', 'ProceduresHistory.csv')

    >>> owners = pd.read_csv(owners_fp)
    >>> pets = pd.read_csv(pets_fp)
    >>> procedure_detail = pd.read_csv(procedure_detail_fp)
    >>> procedure_history = pd.read_csv(procedure_history_fp)
    >>> out = total_cost_per_owner(owners, pets, procedure_history, procedure_detail)
    >>> set(out.index) <= set(owners['OwnerID'])
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
    'q01': [
        'car_null_hypoth', 'car_alt_hypoth',
        'car_test_stat', 'car_p_value'
    ],
    'q02': ['clean_apps', 'store_info'],
    'q03': ['std_reviews_by_app_cat','su_and_spread'],
    'q04': ['read_survey', 'com_stats'],
    'q05': ['combine_surveys', 'check_credit'],
    'q06': ['at_least_once', 'pet_name_by_owner', 'total_cost_per_owner']
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
                In %s, part %s is missing" % (q, elt)
                raise Exception(stmt)

    return True
