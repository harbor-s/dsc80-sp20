
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# Question # 0
# ---------------------------------------------------------------------

def consecutive_ints(ints):
    """
    consecutive_ints tests whether a list contains two 
    adjacent elements that are consecutive integers.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two 
    adjacent elements that are consecutive integers.

    :Example:
    >>> consecutive_ints([5,3,6,4,9,8])
    True
    >>> consecutive_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# Question # 1 
# ---------------------------------------------------------------------

def median(nums):
    """
    median takes a non-empty list of numbers,
    returning the median element of the list.
    If the list has even length, it should return
    the mean of the two elements in the middle.

    :param nums: a non-empty list of numbers.
    :returns: the median of the list.
    
    :Example:
    >>> median([6, 5, 4, 3, 2]) == 4
    True
    >>> median([50, 20, 15, 40]) == 30
    True
    >>> median([1, 2, 3, 4]) == 2.5
    True
    """
    
    while len(nums) > 2:
        nums.remove(min(nums))
        nums.remove(max(nums))
        
    if len(nums) == 2:
        return (nums[0] + nums[1])/2
    elif len(nums) == 1:
        return nums[0]


# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def same_diff_ints(ints):
    """
    same_diff_ints tests whether a list contains
    two list elements i places apart, whose distance
    as integers is also i.

    :param ints: a list of integers
    :returns: a boolean value if ints contains two
    elements as described above.

    :Example:
    >>> same_diff_ints([5,3,1,5,9,8])
    True
    >>> same_diff_ints([1,3,5,7,9])
    False
    """

    if len(ints) == 0:
        return False

    for d in range(1,len(ints)):
        for k in range(len(ints) - d):
            diff = abs(ints[k] - ints[k+d])
            if diff == d:
                return True
    
    return False


# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def prefixes(s):
    """
    prefixes returns a string of every 
    consecutive prefix of the input string.

    :param s: a string.
    :returns: a string of every consecutive prefix of s.

    :Example:
    >>> prefixes('Data!')
    'DDaDatDataData!'
    >>> prefixes('Marina')
    'MMaMarMariMarinMarina'
    >>> prefixes('aaron')
    'aaaaaraaroaaron'
    """


    str = ''
    for n in range(len(s)):
        str += (s[:n+1])
    
    return str


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def evens_reversed(N):
    """
    evens_reversed returns a string containing 
    all even integers from  1  to  N  (inclusive)
    in reversed order, separated by spaces. 
    Each integer is zero padded.

    :param N: a non-negative integer.
    :returns: a string containing all even integers 
    from 1 to N reversed, formatted as decsribed above.

    :Example:
    >>> evens_reversed(7)
    '6 4 2'
    >>> evens_reversed(10)
    '10 08 06 04 02'
    """
    if N == '':
        return ''
    if N == 0:
        return ''
    
    if N % 2 == 1:
        N -= 1
    
    padding = len(str(N))
    
    string = str(N)
    
    while N > 2:
        N = N - 2
        string +=' '
        string += str(N).zfill(padding)
    
    return string


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------

def last_chars(fh):
    """
    last_chars takes a file object and returns a 
    string consisting of the last character of the line.

    :param fh: a file object to read from.
    :returns: a string of last characters from fh

    :Example:
    >>> fp = os.path.join('data', 'chars.txt')
    >>> last_chars(open(fp))
    'hrg'
    """

    lc = ''
    for line in fh:
        lc += line[-2:-1]
    return lc


# ---------------------------------------------------------------------
# Question # 6
# ---------------------------------------------------------------------

def arr_1(A):
    """
    arr_1 takes in a numpy array and
    adds to each element the square-root of
    the index of each element.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> A = np.array([2, 4, 6, 7])
    >>> out = arr_1(A)
    >>> isinstance(out, np.ndarray)
    True
    >>> np.all(out >= A)
    True
    """
    
    ind_sqrt = np.sqrt(np.arange(0,len(A)))
    A = A + ind_sqrt

    return A


def arr_2(A):
    """
    arr_2 takes in a numpy array of integers
    and returns a boolean array (i.e. an array of booleans)
    whose ith element is True if and only if the ith element
    of the input array is divisble by 16.

    :param A: a 1d numpy array.
    :returns: a 1d numpy boolean array.

    :Example:
    >>> out = arr_2(np.array([1, 2, 16, 17, 32, 33]))
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('bool')
    True
    """

    return A%16 == 0


def arr_3(A):
    """
    arr_3 takes in a numpy array of stock
    prices per share on successive days in
    USD and returns an array of growth rates.

    :param A: a 1d numpy array.
    :returns: a 1d numpy array.

    :Example:
    >>> fp = os.path.join('data', 'stocks.csv')
    >>> stocks = np.array([float(x) for x in open(fp)])
    >>> out = arr_3(stocks)
    >>> isinstance(out, np.ndarray)
    True
    >>> out.dtype == np.dtype('float')
    True
    >>> out.max() == 0.03
    True
    """

    changes = np.diff(A)
    days_tracked = A[:-1]
    rates = (changes/days_tracked)
    return np.around(rates, 2)

def arr_4(A):
    """
    Create a function arr_4 that takes in A and 
    returns the day on which you can buy at least 
    one share from 'left-over' money. If this never 
    happens, return -1. The first stock purchase occurs on day 0
    :param A: a 1d numpy array of stock prices.
    :returns: an integer of the total number of shares.

    :Example:
    >>> import numbers
    >>> stocks = np.array([3, 3, 3, 3])
    >>> out = arr_4(stocks)
    >>> isinstance(out, numbers.Integral)
    True
    >>> out == 1
    True
    """

    leftovers = np.cumsum(20 % A)
    
    stocks_buyable = leftovers//A
    
    target = np.where(stocks_buyable > 0)
    if len(target[0]) == 0:
        return -1
    else:
        return target[0][0]
    
    return stocks_buyable


# ---------------------------------------------------------------------
# Question # 7
# ---------------------------------------------------------------------

def movie_stats(movies):
    """
    movies_stats returns a series as specified in the notebook.

    :param movies: a dataframe of summaries of
    movies per year as found in `movies_by_year.csv`
    :return: a series with index specified in the notebook.

    :Example:
    >>> movie_fp = os.path.join('data', 'movies_by_year.csv')
    >>> movies = pd.read_csv(movie_fp)
    >>> out = movie_stats(movies)
    >>> isinstance(out, pd.Series)
    True
    >>> 'num_years' in out.index
    True
    >>> isinstance(out.loc['second_lowest'], str)
    True
    """

    num_years = movies['Year'].max() - movies['Year'].min()
    tot_movies = movies['Number of Movies'].sum()
    yr_fewest_movies = movies.loc[movies['Number of Movies'].idxmin(), 'Year']
    avg_gross = movies['Total Gross'].mean()

    movies['Gross per Movie'] = movies['Total Gross']/movies['Number of Movies']
    highest_per_movie = movies.loc[movies['Gross per Movie'].idxmax(), 'Year']

    two_smallest = movies.nsmallest(2, 'Total Gross')
    second_lowest = two_smallest.loc[two_smallest['Total Gross'].idxmax(), '#1 Movie']

    movies_after_Harry = movies[movies['#1 Movie'].str.contains('Harry')]['Year'] + 1
    avg_after_harry = movies[movies['Year'].isin(movies_after_Harry)]['Number of Movies'].mean()

    responses = pd.Series([num_years, tot_movies, yr_fewest_movies, avg_gross, highest_per_movie, second_lowest, avg_after_harry], 
        index = ['num_years', 'tot_movies', 'yr_fewest_movies', 'avg_gross', 'highest_per_movie', 'second_lowest', 'avg_after_harry'])

    return responses
    

# ---------------------------------------------------------------------
# Question # 8
# ---------------------------------------------------------------------

def parse_malformed(fp):
    """
    Parses and loads the malformed csv data into a 
    properly formatted dataframe (as described in 
    the question).

    :param fh: file handle for the malformed csv-file.
    :returns: a Pandas DataFrame of the data, 
    as specificed in the question statement.

    :Example:
    >>> fp = os.path.join('data', 'malformed.csv')
    >>> df = parse_malformed(fp)
    >>> cols = ['first', 'last', 'weight', 'height', 'geo']
    >>> list(df.columns) == cols
    True
    >>> df['last'].dtype == np.dtype('O')
    True
    >>> df['height'].dtype == np.dtype('float64')
    True
    >>> df['geo'].str.contains(',').all()
    True
    >>> len(df) == 100
    True
    >>> dg = pd.read_csv(fp, nrows=4, skiprows=10, names=cols)
    >>> dg.index = range(9, 13)
    >>> (dg == df.iloc[9:13]).all().all()
    True
    """

    first = []
    last = []
    weight = []
    height = []
    geo = []
    
    with open(fp) as fh:
        next(fh)
        for line in fh:
            l = line.split(',')
            
            to_delete = []
            
            for i in range(0, len(l)):
                l[i] = l[i].replace('"', '')
                l[i] = l[i].replace('\n', '')
                if l[i] == '':
                    to_delete.append(i)
            
            for i in to_delete:
                del l[i]
            
            first.append(l[0])
            last.append(l[1])
            weight.append(float(l[2]))
            height.append(float(l[3]))
            geo.append(l[4]+','+l[5])
            
    d = {'first' : first,
        'last' : last,
        'weight' : weight,
        'height' : height,
        'geo' : geo}
    return pd.DataFrame(d)


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q00': ['consecutive_ints'],
    'q01': ['median'],
    'q02': ['same_diff_ints'],
    'q03': ['prefixes'],
    'q04': ['evens_reversed'],
    'q05': ['last_chars'],
    'q06': ['arr_%d' % d for d in range(1, 5)],
    'q07': ['movie_stats'],
    'q08': ['parse_malformed']
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
