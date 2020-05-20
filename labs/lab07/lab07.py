import os
import pandas as pd
import numpy as np
import requests
import json
import re



# ---------------------------------------------------------------------
# Question # 1
# ---------------------------------------------------------------------


####################
#  Regex
####################



# ---------------------------------------------------------------------
# Problem 1
# ---------------------------------------------------------------------

def match_1(string):
    """
    >>> match_1("abcde]")
    False
    >>> match_1("ab[cde")
    False
    >>> match_1("a[cd]")
    False
    >>> match_1("ab[cd]")
    True
    >>> match_1("1ab[cd]")
    False
    >>> match_1("ab[cd]ef")
    True
    >>> match_1("1b[#d] _")
    True
    """
    #Your Code Here
    pattern = '^..\[..\]'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_2(string):
    """
    Phone numbers that start with '(858)' and
    follow the format '(xxx) xxx-xxxx' (x represents a digit)
    Notice: There is a space between (xxx) and xxx-xxxx

    >>> match_2("(123) 456-7890")
    False
    >>> match_2("858-456-7890")
    False
    >>> match_2("(858)45-7890")
    False
    >>> match_2("(858) 456-7890")
    True
    >>> match_2("(858)456-789")
    False
    >>> match_2("(858)456-7890")
    False
    >>> match_2("a(858) 456-7890")
    False
    >>> match_2("(858) 456-7890b")
    False
    """
    #Your Code Here
    pattern = '^\(858\) \d{3}-\d{4}$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None




def match_3(string):
    """
    Find a pattern whose length is between 6 to 10
    and contains only word character, white space and ?.
    This string must have ? as its last character.

    >>> match_3("qwertsd?")
    True
    >>> match_3("qw?ertsd?")
    True
    >>> match_3("ab c?")
    False
    >>> match_3("ab   c ?")
    True
    >>> match_3(" asdfqwes ?")
    False
    >>> match_3(" adfqwes ?")
    True
    >>> match_3(" adf!qes ?")
    False
    >>> match_3(" adf!qe? ")
    False
    """
    #Your Code Here

    pattern = '^[\w \?]{5,9}\?$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_4(string):
    """
    A string that begins with '$' and with another '$' within, where:
        - Characters between the two '$' can be anything except the 
        letters 'a', 'b', 'c' (lower case).
        - Characters after the second '$' can only have any number 
        of the letters 'a', 'b', 'c' (upper or lower case), with every 
        'a' before every 'b', and every 'b' before every 'c'.
            - E.g. 'AaBbbC' works, 'ACB' doesn't.

    >>> match_4("$$AaaaaBbbbc")
    True
    >>> match_4("$!@#$aABc")
    True
    >>> match_4("$a$aABc")
    False

    >>> match_4("$iiuABc")
    False
    >>> match_4("123$Abc")
    False
    >>> match_4("$$Abc")
    True
    >>> match_4("$qw345t$AAAc")
    False
    >>> match_4("$s$Bca")
    False
    """
    #Your Code Here
    pattern = '^\$[^a-c\$]*\$[aA]+[bB]+[cC]+$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_5(string):
    """
    A string that represents a valid Python file name including the extension.
    *Notice*: For simplicity, assume that the file name contains only letters, numbers and an underscore `_`.

    >>> match_5("dsc80.py")
    True
    >>> match_5("dsc80py")
    False
    >>> match_5("dsc80..py")
    False
    >>> match_5("dsc80+.py")
    False
    """

    #Your Code Here
    pattern = '^\w+\.py$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_6(string):
    """
    Find patterns of lowercase letters joined with an underscore.
    >>> match_6("aab_cbb_bc")
    False
    >>> match_6("aab_cbbbc")
    True
    >>> match_6("aab_Abbbc")
    False
    >>> match_6("abcdef")
    False
    >>> match_6("ABCDEF_ABCD")
    False
    """

    #Your Code Here
    pattern = '^[a-z]+_[a-z]+$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_7(string):
    """
    Find patterns that start with and end with a _
    >>> match_7("_abc_")
    True
    >>> match_7("abd")
    False
    >>> match_7("bcd")
    False
    >>> match_7("_ncde")
    False
    """

    pattern = '^_.*_$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None




def match_8(string):
    """
    Apple registration numbers and Apple hardware product serial numbers
    might have the number "0" (zero), but never the letter "O".
    Serial numbers don't have the number "1" (one) or the letter "i".

    Write a line of regex expression that checks
    if the given Serial number belongs to a genuine Apple product.

    >>> match_8("ASJDKLFK10ASDO")
    False
    >>> match_8("ASJDKLFK0ASDo")
    True
    >>> match_8("JKLSDNM01IDKSL")
    False
    >>> match_8("ASDKJLdsi0SKLl")
    False
    >>> match_8("ASDJKL9380JKAL")
    True
    """

    pattern = '^[^O1i]*$'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None



def match_9(string):
    '''
    >>> match_9('NY-32-NYC-1232')
    True
    >>> match_9('ca-23-SAN-1231')
    False
    >>> match_9('MA-36-BOS-5465')
    False
    >>> match_9('CA-56-LAX-7895')
    True
    '''


    pattern = '(CA-\d{2}-(LAX|SAN)-\d{4})|(NY-\d{2}-[A-Z]{3}-\d{4})'

    #Do not edit following code
    prog = re.compile(pattern)
    return prog.search(string) is not None


def match_10(string):
    '''
    Given an input string, cast it to lower case, remove spaces/punctuation, 
    and return a list of every 3-character substring that satisfy the following:
        - The first character doesn't start with 'a' or 'A'
        - The last substring (and only the last substring) can be shorter than 
        3 characters, depending on the length of the input string.
    
    >>> match_10('ABCdef')
    ['def']
    >>> match_10(' DEFaabc !g ')
    ['def', 'cg']
    >>> match_10('Come ti chiami?')
    ['com', 'eti', 'chi']
    >>> match_10('and')
    []
    >>> match_10( "Ab..DEF")
    ['def']

    '''

    s = string.lower()
    s = re.sub('[aA].{2}', '', s)
    s = re.sub('[^A-Za-z\d]+', '', s)
    return [s[i:i+3] for i in range(0, len(s), 3)]



# ---------------------------------------------------------------------
# Question # 2
# ---------------------------------------------------------------------

def extract_personal(s):
    """
    :Example:
    >>> fp = os.path.join('data', 'messy.test.txt')
    >>> s = open(fp, encoding='utf8').read()
    >>> emails, ssn, bitcoin, addresses = extract_personal(s)
    >>> emails[0] == 'test@test.com'
    True
    >>> ssn[0] == '423-00-9575'
    True
    >>> bitcoin[0] == '1BvBMSEYstWetqTFn5Au4m4GFg7xJaNVN2'
    True
    >>> addresses[0] == '530 High Street'
    True
    """

    emails = re.findall('([A-Za-z0-9]+@[A-Za-z0-9]+\.[A-Za-z0-9]+\.[A-Za-z0-9]+)|([A-Za-z0-9]+@[A-Za-z0-9]+\.[A-Za-z0-9]+)',s)
    emails = [x[0] if (len(x[0]) > 0) else x[1] for x in emails]
    ssns = re.findall('\d{3}-\d{2}-\d{4}',s)
    bitcoins = re.findall('(?<=bitcoin:)[A-Za-z0-9]{20}[A-Za-z0-9]*',s)
    addresses = re.findall('\d+ [A-Z][a-z]+ [A-Z][a-z]+',s)
    return (emails,ssns,bitcoins,addresses)

# ---------------------------------------------------------------------
# Question # 3
# ---------------------------------------------------------------------

def tfidf_data(review, reviews):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> out['cnt'].sum()
    85
    >>> 'before' in out.index
    True
    """
    words = pd.Series(filter(None, review.split(' ')))
    df = pd.DataFrame([], index=set(words.tolist()))
    df = df.assign(cnt=[(' '+review+' ').count(' '+word+' ') for word in df.index])
    df = df.assign(tf=df.cnt/len(words))
    df = df.assign(idf=[np.log(len(reviews) / reviews.str.contains(word).sum()) for word in df.index])
    df = df.assign(tfidf = df.tf * df.idf)
    return df

def relevant_word(out):
    """
    :Example:
    >>> fp = os.path.join('data', 'reviews.txt')
    >>> reviews = pd.read_csv(fp, header=None, squeeze=True)
    >>> review = open(os.path.join('data', 'review.txt'), encoding='utf8').read().strip()
    >>> out = tfidf_data(review, reviews)
    >>> relevant_word(out) in out.index
    True
    """
    return out['tfidf'].idxmax()


# ---------------------------------------------------------------------
# Question # 4
# ---------------------------------------------------------------------

def hashtag_list(tweet_text):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = hashtag_list(test['text'])
    >>> (out.iloc[0] == ['NLP', 'NLP1', 'NLP1'])
    True
    """

    return pd.Series([re.findall('(?<=#)[^\s]*',tweet_text[x]) for x in tweet_text.index])


def most_common_hashtag(tweet_lists):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = hashtag_list(pd.DataFrame(testdata, columns=['text'])['text'])
    >>> most_common_hashtag(test).iloc[0]
    'NLP1'
    """

    flat = []
    for t in tweet_lists:
        for i in t:
            flat.append(i)
    most_common = max(set(flat), key=flat.count)

    l = []
    for t in tweet_lists:
        if len(t) == 1:
            l.append(t[0])
        if len(t) == 0:
            l.append(np.NaN)
        else:
            l.append(most_common)

    return pd.Series(l)


# ---------------------------------------------------------------------
# Question # 5
# ---------------------------------------------------------------------


def clean_text(ira):
    t = [re.sub('[ ]*#[^\s]*','',x) for x in ira['text']]
    t = [re.sub('[ ]*http[s]*://[^\s]*','',x) for x in t]
    t = [re.sub('[ ]*@[^\s]*','',x) for x in t]
    t = [re.sub('[ ]*RT[ ]*','',x) for x in t]
    t = [re.sub('-', ' ', x) for x in t]
    t = [re.sub('[^A-Za-z0-9 ]','',x) for x in t]
    t = [re.sub('[ ]+',' ',x) for x in t]
    t = [x.lower() for x in t]
    return pd.Series(t)
    
def create_features(ira):
    """
    :Example:
    >>> testdata = [['RT @DSC80: Text-cleaning is cool! #NLP https://t.co/xsfdw88d #NLP1 #NLP1']]
    >>> test = pd.DataFrame(testdata, columns=['text'])
    >>> out = create_features(test)
    >>> anscols = ['text', 'num_hashtags', 'mc_hashtags', 'num_tags', 'num_links', 'is_retweet']
    >>> ansdata = [['text cleaning is cool', 3, 'NLP1', 1, 1, True]]
    >>> ans = pd.DataFrame(ansdata, columns=anscols)
    >>> (out == ans).all().all()
    True
    """

    df = pd.DataFrame([], index=ira.index)
    df = df.assign(text = clean_text(ira))
    df = df.assign(num_hashtags = [len(l) for l in hashtag_list(ira['text'])])
    df = df.assign(mc_hashtags = most_common_hashtag(hashtag_list(ira['text'])))
    df = df.assign(num_tags = [len(t) for t in [re.findall('@',x) for x in ira['text']]])
    df = df.assign(num_links = [len(l) for l in[re.findall('http[s]*://[^\s]*',x) for x in ira['text']]])
    df = df.assign(is_retweet = [re.search('RT',x) is not None for x in ira['text']])
    return df

# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['match_%d' % x for x in range(1, 10)],
    'q02': ['extract_personal'],
    'q03': ['tfidf_data', 'relevant_word'],
    'q04': ['hashtag_list', 'most_common_hashtag'],
    'q05': ['create_features']
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
