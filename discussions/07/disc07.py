import requests


def url_list():
    """
    A list of urls to scrape.

    :Example:
    >>> isinstance(url_list(), list)
    True
    >>> len(url_list()) > 1
    True
    """

    return [f'http://example.webscraping.com/places/default/index/{x}' for x in range(0,26)]


def request_until_successful(url, N):
    """
    impute (i.e. fill-in) the missing values of each column 
    using the last digit of the value of column A.

    :Example:
    >>> resp = request_until_successful('http://quotes.toscrape.com', N=1)
    >>> resp.ok
    True
    >>> resp = request_until_successful('http://example.webscraping.com/', N=1)
    >>> isinstance(resp, requests.models.Response) or (resp is None)
    True
    """

    for _ in range(N):
        resp = requests.get(url)
        if resp.status_code == 200:
            return resp
    return None
