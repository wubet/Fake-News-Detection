import numpy as np
import pandas as pd
from tabulate import tabulate
from nltk import ngrams
import collections
import matplotlib.pyplot as plt


def validate_null_value(news_df):
    """
    Check the input dataframe for null values
    :param news_df: input dataframe
    :return: None
    """
    print(tabulate(news_df.head(5), headers='keys', tablefmt='psql'))
    print(news_df.shape)

    # check for a column with any null value
    print("List the count of null values for each column")
    sum_null_value = news_df.isnull().sum()
    print(sum_null_value)

    # check for a column with any null value
    print("List the count of nan values for each column")
    print(np.sum(pd.isna(news_df)))

    # validate if there is any null value
    is_null = news_df.isnull().values.any()
    print("Is there any null value? " + str(is_null))


def validate_unique_record(news_df):
    """
    Check unique values in the label
    :param news_df: Input dataframe
    :return: None
    """
    # Find the unique value of is_fake column
    print("Display the unique Values of is_fake")
    print(news_df.is_fake.unique())

    # Find the unique value of is_news column
    print("Display the unique Values of is_news")
    print(news_df.is_news.unique())


def word_frequency(text: str):
    """
    Get frequency of individual words in a text corpus

    :param text: text corpus
    :return: Counter object of tuples of words and their frequencies
    """
    # get individual words
    tokenized = text.split()

    # count the frequency
    word_counter = collections.Counter(tokenized)
    #frequencies = list(collections.Counter(tokenized).items())

    return word_counter


def ngram_frequency(text: str, n: int = 2):
    """
    Get ngrams frequency in a text corpus

    :param text: the text of interest
    :param n: n for ngram. Default = 2
    :return: Counter object of tuples of ngrams and their frequencies
    """
    # get individual words
    tokenized = text.split()

    # get the list of all the ngrams
    ngram_list = ngrams(tokenized, n)

    # get the frequency of each ngram in the corpus
    ngram_counter = collections.Counter(ngram_list)

    return ngram_counter


