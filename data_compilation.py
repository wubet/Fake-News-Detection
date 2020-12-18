import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords


def clean_text(df, text_field, new_text_field):
    """
    Remove all punctuations and marks
    :param df: data frame to be cleaned
    :param text_field: name of column to be cleaned
    :param new_text_field: result column
    :return: the original data frame with the result column
    """
    df[new_text_field] = df[text_field].str.lower()
    df[new_text_field] = df[new_text_field].apply(
        lambda elem: re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", elem)
    )
    return df


def get_dataset_from_file(dataset: str, is_fake: bool, is_news:bool):
    """
    Return a data frame from files
    :param dataset: name of the dataset
    :param is_fake: whether fake news
    :return: a data frame of the dataset
    """
    # list of news titles
    titles = []
    # list of news contents
    contents = []
    # category of the news
    categories = []

    quality = 'fake' if is_fake else 'legit'

    # grab data from celebrity dataset
    cwd = os.getcwd()
    fakeCelebDir = os.path.join(os.path.join(os.path.join(cwd, 'fakeNewsDatasets'), dataset), quality)

    for filename in os.listdir(fakeCelebDir):

        with open(os.path.join(fakeCelebDir, filename), 'r', encoding='UTF8') as file:
            # read and store the title
            title = file.readline()
            titles.append(title)

            # read and store the content
            content_lines = file.read().splitlines()  # remove new-line characters
            content = " ".join(content_lines)
            contents.append(content)

        # assign the category
        # business      = 1
        # education     = 2
        # entertainment = 3
        # politics      = 4
        # sport         = 5
        # tech          = 6
        # celebrity     = 7
        if is_news:
            if filename[0:3] == "biz": categories.append(1)
            if filename[0:3] == "edu": categories.append(2)
            if filename[0:3] == "ent": categories.append(3)
            if filename[0:3] == "pol": categories.append(4)
            if filename[0:3] == "spo": categories.append(5)
            if filename[0:3] == "tec": categories.append(6)
        else:
            categories.append(7)

    newsDf = pd.DataFrame({'Title': titles, 'Content': contents, 'Category': categories})

    if is_fake:
        newsDf['is_fake'] = 1
    else:
        newsDf['is_fake'] = 0
    if is_news:
        newsDf['is_news'] = 1
    else:
        newsDf['is_news'] = 0

    return newsDf


def build_real_news_dataframe(include_celeb: bool = False):
    """
    Build real news data frame
    :param include_celeb: whether or nor to include the celebrity dataset
    :return: the real news data frame
    """
    # real news dataset
    #column_names = ['Title', 'Content']
    realDf = pd.DataFrame()

    # build the dataf rame with the non-celeb news
    realDf = realDf.append(get_dataset_from_file("fakeNewsDataset", False, True), ignore_index=True)

    # include data from the celeb news
    if include_celeb:
        realDf = realDf.append(get_dataset_from_file("celebrityDataset", False, False), ignore_index=True)

    return realDf


def build_fake_news_dataframe(include_celeb: bool = False):
    """
    Build fake news data frame
    :param include_celeb: whether or not to include the celebrity dataset
    :return: the fake news data frame
    """
    # fake new dataset
    fakeDf = pd.DataFrame()

    # build the data frame with the non-celeb news
    fakeDf = fakeDf.append(get_dataset_from_file("fakeNewsDataset", True, True), ignore_index=True)

    # include data from the celeb news
    if include_celeb:
        fakeDf = fakeDf.append(get_dataset_from_file("celebrityDataset", True, False), ignore_index=True)

    return fakeDf
