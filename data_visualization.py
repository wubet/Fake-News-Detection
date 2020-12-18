import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import nltk
import plotly.express as px
from sklearn.metrics import confusion_matrix

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    )

def visualize_real_fake(news_df):
    '''
    displaying ligit and fake count
    :param news_df: dataset
    :return:
    '''
    news_md_df = news_df.copy()
    news_md_df['is_fake'] = pd.Series(np.where(news_md_df.is_fake.values == 1, True, False),
              news_md_df.index)
    is_fake_data = news_md_df[(news_md_df.is_fake == True)][['is_fake']]
    is_no_fake_data = news_md_df[(news_md_df.is_fake == False)][['is_fake']]
    plotSingleHistogram(is_fake_data, is_no_fake_data, "Is Fake", "Count", "Ligit and Fake Count")

def visualize_news_celebrity(news_df):
    '''
    displaying news and celebrity count
    :param news_df: dataset
    :return:
    '''
    news_md_df = news_df.copy()
    news_md_df['is_fake'] = pd.Series(np.where(news_md_df.is_news.values == 1, True, False),
                                      news_md_df.index)
    is_news_data = news_md_df[(news_md_df.is_news == True)][['is_news']]
    is_no_news_data = news_md_df[(news_md_df.is_news == False)][['is_news']]
    # bins = compute_histogram_bins(is_news_data, 10)
    plotSingleHistogram(is_news_data, is_no_news_data, "Is News", "Count", "News and Celebroty Count")

# visualize content catagory
# def visulaize_content_catagory(df):
#     plt.figure(figsize = (8,8))
#     sns.countplot(y="subject", data= df)
#     plt.show()

# visualis fake and legit
# def visulaize_fake_legit(df):
#     plt.figure(figsize = (8,8))
#     sns.countplot(y="is_fake", data= df)
#     plt.show()

def visualize_fake_word_cloud_plot(df, stop_words):
    '''
    visulaiz main key words of the fake news
    :param df: dataset
    :param stop_words: list of words to be excluded from the visualization
    :return:
    '''
    plt.figure(figsize=(12,12))
    wc = WordCloud(max_words = 2000, width = 8000, height = 8000, stopwords = stop_words).generate(" ".join(df[df.is_fake==1].clean_joined))
    plt.imshow(wc, interpolation='bilinear')
    plt.show()

def visualize_legit_word_cloud_plot(df, stop_words):
    '''
    visualize main key words of the legit news
    :param df: dataset
    :param stop_words: list of words to be excluded from the visulaization
    :return:
    '''
    plt.figure(figsize=(12,12))
    wc = WordCloud(max_words = 2000, width = 8000, height = 8000, stopwords = stop_words).generate(" ".join(df[df.is_fake==0].clean_joined))
    plt.imshow(wc, interpolation='bilinear')
    plt.show()

def visualize_word_distribution(df):
    fig = px.histogram(x=[len(nltk.wordtokenize(x)) for x in df.clean_joined], nbins=100)
    fig.show()

def visualize_confusion_matrix(prediction, y_test):
    '''
    :param prediction:
    :param y_test:
    :return:
    '''
    cm = confusion_matrix(list(y_test), prediction)
    plt.figure(figsize=(10,10))
    sns.heatmap(cm,annot=True)
    plt.show()


#plot two bar histogram
def plotSingleHistogram(first_bar_data, second_bar_data, xlable, ylable, title):
    '''
    generic function to generate two bar histogram chart
    :param first_bar_data: the value of the first bar
    :param second_bar_data: the value of the second bar
    :param xlable: histogram x lable name
    :param ylable: histogram y lable name
    :param title: histogram title
    :return:
    '''
    bars = plt.bar([1,2],[len(first_bar_data), len(second_bar_data)])
    bars[1].set_color('green')
    lablelist = [0, 'True', 'False']
    tickvalues = range(0, len(lablelist))
    plt.xticks(ticks = tickvalues ,labels = lablelist, rotation='horizontal')
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.title(title)
    plt.show()

def visualize_composition(df):
    """
    Visualize the composition of the input dataset
    :param fake_news: fake news dataframe
    :param real_news: real news dataframe
    :return: None
    """
    # separate real from fake
    fake_news = df[df['is_fake'] == 1]
    real_news = df[df['is_fake'] == 0]

    # Categories of news
    # business      = 1
    # education     = 2
    # entertainment = 3
    # politics      = 4
    # sport         = 5
    # tech          = 6
    # celebrity     = 7
    labels = ['Business', 'Education', 'Entertainment', 'Politics', 'Sport', 'Technology', 'Celebrity']

    # Get the counts of the categories in reversed order
    fake_news_counts = fake_news['Category'].value_counts().tolist()
    real_news_counts = real_news['Category'].value_counts().tolist()
    # reverse the order
    fake_news_counts.reverse()
    real_news_counts.reverse()

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width / 2, fake_news_counts, width, label='Fake')
    rects2 = ax.bar(x + width / 2, real_news_counts, width, label='Real')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Count')
    ax.set_title('Composition of the Fake News Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.xticks(rotation=90)
    plt.show()



