import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
import gensim
import pandas as pd
import numpy as np
from tabulate import tabulate

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)
import re
from keras.layers import Embedding
import os


stop_words = []

def process_feature_engineering(df):
    '''

    :param df: dataset
    :return: dataset with column name clean that contain non stop words
    '''
    df_clean, stop_words = remove_stop_words(df)
    total_words = find_total_words(df_clean)
    maxlen = find_max_token_length(df_clean)
    return df_clean, stop_words, total_words, maxlen


def combine_two_columns(df, first_column, second_column, new_column):
    '''
    :param df: dataframe
    :param first_column: first column to be combined
    :param second_column: seconed column to be combined
    :param new_column: the new column to be named
    :return: copy of the new data fram
    '''
    df_original = df.copy()
    df_original[new_column] = df_original[first_column].astype(str) + ' ' + df[second_column]

    return df_original

def preprocess_stop_word(text):
    '''
    convert a document into a list of tokens
    :param text: text to be processed
    :param stop_words: list of stop words
    :return: return the processed text
    '''
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 2 and token not in stop_words:
            result.append(token)
    return result


def remove_stop_words(df):
    '''
    remove stop words from 'clean_joined' feature of the datafram
    :param df:dataframe
    :return:new datafream and stop_words
    '''
    df_original = combine_two_columns(df, 'Title', 'Content', 'original')
    stop_words = stopwords.words('english')
    df_original['clean'] = df_original['original'].apply(preprocess_stop_word)
    df_original['clean_joined'] = df_original['clean'].apply(lambda x: " ".join(x))
    print(tabulate(df_original.head(5), headers='keys', tablefmt='psql'))
    print(df_original.shape)
    return df_original, stop_words


def find_total_words(df):
    '''
    find the total list of words excluding stop words and redundent words
    :param df: dataframe
    :return: list of words
    '''
    list_of_words = []
    for i in df.clean:
        for j in i:
            list_of_words.append(j)
    total_words = len(list(set(list_of_words)))
    return total_words


def find_max_token_length(df):
    '''
    # find the max lengeth of a token
    :param df:
    :return:
    '''
    maxlen = -1
    for doc in df.clean_joined:
        tokens = nltk.wordpunct_tokenize(doc)
        if(maxlen < len(tokens)):
            maxlen = len(tokens)
    return  maxlen


def tfidf_transform(raw_data, tfidf_vectorizer=None):
    """
    Helper function to convert raw data of text into tf-idf matrix
    :param raw_data: raw text data
    :param tfidf_vectorizer: tfidf vectorizer from Scikit-Learn
    :return: tf-idf matrix and reference to the tf-idf vectorizer used
    """
    # tf-idf transformer
    if tfidf_vectorizer is None:
        tfidf_vectorizer = TfidfVectorizer(lowercase=True, smooth_idf=True)
        mat = tfidf_vectorizer.fit_transform(raw_data).todense()
    else:
        mat = tfidf_vectorizer.transform(raw_data).todense()

    return mat, tfidf_vectorizer


def vectorize_ngrams(raw_data, cv_ngram=None):
    """
    Helper function to convert raw data of text into matrix of ngram counts
    :param raw_data: raw text data
    :param cv_ngram: Scikit-Learn CountVectorizer
    :return: ngram count matrix and the CountVectorizer used
    """
    if cv_ngram is None:
        # count vectorizer
        # convert all words to lower case letters
        cv_ngram = CountVectorizer(analyzer='word', ngram_range=(3, 3), lowercase=True)
        # convert the input text data to a matrix of token counts
        mat = cv_ngram.fit_transform(raw_data).todense()
    else:
        mat = cv_ngram.transform(raw_data).todense()

    return mat, cv_ngram


def extract_features(X):
    """
    Extract features from news titles and contents
    :param df: two-column matrix of features (Title and Content)
    :return: feature matrix and feature extracting transformers
    """
    # Convert the titles to Tf-iDF matrix
    mat_title, tfidf_title = tfidf_transform(X[:, 0])

    # Convert the contents to Tf-iDF matrix
    mat_content, tfidf_content = tfidf_transform(X[:, 1])

    # count ngrams in the contents
    mat_ngram, cv_ngram = vectorize_ngrams(X[:, 1])

    X_mat = np.hstack((mat_title, mat_content))

    print("The size of the feature space is:", X_mat.shape)

    return {
        "cv_ngram": cv_ngram,
        "tfidf_content": tfidf_content,
        "tfidf_title": tfidf_title,
        "features": X_mat
    }


def tokenize_words(raw_data, max_length: int, tokenizer=None):
    """
    Tokenize words
    :param raw_data:    input list of texts
    :param max_length:  maximum length of an input sequence
    :param tokenizer:   a trained tokenizer. Create a new one if none
    :return:            list of tokenized input texts and trained tokenizer
    """
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    if tokenizer is None:
        tokenizer = Tokenizer(oov_token=oov_tok)
        tokenizer.fit_on_texts(raw_data)

    # pad the sequence
    sequences = tokenizer.texts_to_sequences(raw_data)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    return padded, tokenizer


def normalize(data):
    normalized = []
    for i in data:
        i = i.lower()
        # get rid of urls
        i = re.sub('https?://\S+|www\.\S+', '', i)
        # get rid of non words and extra spaces
        i = re.sub('\\W', ' ', i)
        i = re.sub('\n', '', i)
        i = re.sub(' +', ' ', i)
        i = re.sub('^ ', '', i)
        i = re.sub(' $', '', i)
        normalized.append(i)
    return normalized

#pre_trained word embedding for LSTM
def get_lstm_grove_embedding(total_words, embed_size):

    # load the whole embedding into memory
    embeddings_index = dict()
    f = open('glove_data/glove.twitter.27B.100d.txt', encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # create a weight matrix for words in training docs
    embedding_matrix = np.zeros((total_words, embed_size))
    tokenizer = Tokenizer(num_words=total_words)
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

# pre-trained word embeddings for cnn
def get_cnn_grove_embedding(total_words, embedding_dim, max_sequence_length):
    GLOVE_DIR = "glove_data"
    embeddings_index = {}
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'), encoding="utf8")
    for line in f:
        values = line.split()
        # print(values[1:])
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    print('Total %s word vectors in Glove.' % len(embeddings_index))
    tokenizer = Tokenizer(num_words=total_words)
    word_index = tokenizer.word_index
    embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        print("data type")
        print(type(embedding_vector))
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                                embedding_dim,
                                weights=[embedding_matrix],
                                input_length=max_sequence_length)


    return embedding_matrix, embedding_layer