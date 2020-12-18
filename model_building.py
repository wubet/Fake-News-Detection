import io
import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    roc_auc_score
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import tensorflow as tf
from keras.utils import plot_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, Dense, LSTM, Dropout, GRU
from keras.optimizers import Adam
from keras.regularizers import l2
import pydot

from feature_engineering import (
    tfidf_transform,
    vectorize_ngrams,
    extract_features,
    tokenize_words,
    process_feature_engineering,
    normalize,
    get_lstm_grove_embedding,
    get_cnn_grove_embedding
)

from data_visualization import(
    visualize_confusion_matrix
)

from data_compilation import clean_text

from keras_evaluation_metrics import (
    precision_m,
    recall_m,
    f1_m
)
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


def evaluate(fit, X_test, Y_test, is_dl: bool = False):
    """
    Evaluate a trained model for accuracy, precision, recall, f_score, and auc
    :param fit: trained model
    :param X_test: test features
    :param Y_test: test labels
    :param is_dl: whether the input model is deep learning
    :return: a dictionary of metrics
    """
    predictions = np.array(fit.predict(X_test))

    # deal with deep learning model
    pred = []
    if is_dl:
        for i in range(len(predictions)):
            # cutoff threshold is 0.5
            if predictions[i] > 0.5:
                pred.append(1)
            else:
                pred.append(0)
        predictions_proba = predictions
        predictions = np.array(pred)
    else:
        # deal with classical model
        predictions_proba = np.array(fit.predict_proba(X_test))
        # select the predictions for positive label
        predictions_proba = predictions_proba[:, 1]

    return {
        "predictions": predictions,
        "probability": predictions_proba,
        "accuracy": accuracy_score(Y_test, predictions),
        "precision": precision_score(Y_test, predictions),
        "recall": recall_score(Y_test, predictions),
        "f_score": f1_score(Y_test, predictions),
        "auc": roc_auc_score(np.array(Y_test), predictions_proba)
    }


def cross_validate(model, features, labels):
    """
    Perform k fold cross validation
    :param model: untrained model
    :param features: feature matrix
    :param labels: label vector
    :return: a data frame of cross validation metrics
    """
    # Stratified 10-fold
    k = 10
    kfold = StratifiedKFold(n_splits=k, shuffle=True)

    # validation metrics
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f_score = 0.0
    auc = 0.0

    for train_indices, test_indices in kfold.split(features, labels):
        X_train = features[train_indices, :]
        Y_train = labels[train_indices]

        X_test = features[test_indices, :]
        Y_test = labels[test_indices]

        # train the model
        fit = model.fit(X_train, Y_train)

        # evaluate the model
        metrics = evaluate(fit, X_test, Y_test)

        accuracy += metrics['accuracy']
        precision += metrics['precision']
        recall += metrics['recall']
        f_score += metrics['f_score']
        auc += metrics['auc']

    # average and pack the results into a data frame
    values = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"],
        "Value": [accuracy / k, precision / k, recall / k, f_score / k, auc / k]
    }
    metrics_df = pd.DataFrame.from_dict(values)
    return metrics_df


def break_down_results_by_category(model_name, indices, Y, predictions, category_sizes):
    """
    Break down the results for different categories of news
    :param model_name: name of the model of interest
    :param indices: a dictionary of indices of the categories in the test set
    :param Y: labels of the test set
    :param predictions: list of predictions on the test set [predictions, predictive probability]
    :param category_sizes: list of the sizes of the categories. Index - 1 is the id of a category
    :return:
    """
    pred = predictions[0]
    pred_proba = predictions[1]

    results = []
    dataframe_indices = []

    for category, ind in indices.items():
        pred_by_category = [pred[i] for i in ind]
        pred_proba_by_category = [pred_proba[i] for i in ind]
        labels = [Y[i] for i in ind]

        results.append([
            accuracy_score(labels, pred_by_category),
            precision_score(labels, pred_by_category),
            recall_score(labels, pred_by_category),
            f1_score(labels, pred_by_category),
            roc_auc_score(labels, pred_proba_by_category)
        ])

        dataframe_indices.append("Class " + str(category) + "(size = " + str(category_sizes[int(category) - 1]) + ")")

    multi_col = pd.MultiIndex.from_tuples([
        (model_name, 'Accuracy'), (model_name, 'Precision'), (model_name, 'Recall'),
        (model_name, 'F1-Score'), (model_name, 'AUC')
    ])

    results_by_category = pd.DataFrame(results, index=dataframe_indices, columns=multi_col)
    results_by_category = results_by_category.stack()

    return results_by_category


def display_results(validation_df = None, test_df = None, category_df = None):
    """
    Display the results to console
    :param validation_df: dataframe of validation results on training set
    :param test_df: dataframe of results on test set
    :param category_df: dataframe of results for individual news categories
    :return: None
    """
    if validation_df is not None:
        print("Cross validation results:")
        print(validation_df)
    if test_df is not None:
        print("Test set evaluation results:")
        print(test_df)
    if category_df is not None:
        print("Test set's results by news categories:")
        print(category_df)


def classical_models(df):
    """
    Build classical models and perform cross validation and evaluate the performance on test set
    :param df: raw data frame
    :return: a dictionary of the trained models and features extracting transformers
    """
    # extract data
    X = df[['Title', 'Content']].values
    Y = df['is_fake'].values
    labels = Y.astype('int')

    # extract the features from the data frame
    feature_pack = extract_features(X)
    features = feature_pack['features']

    # extract the category column in original dataset
    category_col = df['Category'].tolist()
    category_col = np.reshape(category_col, (-1, 1))

    # join the list of categories to the feature matrix
    features = np.hstack([features, category_col])

    # split the dataset 80% / 20% for train and test
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=0, stratify=labels)

    # convert to numpy array
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # extract the list of categories in the test set
    testset_categories = X_test[:, -1]
    category_sizes = np.bincount(testset_categories.astype('int'))[1:]

    # build a dictionary of the indices of different categories in the test set
    indices = {}
    for category in range(1, 8):
        indices[category] = [i for i, x in enumerate(testset_categories) if int(x) == category]

    # model
    models = {
        "Logistic Regression": LogisticRegression(n_jobs=8, solver='lbfgs', C=10),
        "Gaussian NB": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(splitter='best'),
        "Random Forest": RandomForestClassifier(n_estimators=300),
        "XGBoost": XGBClassifier(n_jobs=8, learning_rate=0.1, max_depth=10, n_estimators=200),
        "SVM": SVC(gamma='auto', kernel='poly', probability=True),
    }

    # create a data frame to store validation metrics and test metrics
    metrics = {"Metrics": ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']}
    validation_metrics_df = pd.DataFrame.from_dict(metrics)
    test_metrics_df = pd.DataFrame.from_dict(metrics)

    results_by_category = []

    for model_name, model in models.items():
        print("Working on", model_name)
        # k-fold cross validation
        metrics_df = cross_validate(model, X_train[:, 0:X_train.shape[1]-1], Y_train)
        validation_metrics_df[model_name] = metrics_df["Value"]

        # train the model
        # don't use the category information for training
        fit = model.fit(X_train[:, 0:X_train.shape[1]-1], Y_train)

        # evaluate the model on the test set
        test_results = evaluate(fit, X_test[:, 0:X_test.shape[1]-1], Y_test)
        # pack the results into a data frame
        values = {
            "Value": [test_results['accuracy'],
                      test_results['precision'],
                      test_results['recall'],
                      test_results['f_score'],
                      test_results['auc']]
        }
        test_metrics = pd.DataFrame.from_dict(values)
        test_metrics_df[model_name] = test_metrics["Value"]

        # break down the results for different categories of news
        results_by_category.append(
            break_down_results_by_category(model_name, indices, Y_test,
                                           [test_results['predictions'], test_results['probability']],
                                           category_sizes
                                           )
        )

        # save the model for later use
        filename = 'output/' + model_name.replace(' ', '') + '_Model' + '.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)

    # join results by categories for different models to one dataframe
    results_by_category_df = results_by_category[0]
    for i in range(1, len(results_by_category)):
        results_by_category_df = results_by_category_df.join(results_by_category[i])
    results_by_category_df.to_csv('output/classical_results/none_dl_results_by_category.csv')

    # display the results
    display_results(validation_metrics_df, test_metrics_df, results_by_category_df)

    # Write the results to .csv files
    validation_metrics_df.to_csv('output/classical_results/validation_results.csv')
    test_metrics_df.to_csv('output/classical_results/test_results.csv')

    # save the feature transformers for later use
    with open("output/none_dl_input_transformers.pkl", 'wb') as file:
        pickle.dump((feature_pack['cv_ngram'], feature_pack['tfidf_content'], feature_pack['tfidf_title']), file)

    return {
        "cv_ngram": feature_pack['cv_ngram'],
        "tfidf_content": feature_pack['tfidf_content'],
        "tfidf_title": feature_pack['tfidf_title']
    }


def visualize_dl_training(history, model_name):
    """
    Draw plot of the training history for a deep learning model
    :param history: history object of the training
    :param model_name: name of the deep learning model
    :return: none
    """
    plt.figure()
    # visualize the training history
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(model_name + " Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Test"], loc='upper left')
    plt.savefig('output/dl_results/' + model_name + '_accuracy_hist.png')
    plt.show()

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(model_name + " Model Loss")
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('output/dl_results/' + model_name + '_loss_hist.png')
    plt.show()


def visualize_trained_word_embedding(model, trained_tokenizer, vocab_size, model_name):
    """
    Save files for viewing with Tensorflow Projector
    Go to https://projector.tensorflow.org and load the .tsv file
    :param model: the trained model
    :param trained_tokenizer: trained word tokenizer
    :param vocab_size: size of the vocabulary
    :param model_name: name of the deep learning model
    :return: None
    """
    # get the dictionary of words and frequencies in the corpus
    word_index = trained_tokenizer.word_index
    # reverse the key-value relationship in word_index
    reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

    # get the weights for the embedding layer
    e = model.layers[0]
    weights = e.get_weights()[0]

    # write out the embedding vectors and metadata
    # To view the visualization, go to https://projector.tensorflow.org
    out_v = io.open('output/dl_results/' + model_name + 'content_vectors.tsv', 'w', encoding='utf-8')
    out_m = io.open('output/dl_results/' + model_name + 'content_meta.tsv', 'w', encoding='utf-8')
    for word_num in range(1, vocab_size):
        word = reverse_word_index[word_num]
        embeddings = weights[word_num]
        out_m.write(word + '\n')
        out_v.write('\t'.join([str(x) for x in embeddings]) + '\n')
    # close files
    out_m.close()
    out_v.close()


def build_nn_model(vocab_size, embedding_dim, max_length):
    """
    Construct a neural network with word embedding
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_lstm_model2(vocab_size, embedding_dim, max_length):
    """
    Construct a LSTM model with word embedding
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv1D(64, 5, activation='relu'),
        tf.keras.layers.MaxPool1D(pool_size=4),
        tf.keras.layers.LSTM(20, return_sequences=True),
        tf.keras.layers.LSTM(10),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_gru_model(vocab_size, embedding_dim, max_length):
    """
    Construct a GRU model with word embedding
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.GRU(units=64, dropout=0.2, recurrent_dropout=0.2,
                            recurrent_activation='sigmoid', activation='tanh'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(rate=0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(rate=0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_bidirectional_lstm_model(vocab_size, embedding_dim, max_length):
    """
    Construct a bidirectional LSTM model with word embedding
    Citation: Umer M., et al, "Fake News Stance Detection Using Deep Learning Architecture (CNN-LSTM)"
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def build_combined_cnn_lstm_model(vocab_size, embedding_dim, max_length):
    """
    Construct a combined CNN-LSTM model with word embedding
    :param vocab_size: size of the vocabulary
    :param embedding_dim: embedding dimension
    :param max_length: max length of the input sequences
    :return: a keras model
    """
    return tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5, strides=1, padding='valid', activation='relu'),
        tf.keras.layers.MaxPool1D(4),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(100),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])


def deep_learning_model(df):
    """
    Build a deep learning model
    :param df: input data frame containing raw data
    :return: trained neural network
    """
    embedding_dim = 32
    max_length = 200 #np.max([len(news) for news in df['Content'].tolist()])

    # extract data
    X = df[['Title', 'Content']].values
    Y = df['is_fake'].values
    labels = Y.astype('int')

    # tokenize the words
    features, trained_tokenizer = tokenize_words(raw_data=X[:, 1], max_length=max_length)

    # extract the category column in original dataset
    category_col = df['Category'].tolist()
    category_col = np.reshape(category_col, (-1, 1))

    # join the list of categories to the feature matrix
    features = np.hstack([features, category_col])

    # get the size of the vocabulary from the tokenizer
    vocab_size = len(trained_tokenizer.word_index)

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=0, stratify=labels)

    # convert to numpy array
    X_train = np.array(X_train)
    X_test = np.array(X_test)

    # extract the list of categories in the test set
    testset_categories = X_test[:, -1]
    category_sizes = np.bincount(testset_categories.astype('int'))[1:]

    # build a dictionary of the indices of different categories in the test set
    indices = {}
    for category in range(1, 8):
        indices[category] = [i for i, x in enumerate(testset_categories) if int(x) == category]

    # construct models
    models = {
        'Neural_Network': build_nn_model(vocab_size, embedding_dim, max_length),
        'LSTM': build_lstm_model2(vocab_size, embedding_dim, max_length),
        'GRU': build_gru_model(vocab_size, embedding_dim, max_length),
        'Bidirectional_LSTM': build_bidirectional_lstm_model(vocab_size, embedding_dim, max_length),
        'Combined_CNN_LSTM': build_combined_cnn_lstm_model(vocab_size, embedding_dim, max_length)
    }

    # create a data frame to store test metrics
    metrics = {"Metrics": ['Epoch', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']}
    test_metrics_df = pd.DataFrame.from_dict(metrics)
    evaluation_metrics_df = pd.DataFrame.from_dict(metrics)

    # create training callbacks
    early_stop_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=0.001)

    # define the number of training epochs
    num_epoch = 1000

    # list model results broken down for individual news categories
    results_by_category = []

    # train and evaluate the models
    for model_name, model in models.items():
        # announce current progress
        print("Working on", model_name)

        # save the model architecture to png file
        architecture_fname = 'output/dl_results/' + model_name + "architecture.png"
        plot_model(model, to_file=architecture_fname)

        # compile the model
        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision_m, recall_m, f1_m])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # train the model
        history = model.fit(X_train[:, 0:X_train.shape[1] - 1], Y_train,
                            epochs=num_epoch,
                            validation_data=(X_test[:, 0:X_test.shape[1] - 1], Y_test),
                            callbacks=[early_stop_cb, reduce_lr_cb]
                            )

        # save the models for later use
        cwd = os.getcwd()
        filename = os.path.join(cwd, 'output/' + model_name.replace(' ', '') + '.h5')
        model.save(filename)

        # evaluate the model on the test set
        test_results = evaluate(model, X_test[:, 0:X_test.shape[1] - 1], Y_test, True)
        # pack the results into dataframe
        values = {
            "Value": [early_stop_cb.stopped_epoch,
                      test_results['accuracy'],
                      test_results['precision'],
                      test_results['recall'],
                      test_results['f_score'],
                      test_results['auc']]
        }
        test_metrics = pd.DataFrame.from_dict(values)
        test_metrics_df[model_name] = test_metrics["Value"]

        # evaluate the model on the  set
        evaluation_results = evaluate(model, X_train[:, 0:X_train.shape[1] - 1], Y_train, True)
        # pack the results into dataframe
        values = {
            "Value": [early_stop_cb.stopped_epoch,
                      evaluation_results['accuracy'],
                      evaluation_results['precision'],
                      evaluation_results['recall'],
                      evaluation_results['f_score'],
                      evaluation_results['auc']]
        }
        evaluation_metrics = pd.DataFrame.from_dict(values)
        evaluation_metrics_df[model_name] = evaluation_metrics["Value"]

        # break down results for individual news categories
        results_by_category.append(
            break_down_results_by_category(model_name, indices, Y_test,
                                           [test_results['predictions'], test_results['probability']],
                                           category_sizes
                                           )
        )

        # visualize the training history
        visualize_dl_training(history, model_name)

        # visualize the word embedding
        visualize_trained_word_embedding(model, trained_tokenizer, vocab_size, model_name)

    # join results by categories for different models to one dataframe
    results_by_category_df = results_by_category[0]
    for i in range(1, len(results_by_category)):
        results_by_category_df = results_by_category_df.join(results_by_category[i])

    # save results to file
    results_by_category_df.to_csv('output/dl_results/dl_results_by_category.csv')
    test_metrics_df.to_csv("output/dl_results/dl_test_results.csv")
    evaluation_metrics_df.to_csv("output/dl_results/dl_evaluation_results.csv")

    # display results
    display_results(None, test_metrics_df, results_by_category_df)

    # save the tokenizer for later use
    with open("output/trained_tokenizer.pkl", 'wb') as outfile:
        pickle.dump(trained_tokenizer, outfile)
    # save the embedding dimensions for later use
    with open("output/embedding_dims.txt", 'w') as outfile:
        outfile.writelines(str(vocab_size) + " " + str(embedding_dim) + " " + str(max_length))

    return {
        "tokenizer": trained_tokenizer,
        "vocab_size": vocab_size,
        "embedding_dim": embedding_dim,
        "max_length": max_length
    }


def make_prediction(fit, input_transformers, file_path: str, is_dl: bool):
    """
    Make prediction for a single news file
    :param fit: trained model
    :param input_transformers: a dictionary contained feature extracting transformers
    :param file_path: full file system path to the .txt file containing the news
    :param is_dl: whether the model is deep learning related
    :return: none
    """
    print("Make prediction for", file_path)

    title = []
    content = []

    # open file and get data
    with open(file_path) as file:
        # read and store the title
        title.append(file.readline())

        # read and store the content
        content_lines = file.read().splitlines()
        content.append(" ".join(content_lines))

    # pack the data into a data frame
    sample = pd.DataFrame.from_dict({'Title': title, 'Content': content})

    # clean text
    sample = clean_text(sample, "Content", "Content")
    sample = clean_text(sample, "Title", "Title")

    if is_dl:
        # deep learning model
        trained_tokenizer = input_transformers['tokenizer']
        vocab_size = input_transformers['vocab_size']
        max_length = input_transformers['max_length']
        processed_sample, _ = tokenize_words(sample['Content'], max_length, trained_tokenizer)
    else:
        # classical model
        cv_ngram = input_transformers['cv_ngram']
        tfidf_title = input_transformers['tfidf_title']
        tfidf_content = input_transformers['tfidf_content']

        # extract features
        mat_title, _ = tfidf_transform(raw_data=sample['Title'], tfidf_vectorizer=tfidf_title)
        mat_content, _ = tfidf_transform(raw_data=sample['Content'], tfidf_vectorizer=tfidf_content)
        mat_ngram, _ = vectorize_ngrams(raw_data=sample['Content'], cv_ngram=cv_ngram)
        processed_sample = np.hstack((mat_title, mat_content))

    # make prediction
    pred = fit.predict(processed_sample)

    # Cutoff threshold is 0.5
    if pred[0] > 0.5:
        print("This is fake news. Consume with caution.")
    else:
        print("This is legitimate news. Read away.")


def create_pad_sequence(df, total_words, maxlen):
    MAX_SEQUENCE_LENGTH = 1000
    MAX_VOCAB = 10000
    leng = max(df.clean_joined.apply(lambda x: len(x.split())))
    print("Length: ")
    print(leng)
    print(maxlen)
    x_train, x_test, y_train, y_test = train_test_split(df.clean_joined, df.is_fake, test_size=0.2)
    tokenizer = Tokenizer(num_words=total_words)

    # update internal vocabulary based on a list of tests
    tokenizer.fit_on_texts(x_train)

    # transformation each text into a sequences integer
    train_sequences = tokenizer.texts_to_sequences(x_train)
    test_sequences = tokenizer.texts_to_sequences(x_test)
    padded_train = pad_sequences(train_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    padded_test = pad_sequences(test_sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    return padded_train, padded_test, y_train, y_test


def build_lstm_model(padded_train, padded_test, total_words, y_train, y_test):
    # dictionary for storing weights
    trained_models = dict()
    epochs = 50
    sent_leng = 1000
    batch_size= 100
    embed_size = 128
    model = Sequential()

    # embedding layer
    model.add(Embedding(total_words, output_dim=embed_size, input_length=sent_leng))
    embedding_matrix = get_lstm_grove_embedding(total_words, embed_size)
    # model.add(Embedding(total_words, output_dim=embed_size,weights=[embedding_matrix], input_length=sent_leng,
    #                    trainable=False))

    # Bi-directional RNN/LSTM
    # model.add(Bidirectional(LSTM(100)))
    model.add(Bidirectional(LSTM(32, kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01), bias_regularizer=l2(0.01))))

    # Dense layers
    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer= 'adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    # cp = ModelCheckpoint('model_Rnn.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
    y_train = np.asarray(y_train)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

    # history = model.fit(padded_train, y_train, batch_size=60, epochs=10, validation_split=0.2, shuffle=False, callbacks=[early_stop])
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=0.0000001, verbose=1)

    history = model.fit(padded_train, y_train, validation_data=(padded_test, y_test), epochs=epochs, batch_size=batch_size, shuffle=False, verbose=1, callbacks=[reduce_lr])
    trained_models['lstm'] = history.history

    # # Non-trainable embeddidng layer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience=2, verbose=1, factor=0.5,
                                                 min_lr=0.00001)

    history = model.fit(padded_train, y_train, batch_size=batch_size, validation_data=(padded_test, y_test), epochs=epochs,
                        callbacks=[learning_rate_reduction])

    # visualize the training history
    visualize_dl_training(history)

    return model


def predict_lstm_model(model, padded_test, y_test):
    # create a data frame to store validation metrics and test metrics
    metrics = {"Metrics": ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']}
    test_metrics_df = pd.DataFrame.from_dict(metrics)
    test_metrics_df = pd.DataFrame.from_dict(metrics)

    pred = model.predict(padded_test)
    prediction = []
    # if the predicted value is > 0.5 it is real else it is fake
    for i in range(len(pred)):
        if pred[i] > 0.5:
            prediction.append(1)
        else:
            prediction.append(0)

    # getting the measurement
    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(y_test, prediction)
    recall = recall_score(y_test, prediction)
    f1score = f1_score(y_test, prediction)
    auc = roc_auc_score(y_test, prediction)

    # pack the results into a data frame
    values = {
        "Value": [accuracy,
                  precision,
                  recall,
                  f1score,
                  auc]
    }
    test_metrics = pd.DataFrame.from_dict(values)
    test_metrics_df['lstm'] = test_metrics["Value"]

    # Write the results to .csv files
    test_metrics_df.to_csv('output/lstm_test_results.csv')

    print("LSTM Model Accuracy: ", accuracy)
    print("LSTM Model Precision: ", precision)
    print("LSTM Model Recall: ", recall)
    print("LSTM Model F1_score: ", f1score)
    print("LSTM Model AUC: ", auc)

    return prediction

def create_lstm_predictive_model(df):
    df_clean, stop_words, total_words, token_maxlen = process_feature_engineering(df)
    # visualize_fake_word_cloud_plot(df_clean, stop_words)
    # visualize_ligit_word_cloud_plot(df_clean, stop_words)
    padded_train, padded_test, y_train, y_test = create_pad_sequence(df_clean, total_words, token_maxlen)
    model = build_lstm_model(padded_train, padded_test, total_words, y_train, y_test)
    prediction = predict_lstm_model(model, padded_test, y_test)
    visualize_confusion_matrix(prediction, y_test)