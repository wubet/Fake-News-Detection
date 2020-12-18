import tensorflow as tf
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from feature_engineering import (
    tfidf_transform,
    vectorize_ngrams,
    extract_features,
    tokenize_words
)

from tensorboard.plugins.hparams import api as hp  # for hyperparameter tuning


def none_dl_grid_search(df):
    """
    Using grid search to find the best model and hyperparameters
    :param df: raw data frame
    :return: None
    """
    # extract data
    X = df[['Title', 'Content']].values
    Y = df['is_fake'].values
    labels = Y.astype('int')

    # extract the features from the data frame
    feature_pack = extract_features(X)
    features = feature_pack['features']

    # create models and parameters
    model_params = {
        'Logistic Regression': {
            'model': LogisticRegression(n_jobs=8, solver='lbfgs'),
            'params': {
                'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'splitter': ['best', 'random']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'n_estimators': [50, 100, 200, 300, 400, 500]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(n_jobs=8),
            'params': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [5, 10, 15],
                'learning_rate': [0.1, 0.2, 0.3]
            }
        },
        'SVM': {
            'model': SVC(gamma='auto', kernel='poly', probability=True),
            'params': {
                'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                'gamma': ['auto', 'scale'],
                'kernel': ['rbf', 'poly', 'sigmoid']
            }
        }
    }

    # list of scores
    scores = []

    # iterate over the models
    for model_name, mp in model_params.items():
        print("Working on", model_name)
        clf = GridSearchCV(mp['model'], mp['params'], cv=5, return_train_score=False)
        clf.fit(features, labels)
        scores.append({
            'model': model_name,
            'best_params': clf.best_params_,
            'best_score': clf.best_score_
        })

    # display the results
    resultsDF = pd.DataFrame(scores)
    resultsDF.columns = ['Model', 'Best Params', 'Best Score']
    print(scores)
    print(resultsDF)
    # save the results
    resultsDF.to_csv('output/classical_results/none_dl_grid_search_results.csv')


# setup hyperparameter experiment
HP_EMBEDDING_DIM = hp.HParam('embedding dim', hp.Discrete([8, 16, 32, 64]))
HP_MAX_LENGTH = hp.HParam('max length', hp.Discrete([200, 300, 1000, 2000, 3000]))
HP_NUM_UNITS_1 = hp.HParam('num units 1', hp.Discrete([16, 20, 32, 64, 128, 256]))
HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd', 'RMSprop']))
HP_NUM_UNITS_2 = hp.HParam('num units 2', hp.Discrete([8, 10, 16, 20, 32]))
METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('output/dl_results/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_EMBEDDING_DIM, HP_NUM_UNITS_1, HP_DROPOUT, HP_NUM_UNITS_2, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')]
    )


def train_test_model(df, hparams):
    """
    Helper function for parameter tuning
    :param df: input data
    :param hparams: set of hyperparameter
    :return: evaluation metric
    """
    # extract data
    X = df[['Title', 'Content']].values
    Y = df['is_fake'].values
    labels = Y.astype('int')

    # tokenize the words
    features, trained_tokenizer = tokenize_words(raw_data=X[:, 1], max_length=hparams[HP_MAX_LENGTH])

    # get the size of the vocabulary from the tokenizer
    vocab_size = len(trained_tokenizer.word_index)

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42, stratify=labels)

    # the model to tune for
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size + 1, hparams[HP_EMBEDDING_DIM], input_length=hparams[HP_MAX_LENGTH]),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS_1], activation='relu'),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS_2], activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # compile the model
    model.compile(optimizer=hparams[HP_OPTIMIZER], loss='binary_crossentropy', metrics=['accuracy'])

    # train the model
    model.fit(X_train, Y_train, epochs=20)

    # evaluate
    _, accuracy = model.evaluate(X_test, Y_test)
    return accuracy


def run(run_dir, hparams, df):
    """
    Helper function for hyperparameter tuning
    :param run_dir: directory to store results in
    :param hparams: individual hyperparameter to execute on
    :param df: input data
    :return: None
    """
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        accuracy = train_test_model(df, hparams)
        tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)


def dl_grid_search(df):
    """
    Tune hyperparameters to select deep learning model
    :param df: input data frame containing raw data
    :return: None
    """
    session_num = 0

    for embedding_dim in HP_EMBEDDING_DIM.domain.values:
        for max_length in HP_MAX_LENGTH.domain.values:
            for num_units_1 in HP_NUM_UNITS_1.domain.values:
                for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
                    for num_units_2 in HP_NUM_UNITS_2.domain.values:
                        for optimizer in HP_OPTIMIZER.domain.values:
                            hparams = {
                                HP_EMBEDDING_DIM: embedding_dim,
                                HP_MAX_LENGTH: max_length,
                                HP_NUM_UNITS_1: num_units_1,
                                HP_DROPOUT: dropout_rate,
                                HP_NUM_UNITS_2: num_units_2,
                                HP_OPTIMIZER: optimizer,
                            }
                            run_name = "run-%d" % session_num
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            run('output/dl_results/hparam_tuning/' + run_name, hparams, df)
                            session_num += 1
