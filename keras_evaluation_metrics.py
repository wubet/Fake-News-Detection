import keras.backend as K


def recall_m(y_true, y_pred):
    """
    Calculate recall
    :param y_true: labels
    :param y_pred: predictions
    :return: float recall
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    """
    Compute precision
    :param y_true: labels
    :param y_pred: predictions
    :return: float precision
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    """
    Compute F1-score
    :param y_true: labels
    :param y_pred: predictions
    :return: float f1-score
    """
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
