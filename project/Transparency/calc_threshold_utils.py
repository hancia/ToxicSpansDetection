from ast import literal_eval
from operator import itemgetter

import numpy as np
import pandas as pd
from tqdm import tqdm

from predict_last_model import OrthoModel


def f1(predictions, gold):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(gold) == 0:
        return 1 if len(predictions) == 0 else 0
    nom = 2 * len(set(predictions).intersection(set(gold)))
    denom = len(set(predictions)) + len(set(gold))
    return nom / denom


def one_hot_to_vec(one_hot):
    return [i for i, x in enumerate(one_hot) if x == 1]


def preds_to_spans(preds, threshold=0.2, cumulative=False):
    prediction, attention, spans, list_of_tokens = preds
    pred_spans = []
    if cumulative:
        cumulated_attention = 0
        sorted_idx_attention = np.argsort(attention)[::-1]
        for idx in sorted_idx_attention:
            if spans[idx] is not None:
                i, j = spans[idx]
                pred_spans.extend(range(i, j))
                cumulated_attention += attention[idx]

            if cumulated_attention >= threshold:
                break
    else:
        for attn, span, word in zip(attention, spans, list_of_tokens):
            if attn > threshold and span is not None:
                i, j = span
                pred_spans.extend(range(i, j))

    return pred_spans


def calc_best_threshold_for_preds(preds, true_spans):
    size = len(preds[3])
    scores = []
    for i in np.linspace(0, 1, 100):
        predicted_spans = preds_to_spans(preds, threshold=i, cumulative=True)
        score = f1(predicted_spans, true_spans)
        scores.append((i, score))

    best_score = max(scores, key=itemgetter(1))[1]
    best_score_ids = np.array([threshold for threshold, score in scores if score == best_score])
    average_best_threshold = np.mean(best_score_ids)
    return average_best_threshold, size


if __name__ == '__main__':
    model = OrthoModel()
    trial = pd.read_csv("../../data/spans/tsd_train.csv")
    trial["spans"] = trial.spans.apply(literal_eval)
    data = list()

    for i, row in tqdm(trial.iterrows(), total=len(trial)):
        preds = model.predict(row['text'])
        best_score, length = calc_best_threshold_for_preds(preds, row['spans'])
        data.append([best_score, length])

    df = pd.DataFrame(data, columns=['threshold', 'tokens'])
    df.to_csv('threshold_cumulated.csv', index=False)
