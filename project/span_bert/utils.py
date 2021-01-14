from ast import literal_eval
from configparser import ConfigParser
from io import StringIO

import comet_ml
import pandas as pd
from easydict import EasyDict


def f1_semeval(pred_spans, true_spans):
    """
    F1 (a.k.a. DICE) operating on two lists of offsets (e.g., character).
    >>> assert f1([0, 1, 4, 5], [0, 1, 6]) == 0.5714285714285714
    :param predictions: a list of predicted offsets
    :param gold: a list of offsets serving as the ground truth
    :return: a score between 0 and 1
    """
    if len(true_spans) == 0:
        return 1 if len(pred_spans) == 0 else 0
    nom = 2 * len(set(pred_spans).intersection(set(true_spans)))
    denom = len(set(pred_spans)) + len(set(true_spans))
    return nom / denom


def get_preds_from_experiment(experiment):
    assets_list = experiment.get_asset_list()
    spans_asset = list(filter(lambda x: x['fileName'] == 'spans-pred.txt', assets_list))[0]
    span_id = spans_asset['assetId']

    binary_file = experiment.get_asset(span_id, return_type='text')
    df = pd.read_table(StringIO(binary_file), sep="\t", header=None, names=['id', 'spans'], index_col='id')
    df['spans'] = df['spans'].apply(literal_eval)
    return df


def get_api_and_experiment(experiment_id):
    config = ConfigParser()
    config.read('config.ini')

    comet_config = EasyDict(config['cometml'])
    comet_api = comet_ml.api.API(api_key=comet_config.apikey)
    experiment = comet_api.get(project_name=comet_config.projectname, workspace=comet_config.workspace,
                               experiment=experiment_id)
    return comet_api, experiment


def fill_holes_in_row(spans):
    sorted_spans = sorted(spans)
    new_spans = []
    if spans:
        for i in range(len(sorted_spans) - 1):
            new_spans.append(sorted_spans[i])
            if sorted_spans[i + 1] - sorted_spans[i] == 2:
                new_spans.append(sorted_spans[i] + 1)
        new_spans.append(sorted_spans[-1])
    return new_spans
