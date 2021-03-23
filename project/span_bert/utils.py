import csv
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
    spans_asset = list(filter(lambda x: x['fileName'] == 'spans-pred-filled.txt', assets_list))[0]
    span_id = spans_asset['assetId']

    binary_file = experiment.get_asset(span_id, return_type='text')
    df = pd.read_table(StringIO(binary_file), sep="\t", header=None, names=['id', 'spans'], index_col='id')
    return df


def get_api_and_experiment(experiment_id):
    config = ConfigParser()
    config.read('config.ini')

    comet_config = EasyDict(config['cometml'])
    comet_api = comet_ml.api.API(api_key=comet_config.apikey)
    experiment = comet_api.get(project_name=comet_config.projectname, workspace=comet_config.workspace,
                               experiment=experiment_id)
    return comet_api, experiment


def fill_holes_in_row(spans: str) -> str:
    sorted_spans = sorted(literal_eval(spans))
    new_spans = []
    if sorted_spans and len(sorted_spans) > 1:
        for i in range(len(sorted_spans) - 1):
            new_spans.append(sorted_spans[i])
            if sorted_spans[i + 1] - sorted_spans[i] == 2:
                new_spans.append(sorted_spans[i] + 1)
        new_spans.append(sorted_spans[-1])
    return str(new_spans)

def fill_holes_in_row_three(spans: str) -> str:
    sorted_spans = sorted(literal_eval(spans))
    new_spans = []
    if sorted_spans and len(sorted_spans) > 1:
        for i in range(len(sorted_spans) - 1):
            new_spans.append(sorted_spans[i])
            if sorted_spans[i + 1] - sorted_spans[i] == 2:
                new_spans.append(sorted_spans[i] + 1)
            elif sorted_spans[i + 1] - sorted_spans[i] == 3:
                new_spans.append(sorted_spans[i] + 1)
                new_spans.append(sorted_spans[i] + 2)
        new_spans.append(sorted_spans[-1])
    return str(new_spans)

def remove_ones_in_row(spans: str) -> str:
    sorted_spans = sorted(literal_eval(spans))
    new_spans = []
    if sorted_spans and len(sorted_spans) > 1:
        if sorted_spans[1] - sorted_spans[0] == 1:
            new_spans.append(sorted_spans[0])

        for i in range(1, len(sorted_spans) - 1):
            if sorted_spans[i + 1] - sorted_spans[i] == 1 or sorted_spans[i] - sorted_spans[i - 1] == 1:
                new_spans.append(sorted_spans[i])

        if sorted_spans[-1] - sorted_spans[-2] == 1:
            new_spans.append(sorted_spans[-1])
    return str(new_spans)


def log_predicted_spans(df, logger):
    df.to_csv('spans-pred.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')
    logger.experiment.log_asset('spans-pred.txt')

    df['spans'] = df['spans'].apply(fill_holes_in_row)
    df.to_csv('spans-pred-filled.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')
    logger.experiment.log_asset('spans-pred-filled.txt')

    df['spans'] = df['spans'].apply(remove_ones_in_row)
    df.to_csv('spans-pred-filled-removed.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')
    logger.experiment.log_asset('spans-pred-filled-removed.txt')
