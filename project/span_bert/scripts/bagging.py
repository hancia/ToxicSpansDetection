import csv
from ast import literal_eval
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import get_preds_from_experiment, get_api_and_experiment, fill_holes_in_row, remove_ones_in_row


def get_dfs(exps):
    result = list()
    for exp in exps:
        comet_api, experiment = get_api_and_experiment(exp)
        df = get_preds_from_experiment(experiment)
        result.append(df)
    return result


if __name__ == '__main__':
    dfs = get_dfs(['c0a73a0394364aadb4cf7e7fd9041bcd', 'ab0ed37bc24543888ca59017c30443cb'])
    result_df = pd.DataFrame({'spans': pd.Series(np.zeros(len(dfs[0]))).values})

    for i in tqdm(range(len(dfs[0]))):
        pred_dict = defaultdict(int)
        for df in dfs:
            spans = literal_eval(df.loc[i, 'spans'])
            for k in spans:
                pred_dict[k] += 1
        common_preds = sorted(list(filter(lambda x: pred_dict[x] > len(dfs) / 2, pred_dict)))
        result_df.loc[i, 'spans'] = str(common_preds)

    result_df.to_csv('filled/spans-pred-bagging.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')

    result_df['spans'] = result_df['spans'].apply(fill_holes_in_row)
    result_df.to_csv('filled/spans-pred-bagging-filled.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE,
                     escapechar='\n')

    result_df['spans'] = result_df['spans'].apply(remove_ones_in_row)
    result_df.to_csv('filled/spans-pred-bagging-filled-removed.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE,
                     escapechar='\n')
