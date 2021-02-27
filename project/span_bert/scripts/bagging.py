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
    dfs = get_dfs([
        'a825abe773b344748d4bb3f9d813a7ef',
        '0919538d5186438bb7c7ce58e1a2eb9d',
        '16afa12678614d9f9add964189b3b27f',
        '9a817d7bc84d4f39a75553534d0c4062',
        'e94ed14aeb7647578a667aa855c2295a'
    ])
    print(len(dfs))
    result_df = pd.DataFrame({'spans': pd.Series(np.zeros(len(dfs[0]))).values})

    for i in tqdm(range(len(dfs[0]))):
        pred_dict = defaultdict(int)
        for df in dfs:
            spans = literal_eval(df.loc[i, 'spans'])
            for k in spans:
                pred_dict[k] += 1
        common_preds = sorted(list(filter(lambda x: pred_dict[x] > len(dfs) / 2, pred_dict)))
        result_df.loc[i, 'spans'] = str(common_preds)

    result_df.to_csv('spans-pred-bagging.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')

    # result_df['spans'] = result_df['spans'].apply(fill_holes_in_row)
    # result_df.to_csv('filled/spans-pred-bagging-filled.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE,
    #                  escapechar='\n')
    #
    # result_df['spans'] = result_df['spans'].apply(remove_ones_in_row)
    # result_df.to_csv('filled/spans-pred-bagging-filled-removed.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE,
    #                  escapechar='\n')
