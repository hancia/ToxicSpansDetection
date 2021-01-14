import csv

from utils import get_preds_from_experiment, get_api_and_experiment, fill_holes_in_row

if __name__ == '__main__':
    comet_api, experiment = get_api_and_experiment('ab0ed37bc24543888ca59017c30443cb')
    df = get_preds_from_experiment(experiment)
    df['spans'] = df['spans'].apply(fill_holes_in_row)
    df.to_csv('filled/spans-pred.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')
