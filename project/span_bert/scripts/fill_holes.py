import csv

from utils import get_preds_from_experiment, get_api_and_experiment, fill_holes_in_row

if __name__ == '__main__':
    comet_api, experiment = get_api_and_experiment('c0a73a0394364aadb4cf7e7fd9041bcd')
    df = get_preds_from_experiment(experiment)
    df['spans'] = df['spans'].apply(fill_holes_in_row)
    df.to_csv('filled/spans-pred.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')
