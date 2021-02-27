import csv

import numpy as np
import pandas as pd
from tqdm import tqdm

from Transparency.calc_threshold_utils import preds_to_spans
from Transparency.OrthoModel import OrthoModel

trial = pd.read_csv("../../data/spans/tsd_test.csv")
trial = trial.assign(spans=pd.Series(np.zeros(len(trial))).values)

model = OrthoModel()
for i, row in tqdm(trial.iterrows(), total=len(trial)):
    preds = model.predict(row['text'])
    predicted_spans = preds_to_spans(preds, threshold=0.5, cumulative=True)
    trial.loc[i, 'spans'] = str(predicted_spans)

trial = trial.drop(columns=['text'])
trial.to_csv('spans-pred.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')
