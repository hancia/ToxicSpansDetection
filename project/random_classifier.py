from ast import literal_eval

import numpy as np
import pandas as pd

from Transparency.calc_threshold_utils import f1, one_hot_to_vec

if __name__ == '__main__':
    trial = pd.read_csv("../data/spans/tsd_train.csv")
    trial["spans"] = trial.spans.apply(literal_eval)
    f1_scores = list()
    for i, row in trial.iterrows():
        grand_truth_spans = row['spans']
        words = row['text'].split(' ')
        # vec = np.random.randint(2, size=len(words))
        vec = np.random.choice(2, len(words), p=[0.05, 0.95])
        predicted = list()
        for j in range(len(words)):
            for _ in range(len(words[j])):
                predicted.append(vec[j])
            predicted.append(0)
        predicted = one_hot_to_vec(predicted)
        score = f1(predicted, grand_truth_spans)
        f1_scores.append(score)
    f1 = np.mean(np.array(f1_scores))
    print(f1)
