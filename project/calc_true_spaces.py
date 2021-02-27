import string
from ast import literal_eval

import pandas as pd
from tqdm import tqdm

df = pd.read_csv('data/spans/tsd_train.csv')
df['spans'] = df['spans'].apply(literal_eval)

all_spans, spaces, interpunction = 0, 0, 0
for idx, row in df.iterrows():
    text, spans = row['text'], row['spans']
    all_spans += len(spans)
    for span in spans:
        if text[span] == ' ':
            spaces += 1
        elif text[span] in string.punctuation:
            interpunction+=1


print(f'{spaces/all_spans:.2%} spaces in spans')
print(f'{interpunction/all_spans:.2%} interpunction in spans')