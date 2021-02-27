import re

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from preprocess.vectorizer import cleaner


def cleaner_20(text):
    text = cleaner(text)
    text = re.sub(r'(\W)+', r' \1 ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


df = pd.read_csv('../../../../data/civil_comments/all_data.csv')

df_negative = df[df['toxicity'] >= 0.5]
df_positive = df[df['toxicity'] < 0.5]

print(len(df_negative))
print(len(df_positive))

df_positive = df_positive.sample(n=len(df_negative))

print(len(df_negative))
print(len(df_positive))

print(df_negative.head(5))
print(df_positive.head(5))

all_data_df = pd.concat([df_negative, df_positive])
base, test = train_test_split(all_data_df, test_size=0.1)
train, dev = train_test_split(all_data_df, test_size=0.11)

train.loc[:, 'exp_split'] = 'train'
test.loc[:, 'exp_split'] = 'test'
dev.loc[:, 'exp_split'] = 'dev'

print(len(train), len(dev), len(test))
final_df = pd.concat([train, test, dev])
data = []
for index, row in tqdm(final_df.iterrows(), total=final_df.shape[0]):
    text = cleaner_20(str(row['comment_text']))
    text = text.replace('\n', '')
    label = int(row['toxicity'] > 0.5)
    data.append([text, label, row['exp_split']])

result_df = pd.DataFrame(data, columns=['text', 'label', 'exp_split'])

print(result_df.head())

result_df.to_csv('civil_dataset.csv', index=False)
