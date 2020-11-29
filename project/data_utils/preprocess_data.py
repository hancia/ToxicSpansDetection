import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/all_data.csv')
df = df[['comment_text', 'toxicity']]

train, test = train_test_split(df, test_size=0.1)
train.to_csv('data/train.csv')
test.to_csv('data/test.csv')
