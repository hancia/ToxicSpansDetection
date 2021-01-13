from ast import literal_eval
import pandas as pd
df = pd.read_csv('data/spans/tsd_trial_128.csv')
df.loc[:, 'spans'] = df['spans'].apply(literal_eval)

df['spans2'] = df['spans'].apply(lambda x:len(x))
print(df.head())
print(df['spans2'].max())