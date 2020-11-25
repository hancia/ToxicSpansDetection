import pandas as pd
import plotly.figure_factory as ff

df = pd.read_csv('data/civil_comments/train.csv').head(10000)
df2 = df
df2['toxicity'] = (df['toxicity'] > 0.1) * 1
print(df2.head(5))
fig = ff.create_distplot([df2['toxicity']], ['toxicity'], bin_size=10)
fig.show()
