import pandas as pd
import plotly.figure_factory as ff

df = pd.read_csv('data/civil_comments/train.csv').head(10000)
fig = ff.create_distplot([df['toxicity']], ['toxicity'], bin_size=10)
fig.show()
