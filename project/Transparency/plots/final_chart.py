import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)
pd.set_option('display.max_rows', 110)

df = pd.read_csv('threshold_cumulative.csv')
df = df.drop(labels=['Unnamed: 0'], axis=1)
df.columns = ['threshold', 'cumulative']

df2 = pd.read_csv('threshold_not_cumulative.csv')
df2 = df2.drop(labels=['Unnamed: 0'], axis=1)
df2.columns = ['threshold', 'value']

df['value'] = df2['value']

print(df)
print('IDXmax', df.idxmax())
print('MAX', df.max())

df = df.set_index('threshold')
plt.ylabel('f1')

g = sns.scatterplot(data=df)
plt.ylim(0, 0.62)
# plt.show()
g.get_figure().savefig('elbow_attention.png')
