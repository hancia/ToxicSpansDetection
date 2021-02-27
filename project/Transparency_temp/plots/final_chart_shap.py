import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_style('whitegrid')
sns.set_context('paper', font_scale=1.2)
pd.set_option('display.max_rows', 110)

df = pd.read_csv('shap_thresholds_2000.csv')
print('IDXmax', df.idxmax())
print('MAX', df.max())

df = df.set_index('threshold')
plt.ylabel('f1')

g = sns.scatterplot(data=df)
plt.ylim(0,0.62)
# plt.show()
g.get_figure().savefig('elbow_attention_shap.png')
