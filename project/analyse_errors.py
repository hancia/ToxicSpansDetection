from ast import literal_eval

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm, ListedColormap
from termcolor import colored

sns.set_context('paper', font_scale=1.65)


def cm_analysis(cm, filename, labels, ax, ymap=None, figsize=(10, 10)):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args:
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            if c==0:
                annot[i, j] = ''
            else:
                s = cm_sum[i]
                p = cm_perc[i, j]
                if i == j:
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                else:
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c,s)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'

    # fig, ax = plt.subplots(figsize=figsize)
    # sns.heatmap(cm, annot=annot, fmt='', ax=ax, cbar=True, robust=True)
    # sns.heatmap(cm, ax=ax, cbar=False, cmap=ListedColormap(['white']), square=True, linecolor='black', linewidths=1, rasterized=False)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax, cbar=False, cmap=ListedColormap(['white']), square=True, linecolor='black', linewidths=1, rasterized=False)



def get_confusion_matrix(test_df, preds):
    _tp, _tn, _fp, _fn = 0, 0, 0, 0
    for i in range(2000):
        # print('*' * 100)
        sentence = test_df.loc[i, 'text']

        true_spans = set(test_df.loc[i, 'spans'])
        negative_spans = set(range(len(sentence))) - true_spans
        pred_true_spans = set(preds.loc[i, 'spans'])
        pred_negative_spans = set(range(len(sentence))) - set(preds.loc[i, 'spans'])

        true_positive = true_spans & pred_true_spans
        true_negative = negative_spans & pred_negative_spans

        false_positive = pred_true_spans - true_spans
        false_negative = pred_negative_spans - negative_spans
        if len(sentence) != len(true_negative) + len(true_positive) + len(false_negative) + len(false_positive):
            raise ValueError('errror2')

        _tp += len(list(true_positive))
        _tn += len(list(true_negative))
        _fp += len(list(false_positive))
        _fn += len(list(false_negative))

        result = ''
        for i in range(len(sentence)):
            if i in true_positive:
                result += colored(sentence[i], color='green', attrs=['reverse'])
            elif i in true_negative:
                result += sentence[i]
            elif i in false_positive:
                result += colored(sentence[i], color='magenta', attrs=['reverse'])
            elif i in false_negative:
                result += colored(sentence[i], color='blue', attrs=['reverse'])
            else:
                raise ValueError('error')
        # print(result)
    return _tp, _tn, _fp, _fn
fig, ax = plt.subplots(ncols=4, figsize=(18,5))

ax[0].set_title('OrthoLSTM')
ax[1].set_title('SHAP')
ax[2].set_title('BERT')
ax[3].set_title('Ensemble')

test_df = pd.read_csv('data/spans/tsd_test_true_spans.csv')
test_df['spans'] = test_df['spans'].apply(literal_eval)

preds = pd.read_csv('data/spans/lstm_preds.txt', sep='\t', header=None, names=['spans'])
preds['spans'] = preds['spans'].apply(literal_eval)
tp, tn, fp, fn = get_confusion_matrix(test_df, preds)
cm = np.array([[tn, fp], [fn, tp]])
cm_analysis(cm, 'here2.png', labels=[0, 1],ax=ax[0])

preds = pd.read_csv('data/spans/shap_preds.txt', sep='\t', header=None, names=['spans'])
preds['spans'] = preds['spans'].apply(literal_eval)
tp, tn, fp, fn = get_confusion_matrix(test_df, preds)
cm = np.array([[tn, fp], [fn, tp]])
cm_analysis(cm, 'here2.png', labels=[0, 1],ax=ax[1])


preds = pd.read_csv('data/spans/bert_preds.txt', sep='\t', header=None, names=['spans'])
preds['spans'] = preds['spans'].apply(literal_eval)
tp, tn, fp, fn = get_confusion_matrix(test_df, preds)
cm = np.array([[tn, fp], [fn, tp]])
cm_analysis(cm, 'here2.png', labels=[0, 1],ax=ax[2])

preds = pd.read_csv('data/spans/ensemble_preds.txt', sep='\t', header=None, names=['spans'])
preds['spans'] = preds['spans'].apply(literal_eval)
tp, tn, fp, fn = get_confusion_matrix(test_df, preds)
cm = np.array([[tn, fp], [fn, tp]])
cm_analysis(cm, 'here2.png', labels=[0, 1],ax=ax[3])


fig.tight_layout()
plt.savefig('here4.png')