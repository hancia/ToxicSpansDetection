import numpy as np
import scipy as sp
import shap
import torch
from datasets import load_dataset
from tqdm.contrib import tenumerate

from project.binary_bert.utils import load_binary_bert

model, tokenizer, class_names = load_binary_bert()


def get_tensor(split, size):
    data = load_dataset("civil_comments", split=split)
    result = list()

    for id, sample in tenumerate(data, total=size):
        if id > size:
            break

        result.append(sample['text'])

    return result


train = get_tensor('train', 1)
test = get_tensor('test', 1)


# define a prediction function

def f(x):
    tv = torch.tensor(
        [tokenizer.encode(v, pad_to_max_length=True, max_length=500, truncation=True) for v in x])
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
    return val


# build an explainer using a token masker
explainer = shap.Explainer(f, tokenizer)

# explain the model's predictions on IMDB reviews
# imdb_train = nlp.load_dataset("imdb")["train"]
shap_values = explainer(train)

# e = shap.DeepExplainer(model, train)
# shap_values = e.shap_values(test)
#
# print(shap_values)
