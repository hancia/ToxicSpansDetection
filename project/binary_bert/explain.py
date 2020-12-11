import shap
import numpy as np
import torch
from datasets import load_dataset
from tqdm.contrib import tenumerate

from project.binary_bert.utils import load_binary_bert

train = load_dataset("civil_comments", split='train')
test = load_dataset("civil_comments", split='test')

model, tokenizer, class_names = load_binary_bert()

txts = []
for id, batch in tenumerate(test, total=10):
    if id > 10:
        break
    inputs = tokenizer(
        batch['text'], return_tensors="pt", truncation=True, padding=True
    )
    txts.append(inputs['input_ids'].tolist())

train_txt = []
for id, batch in tenumerate(train, total=10):
    if id > 10:
        break
    inputs = tokenizer(
        batch['text'], return_tensors="pt", truncation=True, padding=True
    )
    train_txt.append(inputs['input_ids'].tolist())

model.eval()
with torch.no_grad():
    e = shap.DeepExplainer(model, torch.tensor(train_txt[0]))
    shap_values = e.shap_values(torch.tensor(txts[0]))

print(shap_values)