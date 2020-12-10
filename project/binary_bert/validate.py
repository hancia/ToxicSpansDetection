from collections import defaultdict

import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from tqdm.contrib import tenumerate

from project.binary_bert.utils import load_binary_bert

dataset = load_dataset("civil_comments", split='test')

model, tokenizer, class_names = load_binary_bert()
true, pred = defaultdict(list), defaultdict(list)
model.eval()

length = 5000
results = {}

for id, sample in tenumerate(dataset, total=length):
    if id >= length:
        break

    with torch.no_grad():
        inputs = tokenizer(sample['text'], return_tensors="pt", truncation=True, padding=True).to(model.device)
        out = model(inputs['input_ids'])
        scores = torch.sigmoid(out[0]).cpu().detach().numpy()

    for i, cla in enumerate(class_names):
        results[cla] = (
            scores[0][i]
            if isinstance(sample['text'], str)
            else [scores[ex_i][i].tolist() for ex_i in range(len(scores))]
        )
        sample_class = 'identity_attack' if cla == 'identity_hate' else cla

        true[sample_class].append(int(sample[sample_class] > 0.5))
        pred[sample_class].append(results[cla])

for i in class_names:
    try:
        print(i, roc_auc_score(true[i], pred[i]))
    except ValueError:
        print(i, "Only one class present in y_true")
