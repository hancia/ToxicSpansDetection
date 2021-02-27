from collections import defaultdict

import torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from tqdm.contrib import tenumerate
import numpy as np

from project.binary_bert.utils import load_binary_bert

dataset = load_dataset("civil_comments", split='test')
dataloader = DataLoader(dataset, batch_size=8)
model, tokenizer, class_names = load_binary_bert()
true, pred = defaultdict(list), defaultdict(list)
model.eval()
with torch.no_grad():
    for id, batch in tenumerate(dataloader, total=len(dataloader)):
        inputs = tokenizer(
            batch['text'], return_tensors="pt", truncation=True, padding=True
        ).to(model.device)
        out = model(inputs['input_ids'])
        scores = torch.sigmoid(out[0]).cpu().detach().numpy()
        results = {}
        for i, cla in enumerate(class_names):
            results[cla] = (
                scores[0][i]
                if isinstance(batch['text'], str)
                else [scores[ex_i][i].tolist() for ex_i in range(len(scores))]
            )
            if cla == 'identity_hate':
                batch_cla = 'identity_attack'
            else:
                batch_cla = cla
            true[cla].extend((batch[batch_cla] > 0.5).int())
            pred[cla].extend(results[cla])

    roc_auc = []

    for i in class_names:
        try:
            print(i, roc_auc_score(true[i], pred[i]))
            roc_auc.append(roc_auc_score(true[i], pred[i]))
        except ValueError:
            print("Only one class present in y_true")

    print(np.mean(roc_auc))