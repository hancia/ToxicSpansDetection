from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from transformers import SqueezeBertForTokenClassification

from semeval_utils import f1_semeval


class LitModule(pl.LightningModule):

    def __init__(self, model, tokenizer, freeze, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.tokenizer = tokenizer

        if freeze > 0:
            for name, param in self.model.base_model.embeddings.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False

            encoder = self.model.base_model.encoder
            encoder_layers = encoder.layers \
                if isinstance(self.model, SqueezeBertForTokenClassification) \
                else encoder.layer

            layers_size = len(encoder_layers)
            freeze_layers = int(layers_size * freeze)
            print(f'Freeze {freeze_layers}/{layers_size}')

            for name, param in encoder_layers[:freeze_layers].named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False

        train_params = sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, self.model.parameters())])
        all_params = sum([np.prod(p.size()) for p in self.model.parameters()])
        print(f'Train {train_params / all_params:.4%} params')

    def forward(self, *args, **kwargs):
        pred = self.model(*args, **kwargs)
        return pred

    def training_step(self, batch, batch_nb):
        outputs = self(
            batch['input_ids'],
            token_type_ids=None,
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss.item(), logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        outputs = self(
            batch['input_ids'],
            token_type_ids=None,
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('val_loss', loss.item(), logger=True, on_step=False, on_epoch=True)

        logits = outputs.logits.detach().cpu().numpy()
        y_pred = np.argmax(logits, axis=-1).astype(int)

        y_true = batch['labels'].to('cpu').numpy().astype(int)
        no_pad_id = batch['attention_mask'].to('cpu').numpy().astype('bool')

        f1_avg = list()
        for i in range(len(y_true)):
            y_pred_no_pad = y_pred[i][no_pad_id[i]]
            y_true_no_pad = y_true[i][no_pad_id[i]]
            f1 = f1_score(y_true_no_pad, y_pred_no_pad)
            f1_avg.append(f1)

        self.log('f1', np.mean(np.array(f1_avg)), prog_bar=True, logger=True, on_step=False, on_epoch=True)

        pad_span = batch['pad_span'].to('cpu').numpy().astype(int)
        offset_mapping = batch['offset_mapping'].to('cpu').numpy().astype(int)
        sentence_id = batch['sentence_id'].to('cpu').numpy().astype(int)
        sentence_offset = batch['offset'].to('cpu').numpy().astype(int)

        f1_semeval_avg = list()
        result_spans = defaultdict(lambda: defaultdict(list))
        for i in range(len(y_true)):
            true_spans = list(set(pad_span[i]) - {-1})  # remove padding
            predicted_offsets = offset_mapping[i][y_pred[i].astype(bool)]
            predicted_spans = [i for offset in predicted_offsets for i in range(offset[0], offset[1])]

            f1 = f1_semeval(predicted_spans, true_spans)
            f1_semeval_avg.append(f1)
            result_spans[sentence_id[i]]['true'].extend(list(np.array(true_spans) + sentence_offset[i]))
            result_spans[sentence_id[i]]['pred'].extend(list(np.array(predicted_spans) + sentence_offset[i]))

        self.log('f1_spans', np.mean(np.array(f1_semeval_avg)), prog_bar=True, logger=True, on_step=False,
                 on_epoch=True)

        return result_spans

    def validation_epoch_end(self, outs):
        result_spans = defaultdict(lambda: defaultdict(list))
        for out in outs:
            for sentence_id in out:
                result_spans[sentence_id]['true'].extend(out[sentence_id]['true'])
                result_spans[sentence_id]['pred'].extend(out[sentence_id]['pred'])

        f1_semeval_avg = list()
        for sentence_id in result_spans:
            f1 = f1_semeval(result_spans[sentence_id]['true'], result_spans[sentence_id]['pred'])
            f1_semeval_avg.append(f1)
        self.log('f1_spans_sentence', np.mean(np.array(f1_semeval_avg)), prog_bar=True, logger=True)

    def test_step(self, *args, **kwargs):
        print('*'*50)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=4e-5, eps=1e-8)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
