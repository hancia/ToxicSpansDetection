import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import SqueezeBertForTokenClassification

from semeval_utils import f1_semeval


class LitModule(pl.LightningModule):

    def __init__(self, model, freeze, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

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
            batch['tokens'],
            token_type_ids=None,
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        self.log('train_loss', loss.item(), logger=True, on_step=False, on_epoch=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_nb):
        outputs = self(
            batch['tokens'],
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
        # it takes into account special tokens but it should not affect results

        f1_avg = list()
        f1_pad_avg = list()
        for i in range(len(y_true)):
            y_pred_no_pad = y_pred[i][no_pad_id[i]]
            y_true_no_pad = y_true[i][no_pad_id[i]]
            f1 = f1_score(y_true_no_pad, y_pred_no_pad)
            f1_avg.append(f1)
            f1_pad_avg.append(f1_score(y_pred[i], y_true[i]))

        self.log('f1', np.mean(np.array(f1_avg)), prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('f1_pad', np.mean(np.array(f1_pad_avg)), prog_bar=True, logger=True, on_step=False, on_epoch=True)

        pad_span = batch['pad_span'].to('cpu').numpy().astype(int)
        pad_offset_mapping = batch['pad_offset_mapping'].to('cpu').numpy().astype(int)

        f1_semeval_avg = list()
        for i in range(len(y_true)):
            true_spans = list(set(pad_span[i]) - {-1})  # remove padding
            predicted_offsets = pad_offset_mapping[i][y_pred[i].astype(bool)]
            predicted_spans = [i for offset in predicted_offsets for i in range(offset[0], offset[1])]
            # because of set used in f1 func we dont have to care about no_pad_id
            # in worst case one token '0' will be counted ;) but attention mask can be used here

            f1 = f1_semeval(predicted_spans, true_spans)
            f1_semeval_avg.append(f1)

        self.log('f1_spans', np.mean(np.array(f1_semeval_avg)), prog_bar=True, logger=True, on_step=False,
                 on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
