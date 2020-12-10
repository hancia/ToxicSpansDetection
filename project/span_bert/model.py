import numpy as np
import pytorch_lightning as pl
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertForTokenClassification


class LitModule(pl.LightningModule):

    def __init__(self, freeze=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2,
                                                                output_attentions=False, output_hidden_states=False)
        if freeze:
            for name, param in self.model.bert.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False

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

        # logits = outputs.logits
        # logits = logits.detach().cpu().numpy()
        # label_ids = b_labels.to('cpu').numpy()
        # pred_flat = np.argmax(preds, axis=-1).flatten().astype(int)
        # labels_flat = label_ids.flatten().astype(int)
        # f1 = f1_score(labels_flat, pred_flat)
        # self.log('f1', f1, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(), lr=2e-6, eps=1e-8)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
