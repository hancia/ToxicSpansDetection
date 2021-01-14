import csv
import os
from configparser import ConfigParser
from pathlib import Path

import click
import numpy as np
import torch
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from transformers import BertTokenizerFast, BertForTokenClassification, MobileBertTokenizerFast, \
    MobileBertForTokenClassification, SqueezeBertTokenizerFast, SqueezeBertForTokenClassification, \
    AlbertForTokenClassification, AlbertTokenizerFast, ElectraForTokenClassification, RobertaForTokenClassification, \
    XLNetForTokenClassification, XLNetTokenizerFast, RobertaTokenizerFast, ElectraTokenizerFast

from dataset import DatasetModule
from fill_holes import fill_holes_in_row
from model import LitModule
from split_sentences import split_sentence

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['COMET_DISABLE_AUTO_LOGGING'] = '1'


@click.command()
@click.option('-n', '--name', required=True, type=str)
@click.option('-dp', '--data-path', required=True, type=str)
@click.option('-m', '--model', default='bert',
              type=click.Choice(['bert', 'mobilebert', 'squeezebert', 'albert', 'xlnet', 'roberta', 'electra']))
@click.option('-l', '--length', default=512, type=click.Choice([128, 512]))
@click.option('--logger/--no-logger', default=True)
@click.option('-e', '--epochs', default=4, type=int)
@click.option('-f', '--freeze', default=0, type=float)
@click.option('--seed', default=0, type=int)
@click.option('-bs', '--batch-size', default=32, type=int)
@click.option('-fdr', '--fast-dev-run', default=False, is_flag=True)
@click.option('-sm', '--smoothing', default=False, is_flag=True)
def train(**params):
    params = EasyDict(params)
    seed_everything(params.seed)

    config = ConfigParser()
    config.read('config.ini')

    logger, callbacks = False, list()
    if params.logger:
        comet_config = EasyDict(config['cometml'])
        logger = CometLogger(api_key=comet_config.apikey, project_name=comet_config.projectname,
                             workspace=comet_config.workspace)
        logger.experiment.set_code(filename='project/span_bert/train.py', overwrite=True)
        logger.log_hyperparams(params)
        logger.experiment.log_asset_folder('project/span_bert')
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    model_checkpoint = ModelCheckpoint(filepath='checkpoints/{epoch:02d}-{f1_spans:.4f}-{f1_spans_sentence:.4f}',
                                       save_weights_only=True, save_top_k=3, monitor='f1_spans_sentence', mode='max',
                                       period=1)
    callbacks.extend([model_checkpoint])

    model_data = {
        'bert': [BertForTokenClassification, BertTokenizerFast, 'bert-base-uncased'],
        'albert': [AlbertForTokenClassification, AlbertTokenizerFast, 'albert-base-v2'],
        'electra': [ElectraForTokenClassification, ElectraTokenizerFast, 'google/electra-small-discriminator'],
        'roberta': [RobertaForTokenClassification, RobertaTokenizerFast, 'roberta-base'],
        'xlnet': [XLNetForTokenClassification, XLNetTokenizerFast, 'xlnet-base-cased'],
        'mobilebert': [MobileBertForTokenClassification, MobileBertTokenizerFast, 'google/mobilebert-uncased'],
        'squeezebert': [SqueezeBertForTokenClassification, SqueezeBertTokenizerFast,
                        'squeezebert/squeezebert-mnli-headless']
    }
    model_class, tokenizer_class, model_name = model_data[params.model]
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=True)
    model_backbone = model_class.from_pretrained(model_name, num_labels=2, output_attentions=False,
                                                 output_hidden_states=False)

    data_module = DatasetModule(data_dir=params.data_path, tokenizer=tokenizer, batch_size=params.batch_size,
                                length=params.length, smoothing=params.smoothing)
    model = LitModule(model=model_backbone, tokenizer=tokenizer, freeze=params.freeze)

    trainer = Trainer(logger=logger, max_epochs=params['epochs'], callbacks=callbacks, gpus=1, deterministic=True,
                      val_check_interval=0.5, fast_dev_run=params.fast_dev_run)
    trainer.fit(model, datamodule=data_module)

    if params.logger:
        for absolute_path in model_checkpoint.best_k_models.keys():
            logger.experiment.log_model(Path(absolute_path).name, absolute_path)
        logger.log_metrics({'best_model_score': model_checkpoint.best_model_score.tolist()})

        best_model = LitModule.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path,
                                                    model=model_backbone, tokenizer=tokenizer, freeze=params.freeze)
        best_model.eval()
        best_model.cuda()
        for i, row in data_module.test_df.iterrows():
            texts, offsets, _ = split_sentence(tokenizer, row['text'], max_sentence_length=params.length)
            predicted_spans = list()
            for text, offset in zip(texts, offsets):
                encoded = tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True,
                                    return_offsets_mapping=True, max_length=params.length)
                item = {k: torch.tensor(v).unsqueeze(0).long().cuda() for k, v in encoded.items()}

                output = best_model(item['input_ids'], token_type_ids=None, attention_mask=item['attention_mask'])
                logits = output.logits.detach().cpu().numpy()
                y_pred = np.argmax(logits, axis=-1).squeeze().astype(int)
                predicted_offsets = np.array(encoded['offset_mapping'])[y_pred.astype(bool)]
                spans = [i for offset in predicted_offsets for i in range(offset[0], offset[1])]
                spans = np.array(spans) + offset
                predicted_spans.extend(list(spans))
            data_module.test_df.loc[i, 'spans'] = str(predicted_spans)

        print(data_module.test_df.head())
        data_module.test_df = data_module.test_df.drop(columns=['text'])
        data_module.test_df.to_csv('spans-pred.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')
        logger.experiment.log_asset('spans-pred.txt')

        data_module.test_df['spans'] = data_module.test_df['spans'].apply(fill_holes_in_row)
        data_module.test_df.to_csv('spans-pred-filled.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE,
                                   escapechar='\n')
        logger.experiment.log_asset('spans-pred-filled.txt')


if __name__ == '__main__':
    train()
