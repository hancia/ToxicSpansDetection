import os
from configparser import ConfigParser
from pathlib import Path

import click
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger
from transformers import BertTokenizerFast, BertForTokenClassification, MobileBertTokenizerFast, \
    MobileBertForTokenClassification, SqueezeBertTokenizerFast, SqueezeBertForTokenClassification

from dataset import DatasetModule
from model import LitModule

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@click.command()
@click.option('-n', '--name', required=True, type=str)
@click.option('-dp', '--data-path', required=True, type=str)
@click.option('-m', '--model', default='bert', type=click.Choice(['bert', 'mobilebert', 'squeezebert']))
@click.option('--logger/--no-logger', default=True)
@click.option('-e', '--epochs', default=3, type=int)
@click.option('-f', '--freeze', default=0, type=float)
@click.option('--seed', default=0, type=int)
@click.option('-bs', '--batch-size', default=32, type=int)
@click.option('--data-cutoff', default=None, type=int,
              help='Number of data samples used in training and validation, used for local testing the code')
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
        logger.experiment.set_code(filename='projectq/span_bert/train.py', overwrite=True)
        logger.log_hyperparams(params)
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))

    model_checkpoint = ModelCheckpoint(filepath='checkpoints/{epoch:02d}-{f1_spans:.4f}', save_weights_only=True,
                                       save_top_k=3, monitor='f1_spans', mode='max', period=1)
    early_stop_callback = EarlyStopping(monitor='f1_spans', mode='max', min_delta=0.01, patience=10, verbose=True)
    callbacks.extend([model_checkpoint, early_stop_callback])

    model_data = {
        'bert': [BertForTokenClassification, BertTokenizerFast, 'bert-base-uncased'],
        'mobilebert': [MobileBertForTokenClassification, MobileBertTokenizerFast, 'google/mobilebert-uncased'],
        'squeezebert': [SqueezeBertForTokenClassification, SqueezeBertTokenizerFast,
                        'squeezebert/squeezebert-mnli-headless']
    }
    model_class, tokenizer_class, model_name = model_data[params.model]
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=True)
    model_backbone = model_class.from_pretrained(model_name, num_labels=2, output_attentions=False,
                                                 output_hidden_states=False)

    data_module = DatasetModule(data_dir=params.data_path, tokenizer=tokenizer, batch_size=params.batch_size,
                                cutoff=params.data_cutoff)
    model = LitModule(model=model_backbone, tokenizer=tokenizer, freeze=params.freeze)

    trainer = Trainer(logger=logger, max_epochs=params['epochs'], callbacks=callbacks, gpus=1, deterministic=True)
    trainer.fit(model, datamodule=data_module)

    if params.logger:
        for absolute_path in model_checkpoint.best_k_models.keys():
            logger.experiment.log_model(Path(absolute_path).name, absolute_path)
        logger.log_metrics({'best_model_score': model_checkpoint.best_model_score.tolist()})


if __name__ == '__main__':
    train()
