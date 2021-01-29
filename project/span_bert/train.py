import os
from configparser import ConfigParser
from pathlib import Path

import click
import pandas as pd
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CometLogger
from transformers import BertTokenizerFast, BertForTokenClassification, MobileBertTokenizerFast, \
    MobileBertForTokenClassification, SqueezeBertTokenizerFast, SqueezeBertForTokenClassification, \
    AlbertForTokenClassification, AlbertTokenizerFast, ElectraForTokenClassification, RobertaForTokenClassification, \
    XLNetForTokenClassification, XLNetTokenizerFast, RobertaTokenizerFast, ElectraTokenizerFast

from dataset import DatasetModule
from model import LitModule
from utils import log_predicted_spans

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
@click.option('-lr', '--epochs', default=None, type=float)
@click.option('-f', '--freeze', default=0, type=float)
@click.option('--seed', default=0, type=int)
@click.option('-bs', '--batch-size', default=32, type=int)
@click.option('-fdr', '--fast-dev-run', default=False, is_flag=True)
@click.option('--augmentation', default=False, is_flag=True)
@click.option('--valintrain', default=False, is_flag=True)
@click.option('--pseudolabel', default=False, is_flag=True)
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

    model_checkpoint = ModelCheckpoint(filepath='checkpoints/{epoch:02d}-{val_loss:.4f}-{f1_spans_sentence:.4f}',
                                       save_weights_only=True, save_top_k=10, monitor='val_loss', mode='min', period=1)
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
                                length=params.length, augmentation=params.augmentation, valintrain=params.valintrain)
    model = LitModule(model=model_backbone, tokenizer=tokenizer, freeze=params.freeze, lr=params.lr)

    trainer = Trainer(logger=logger, max_epochs=params.epochs, callbacks=callbacks, gpus=1, deterministic=True,
                      val_check_interval=0.5, fast_dev_run=params.fast_dev_run)

    if params.lr is None:
        lr_finder = trainer.tuner.lr_find(model, datamodule=data_module)
        model.lr = lr_finder.suggestion()

    print(model.lr)
    if params.logger:
        logger.log_hyperparams({'lr': model.lr})

    trainer.fit(model, datamodule=data_module)

    if params.pseudolabel:
        pseudolabel_data = pd.read_csv('data/civil_comments/all_civil_data_512.csv')
        already_labeled = pd.DataFrame(columns=[*pseudolabel_data.columns, 'spans'])
        model = LitModule.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path, model=model_backbone,
                                               tokenizer=tokenizer, freeze=params.freeze, lr=2e-5, scheduler=False)

        while len(pseudolabel_data) > 0:
            size_to_label = min(len(data_module.train_df), len(pseudolabel_data))
            df_subset = pseudolabel_data.sample(size_to_label)
            pseudolabel_data = pseudolabel_data.drop(df_subset.index)

            df_subset['spans'] = model.predict_dataframe(df_subset, params.length)

            already_labeled = pd.concat([already_labeled, df_subset])
            data_module = DatasetModule(data_dir=params.data_path, tokenizer=tokenizer, batch_size=params.batch_size,
                                        length=params.length, augmentation=params.augmentation,
                                        valintrain=params.valintrain, injectdataset=already_labeled)

            trainer = Trainer(logger=logger, max_epochs=params.epochs, callbacks=callbacks, gpus=1, deterministic=True,
                              val_check_interval=0.5, fast_dev_run=params.fast_dev_run)
            trainer.fit(model, datamodule=data_module)

    if params.logger:
        for absolute_path in model_checkpoint.best_k_models.keys():
            logger.experiment.log_model(Path(absolute_path).name, absolute_path)
        logger.log_metrics({'best_model_score': model_checkpoint.best_model_score.tolist()})

        best_model = LitModule.load_from_checkpoint(checkpoint_path=model_checkpoint.best_model_path,
                                                    model=model_backbone, tokenizer=tokenizer, freeze=params.freeze)

        predicted_df = best_model.predict_dataframe(data_module.test_df, params.length)
        log_predicted_spans(predicted_df, logger)
        print(predicted_df.head())


if __name__ == '__main__':
    train()
