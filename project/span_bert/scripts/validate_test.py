import csv
from ast import literal_eval
from collections import defaultdict

import click
import numpy as np
import torch
from tqdm import tqdm
from transformers import BertForTokenClassification, AlbertForTokenClassification, ElectraForTokenClassification, \
    RobertaForTokenClassification, XLNetForTokenClassification, MobileBertForTokenClassification, \
    SqueezeBertForTokenClassification, SqueezeBertTokenizerFast, XLNetTokenizerFast, MobileBertTokenizerFast, \
    RobertaTokenizerFast, ElectraTokenizerFast, AlbertTokenizerFast, BertTokenizerFast

from dataset import DatasetModule
from utils import get_api_and_experiment, f1_semeval, fill_holes_in_row, remove_ones_in_row, fill_holes_in_row_three
from model import LitModule


@click.command()
@click.option('--experiment', required=True, type=str, help='For example ce132011516346c99185d139fb23c70c')
@click.option('--weights-path', required=True, type=str, help='For example epoch=25-val_mae=8.2030.ckpt')
def validate(experiment, weights_path):
    _, experiment = get_api_and_experiment(experiment)
    model_param = experiment.get_parameters_summary("model")['valueCurrent']
    # length = int(experiment.get_parameters_summary("length")['valueCurrent'])
    length=512
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
    model_class, tokenizer_class, model_name = model_data[model_param]
    tokenizer = tokenizer_class.from_pretrained(model_name, do_lower_case=True)
    model_backbone = model_class.from_pretrained(model_name, num_labels=2, output_attentions=False,
                                                 output_hidden_states=False)

    data_module = DatasetModule(data_dir='data/spans', tokenizer=tokenizer, batch_size=4, length=length)
    data_module.prepare_data()
    # experiment.download_model(name=weights_path, output_path='comet-ml/', expand=True)
    model = LitModule.load_from_checkpoint(weights_path, model=model_backbone, tokenizer=tokenizer, lr=4.7e-5,
                                           freeze=False)
    model.eval()
    model.cuda()

    result_spans = defaultdict(lambda: defaultdict(list))
    for batch in tqdm(data_module.val_dataloader()):
        with torch.no_grad():
            outputs = model(batch['input_ids'].cuda(), token_type_ids=None,
                            attention_mask=batch['attention_mask'].cuda(),
                            labels=batch['labels'].cuda())
            logits = outputs.logits.detach().cpu().numpy()
            y_pred = np.argmax(logits, axis=-1).astype(int)

            y_true = batch['labels'].cpu().numpy().astype(int)
            pad_span = batch['pad_span'].cpu().numpy().astype(int)
            offset_mapping = batch['offset_mapping'].cpu().numpy().astype(int)
            sentence_id = batch['sentence_id'].cpu().numpy().astype(int)
            sentence_offset = batch['offset'].cpu().numpy().astype(int)

            for i in range(len(y_true)):
                true_spans = list(set(pad_span[i]) - {-1})  # remove padding
                predicted_offsets = offset_mapping[i][y_pred[i].astype(bool)]
                predicted_spans = [i for offset in predicted_offsets for i in range(offset[0], offset[1])]

                result_spans[sentence_id[i]]['true'].extend(list(np.array(true_spans) + sentence_offset[i]))
                result_spans[sentence_id[i]]['pred'].extend(list(np.array(predicted_spans) + sentence_offset[i]))
    f1_semeval_avg = np.array([f1_semeval(result_spans[sentence_id]['true'], result_spans[sentence_id]['pred'])
                               for sentence_id in result_spans])
    print(np.mean(f1_semeval_avg))
    f1_semeval_avg = np.array([f1_semeval(result_spans[sentence_id]['true'],
                                          literal_eval(fill_holes_in_row(str(result_spans[sentence_id]['pred']))))
                               for sentence_id in result_spans])
    print(np.mean(f1_semeval_avg))
    f1_semeval_avg = np.array([f1_semeval(result_spans[sentence_id]['true'],
                                          literal_eval(fill_holes_in_row_three(str(result_spans[sentence_id]['pred']))))
                               for sentence_id in result_spans])
    print(np.mean(f1_semeval_avg))
    f1_semeval_avg = np.array([f1_semeval(result_spans[sentence_id]['true'],
                                          literal_eval(remove_ones_in_row(str(result_spans[sentence_id]['pred']))))
                               for sentence_id in result_spans])
    print(np.mean(f1_semeval_avg))
    f1_semeval_avg = np.array([f1_semeval(result_spans[sentence_id]['true'], literal_eval(
        remove_ones_in_row(fill_holes_in_row(str(result_spans[sentence_id]['pred'])))))
                               for sentence_id in result_spans])
    print(np.mean(f1_semeval_avg))
    #
    # predicted_df = model.predict_dataframe(data_module.test_df, length)
    # predicted_df.to_csv('spans-pred.txt', header=False, sep='\t', quoting=csv.QUOTE_NONE, escapechar='\n')


if __name__ == '__main__':
    validate()
