from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from datasets import tqdm
from torch.utils.data import DataLoader, Dataset


class DatasetModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, tokenizer, batch_size=32, cutoff=None):
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.train_df, self.val_df = None, None
        self.cutoff = cutoff

    def prepare_data(self, *args, **kwargs):
        train_df = pd.read_csv(str(self.data_dir / "tsd_train.csv"))
        val_df = pd.read_csv(str(self.data_dir / "tsd_trial.csv"))

        self.train_df = self._preprocess_df(train_df)
        self.val_df = self._preprocess_df(val_df)

        pd.set_option('display.max_columns', 500)
        print(self.train_df.head())

    def train_dataloader(self):
        return DataLoader(SemevalDataset(self.train_df), num_workers=8, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(SemevalDataset(self.val_df), num_workers=8, batch_size=self.batch_size)

    def _preprocess_df(self, df):
        max_length = 512
        if self.cutoff:
            df = df.head(self.cutoff)
        df.loc[:, 'spans'] = df['spans'].apply(literal_eval)

        texts, spans = df.text.values, df.spans

        data_list = list()
        for sentence, span in tqdm(zip(texts, spans)):
            encoded = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_offsets_mapping=True,
                                                 padding='max_length', max_length=max_length)
            encoded_span = np.array([
                1 if any((left <= chr_pos < right for chr_pos in span)) else 0
                for left, right in encoded['offset_mapping']
            ])
            tokens = np.array(encoded.encodings[0].tokens)
            no_pad_id = [i for i in range(max_length) if encoded['offset_mapping'][i] != (0, 0)]

            data_list.append(
                [sentence, span, encoded['input_ids'], encoded['attention_mask'], encoded['offset_mapping'],
                 encoded_span, tokens, no_pad_id])

        result_df = pd.DataFrame(
            data_list,
            columns=['raw_text', 'raw_spans', 'pad_tokenized_text', 'pad_attention_mask', 'pad_offset_mapping',
                     'pad_spans', 'pad_tokens', 'no_pad_id']
        )

        return result_df


class SemevalDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            'tokens': torch.tensor(row['pad_tokenized_text']),
            'attention_mask': torch.tensor(row['pad_attention_mask']),
            'labels': torch.tensor(row['pad_spans']),
            'no_pad_id': torch.tensor(row['no_pad_id']),
            'pad_offset_mapping': torch.tensor(row['pad_offset_mapping']),
            'raw_spans': torch.tensor(row['raw_spans'])
        }
