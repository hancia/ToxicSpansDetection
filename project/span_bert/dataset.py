from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class DatasetModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, tokenizer, batch_size=32, max_len=512, cutoff=None):
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_len
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
        if self.cutoff:
            df = df.head(self.cutoff)
        df.loc[:, 'spans'] = df['spans'].apply(literal_eval)

        texts, spans = df.text.values, df.spans

        data_list = list()
        for sentence, span in tqdm(zip(texts, spans), total=len(texts)):
            encoded = self.tokenizer.encode_plus(sentence, add_special_tokens=True, return_offsets_mapping=True,
                                                 padding='max_length', max_length=self.max_length)
            encoded_span = np.array([
                1 if any((left <= chr_pos < right for chr_pos in span)) else 0
                for left, right in encoded['offset_mapping']
            ])
            tokens = np.array(encoded.encodings[0].tokens)
            padded_span = np.pad(span, mode='constant', pad_width=(0, 1024 - len(span)),
                                 constant_values=-1)  # 994 is the longest input
            data_list.append([sentence, span, padded_span, encoded['input_ids'], encoded['attention_mask'],
                              encoded['offset_mapping'], encoded_span, tokens])

        result_df = pd.DataFrame(
            data_list,
            columns=['raw_text', 'raw_spans', 'pad_raw_spans', 'pad_tokenized_text', 'pad_attention_mask',
                     'pad_offset_mapping', 'pad_encoded_span', 'pad_tokens']
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
            'tokens': torch.tensor(row['pad_tokenized_text']).long(),
            'attention_mask': torch.tensor(row['pad_attention_mask']).long(),
            'labels': torch.tensor(row['pad_encoded_span']).long(),
            'pad_offset_mapping': torch.tensor(row['pad_offset_mapping']).long(),
            'pad_span': torch.tensor(row['pad_raw_spans']).long()
        }
