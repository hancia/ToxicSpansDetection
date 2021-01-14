from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class DatasetModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, tokenizer, batch_size=32, length=512, smoothing=False):
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.length = length
        self.smoothing = smoothing
        self.train_df, self.val_df, self.test_df = None, None, None

    def prepare_data(self, *args, **kwargs):
        self.train_df = pd.read_csv(str(self.data_dir / f'tsd_train_{str(self.length)}.csv'))
        self.val_df = pd.read_csv(str(self.data_dir / f'tsd_trial_{str(self.length)}.csv'))
        self.test_df = pd.read_csv(str(self.data_dir / "tsd_test.csv"))

        self.train_df.loc[:, 'spans'] = self.train_df['spans'].apply(literal_eval)
        self.val_df.loc[:, 'spans'] = self.val_df['spans'].apply(literal_eval)

    def train_dataloader(self):
        return DataLoader(
            SemevalDataset(self.train_df, tokenizer=self.tokenizer, length=self.length, smoothing=self.smoothing),
            num_workers=8, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            SemevalDataset(self.val_df, tokenizer=self.tokenizer, length=self.length, smoothing=self.smoothing),
            num_workers=8, batch_size=self.batch_size, shuffle=False)


class SemevalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, length, smoothing=False):
        self.df = df
        self.tokenizer = tokenizer
        self.length = length
        if smoothing:
            self.pos, self.neg = 0.9, 0.1
        else:
            self.pos, self.neg = 1, 0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = self.tokenizer(row['text'], add_special_tokens=True, padding='max_length', truncation=True,
                                 return_offsets_mapping=True, max_length=self.length)
        encoded['labels'] = np.array([
            self.pos if any((left <= chr_pos < right for chr_pos in row['spans'])) else self.neg
            for left, right in encoded['offset_mapping']
        ])
        encoded['sentence_id'] = row['sentence_id']
        encoded['offset'] = row['offset']
        # 994 is the longest input, 553 after splitting
        encoded['pad_span'] = np.pad(row['spans'], mode='constant', pad_width=(0, 560 - len(row['spans'])),
                                     constant_values=-1)

        item = {k: torch.tensor(v).long() for k, v in encoded.items()}
        return item
