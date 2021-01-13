from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
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
        train_df = pd.read_csv(str(self.data_dir / "tsd_train_128.csv"))
        val_df = pd.read_csv(str(self.data_dir / "tsd_trial_128.csv"))

        self.train_df = self._preprocess_df(train_df)
        self.val_df = self._preprocess_df(val_df)

    def train_dataloader(self):
        return DataLoader(SemevalDataset(self.train_df, tokenizer=self.tokenizer), num_workers=8,
                          batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(SemevalDataset(self.val_df, tokenizer=self.tokenizer), num_workers=8,
                          batch_size=self.batch_size)

    def _preprocess_df(self, df):
        if self.cutoff:
            df = df.head(self.cutoff)
        df.loc[:, 'spans'] = df['spans'].apply(literal_eval)
        return df


class SemevalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer):
        self.df = df
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = self.tokenizer(row['text'], add_special_tokens=True, padding='max_length', truncation=True,
                                 return_offsets_mapping=True, max_length=128)
        encoded['labels'] = np.array([
            1 if any((left <= chr_pos < right for chr_pos in row['spans'])) else 0
            for left, right in encoded['offset_mapping']
        ])
        encoded['sentence_id'] = row['sentence_id']
        encoded['offset'] = row['offset']
        # 994 is the longest input, 533 after splitting
        encoded['pad_span'] = np.pad(row['spans'], mode='constant', pad_width=(0, 560 - len(row['spans'])),
                                     constant_values=-1)

        item = {k: torch.tensor(v).long() for k, v in encoded.items()}
        return item
