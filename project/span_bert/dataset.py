from ast import literal_eval
from pathlib import Path
from random import randint, sample

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset


class DatasetModule(pl.LightningDataModule):

    def __init__(self, data_dir: str, tokenizer, batch_size=32, length=512, augmentation=False, valintrain=False):
        super().__init__()
        self.data_dir: Path = Path(data_dir)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.length = length
        self.train_df, self.val_df, self.test_df = None, None, None
        self.augmentation = augmentation
        self.valintrain = valintrain

    def prepare_data(self, *args, **kwargs):
        self.train_df = pd.read_csv(str(self.data_dir / f'tsd_train_{str(self.length)}.csv'))
        self.val_df = pd.read_csv(str(self.data_dir / f'tsd_trial_{str(self.length)}.csv'))
        self.test_df = pd.read_csv(str(self.data_dir / "tsd_test.csv"))

        if self.valintrain:
            self.train_df = pd.concat([self.train_df, self.val_df])

        self.train_df.loc[:, 'spans'] = self.train_df['spans'].apply(literal_eval)
        self.val_df.loc[:, 'spans'] = self.val_df['spans'].apply(literal_eval)

    def train_dataloader(self):
        return DataLoader(
            SemevalDataset(self.train_df, tokenizer=self.tokenizer, length=self.length, augmentation=self.augmentation),
            num_workers=8, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(
            SemevalDataset(self.val_df, tokenizer=self.tokenizer, length=self.length),
            num_workers=8, batch_size=self.batch_size, shuffle=False)


class SemevalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, length, augmentation=False):
        self.df = df
        self.tokenizer = tokenizer
        self.length = length
        self.augmentation = augmentation

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = self.tokenizer(row['text'], add_special_tokens=True, padding='max_length', truncation=True,
                                 return_offsets_mapping=True, max_length=self.length)
        encoded['labels'] = np.array([
            1 if any(left <= chr_pos < right for chr_pos in row['spans']) else 0
            for left, right in encoded['offset_mapping']
        ])

        encoded['sentence_id'] = row['sentence_id']
        encoded['offset'] = row['offset']
        # 994 is the longest input, 553 after splitting
        encoded['pad_span'] = np.pad(row['spans'], mode='constant', pad_width=(0, 560 - len(row['spans'])),
                                     constant_values=-1)

        item = {k: torch.tensor(v).long() for k, v in encoded.items()}

        if self.augmentation:
            number_of_tokens = sum(encoded['attention_mask'])
            indices_list_to_shuffle = list(range(number_of_tokens))
            number_swaps = int(0.5 * number_of_tokens)
            for i in range(number_swaps):
                a = randint(1, number_of_tokens - 2)  # dont swap first and last token!
                neigh_len = randint(1, 3)
                symbol = sample([-1, 1], 1)[0]
                b = a + neigh_len * symbol
                if 0 <= b <= number_of_tokens - 1 and a != b:
                    indices_list_to_shuffle[a], indices_list_to_shuffle[b] = indices_list_to_shuffle[b], \
                                                                             indices_list_to_shuffle[a]

            def swap_with_indices(input_tensor):
                if len(input_tensor.shape) > 0 and input_tensor.shape[0] == self.length:
                    input_tensor[:number_of_tokens] = input_tensor[torch.LongTensor(indices_list_to_shuffle)]
                return input_tensor

            item = {k: swap_with_indices(v) for k, v in item.items()}

        return item
