from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class TextDetectionDataset(Dataset):
    def __init__(self, data_path, n_samples=10000):
        super().__init__()
        data_df = pd.read_json(data_path, lines=True)
        self.data = data_df.iloc[:n_samples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data["text"][idx], self.data["label"][idx]


class Collator:
    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        texts = [elem[0] for elem in batch]
        labels = [elem[1] for elem in batch]

        tokenized = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding="longest",
            max_length=self.max_len,
            add_special_tokens=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class DetectionDataModule(pl.LightningDataModule):
    def __init__(self, data_config, project_root: Path):
        super().__init__()
        self.data_config = data_config
        self.project_root = project_root  # Store project root

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None  # Optional test set
        self.tokenizer = AutoTokenizer.from_pretrained(self.data_config.tokenizer_name)
        self.collator = Collator(
            tokenizer=self.tokenizer, max_len=self.data_config.max_len
        )

    def prepare_data(self):
        # This method is called once per node.
        # For this setup, DVC pull is handled in train.py / predict.py scripts
        # other processing done on the fly in dataloaders, so no need for this method
        pass

    def setup(self, stage: str | None = None):
        # Construct full paths to data files
        train_file_path = (
            self.project_root / self.data_config.data_dir / self.data_config.train_file
        )
        test_file_path = (
            self.project_root / self.data_config.data_dir / self.data_config.test_file
        )

        if stage == "fit" or stage is None:
            self.train_dataset = TextDetectionDataset(
                data_path=train_file_path, n_samples=self.data_config.n_samples_train
            )
            self.val_dataset = TextDetectionDataset(
                data_path=test_file_path, n_samples=self.data_config.n_samples_test
            )

        if stage == "test" or stage is None:  # Could be same as val or a different one
            self.test_dataset = TextDetectionDataset(
                data_path=test_file_path, n_samples=self.data_config.n_samples_test
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=True,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.data_config.batch_size,
            shuffle=False,
            num_workers=self.data_config.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )
