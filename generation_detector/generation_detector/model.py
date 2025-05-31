import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchmetrics
from transformers import AutoModel


class BertClassifierHead(nn.Module):
    """Classifier head for BERT-like model"""

    def __init__(
        self,
        bert_output_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float,
    ):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(bert_output_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, cls_embedding):
        return self.classifier(cls_embedding)


class DetectionModelPL(pl.LightningModule):
    def __init__(self, model_config):
        super().__init__()
        self.save_hyperparameters(model_config)

        self.backbone = AutoModel.from_pretrained(model_config.pretrained_model_path)

        self.clf_head = BertClassifierHead(
            bert_output_dim=self.backbone.encoder.layer[1].output.dense.out_features,
            hidden_dim=model_config.cls_hidden_dim,
            output_dim=model_config.output_dim,
            dropout_rate=model_config.dropout_rate,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.train_accuracy = torchmetrics.Accuracy(
            task="binary" if model_config.output_dim == 2 else "multiclass",
            num_classes=model_config.output_dim,
        )
        self.val_accuracy = torchmetrics.Accuracy(
            task="binary" if model_config.output_dim == 2 else "multiclass",
            num_classes=model_config.output_dim,
        )
        self.test_accuracy = torchmetrics.Accuracy(
            task="binary" if model_config.output_dim == 2 else "multiclass",
            num_classes=model_config.output_dim,
        )

    def forward(self, input_ids, attention_mask):
        # Ensure 2D input for backbone
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)

        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state shape: (batch_size, sequence_length, hidden_size)
        # We need the [CLS] token embedding, which is the first token's embedding
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.clf_head(cls_embedding)
        return logits

    def common_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        logits = self(input_ids, attention_mask)
        loss = self.criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, labels

    def training_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch, batch_idx)
        self.train_accuracy(preds, labels)

        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_acc",
            self.train_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch, batch_idx)
        self.val_accuracy(preds, labels)

        self.log(
            "val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "val_acc",
            self.val_accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, preds, labels = self.common_step(batch, batch_idx)
        self.test_accuracy(preds, labels)

        self.log("test_loss", loss, on_step=False, on_epoch=True, logger=True)
        self.log(
            "test_acc", self.test_accuracy, on_step=False, on_epoch=True, logger=True
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.learning_rate,
        )
        return optimizer

    def on_train_epoch_end(self):
        pass

    def on_validation_epoch_end(self):
        pass
