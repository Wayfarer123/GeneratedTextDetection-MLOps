import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from .data_module import DetectionDataModule
from .model import DetectionModelPL
from .utils import get_project_root, pull_dvc_data


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def train(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed, workers=True)

    project_root = get_project_root()

    # --- DVC Data Pull ---
    # Construct full paths for DVC targets relative to project_root
    # This assumes data.train_file and data.test_file are like "raw/filename.jsonl"
    # and data.data_dir is like "data"
    train_dvc_target = Path(cfg.data.data_dir) / cfg.data.train_file
    test_dvc_target = Path(cfg.data.data_dir) / cfg.data.test_file

    pull_dvc_data(train_dvc_target, test_dvc_target, dvc_root_path=project_root)

    # --- DataModule ---
    # Pass project_root to DataModule so it can construct absolute paths
    datamodule = DetectionDataModule(data_config=cfg.data, project_root=project_root)
    datamodule.setup(stage="fit")

    # --- Model ---
    model = DetectionModelPL(model_config=cfg.model)

    # --- Callbacks ---
    callbacks = []
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            os.getcwd(), "checkpoints"
        ),  # Saves checkpoints in hydra's output dir
        filename="{epoch}-{val_loss:.2f}-{val_acc:.2f}",
        monitor="val_acc",  # Monitor validation accuracy
        mode="max",  # Save the model with the highest validation accuracy
        save_top_k=1,
    )
    callbacks.append(checkpoint_callback)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    callbacks.append(lr_monitor)

    mlf_logger = MLFlowLogger(
        experiment_name=cfg.logging.experiment_name,
        tracking_uri=cfg.logging.tracking_uri,
        log_model=cfg.logging.log_model,
        artifact_location=None,
        run_name=None,
    )
    # Manually log hydra config to MLflow
    # MLFlowLogger automatically logs hparams, but full config can be useful
    mlf_logger.experiment.log_dict(
        mlf_logger.run_id,
        OmegaConf.to_container(cfg, resolve=True),
        "hydra_config.yaml",
    )

    # --- Trainer ---
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        logger=mlf_logger,
        callbacks=callbacks,
        precision=cfg.training.precision,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.training.log_every_n_steps,
        deterministic=True,
    )
    print("Start training...")
    trainer.fit(model, datamodule=datamodule)
    print("Model trained")

    trainer.test(
        model, datamodule=datamodule
    )  # Loads best checkpoint by default if model path not given

    print(f"Best model checkpoint path: {checkpoint_callback.best_model_path}")
    # Log the best model path to MLflow as an artifact or param
    if checkpoint_callback.best_model_path:
        mlf_logger.experiment.log_param(
            mlf_logger.run_id,
            "best_checkpoint_path_local",
            checkpoint_callback.best_model_path,
        )

    # Make sure all logs are flushed
    mlf_logger.finalize("success")


if __name__ == "__main__":
    train()
