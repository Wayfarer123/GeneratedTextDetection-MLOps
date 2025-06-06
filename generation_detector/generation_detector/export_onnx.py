import json
import os
import shutil
import tempfile
from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

from .model import DetectionModelPL
from .utils import get_project_root


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def export_to_onnx(cfg: DictConfig) -> None:
    print("Starting ONNX export process...")

    project_root = get_project_root()

    # --- Load Model from Checkpoint ---
    checkpoint_path = Path(os.getcwd()) / "checkpoints" / "best_checkpoint.ckpt"

    hparams_overrides = OmegaConf.to_container(cfg.model, resolve=True)
    _ = hparams_overrides.pop("checkpoint_path")  # kind of kostyl

    trained_model = DetectionModelPL.load_from_checkpoint(
        checkpoint_path, **hparams_overrides
    )
    trained_model.to("cpu")
    trained_model.eval()

    # --- Prepare Dummy Input ---
    # This needs to match the expected input shape for the model's forward method
    # batch_size, seq_len
    batch_size = cfg.export.input_sample_batch_size
    seq_len = cfg.data.max_len

    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    dummy_input = (dummy_input_ids, dummy_attention_mask)  # Tuple for forward method

    # --- Define ONNX Output Path ---
    onnx_output_path_str = cfg.export.onnx_output_path
    onnx_output_path = (
        project_root / onnx_output_path_str
    )  # Assume relative to project root
    onnx_output_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Create directory if it doesn't exist

    # --- Export ---
    torch.onnx.export(
        trained_model,
        dummy_input,
        str(onnx_output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input_ids", "attention_mask"],
        output_names=["output_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output_logits": {0: "batch_size"},
        },
    )
    print(f"Model successfully exported to ONNX: {onnx_output_path}")
    print(
        "Remember to add the ONNX model to DVC: `dvc add models/onnx/detector_model.onnx`"
    )

    mlflow_run_id = cfg.export.get("mlflow_run_id_for_onnx_log")

    if mlflow_run_id:
        print(f"Logging ONNX model to MLflow run_id: {mlflow_run_id}")
        mlflow.set_tracking_uri(cfg.logging.tracking_uri)

        tokenizer_name_or_path = cfg.model.pretrained_model_path
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        with tempfile.TemporaryDirectory() as data_for_pyfunc_model_dir:
            data_for_pyfunc_model_path = Path(data_for_pyfunc_model_dir)
            onnx_filename_in_data_path = "model.onnx"
            shutil.copy(
                str(onnx_output_path),
                str(data_for_pyfunc_model_path / onnx_filename_in_data_path),
            )

            tokenizer_subdir_in_data_path = "tokenizer_files"
            path_to_tokenizer_in_data_path = (
                data_for_pyfunc_model_path / tokenizer_subdir_in_data_path
            )

            tokenizer.save_pretrained(path_to_tokenizer_in_data_path)
            max_len_value = cfg.data.max_len
            hparams_for_pyfunc = {"max_len": max_len_value}

            hparams_file = path_to_tokenizer_in_data_path / "hparams.json"
            with open(hparams_file, "w") as f:
                json.dump(hparams_for_pyfunc, f)

            pyfunc_model_artifact_path_in_mlflow = "onnx_with_tokenizer_for_serving"

            input_example_df = pd.DataFrame(
                {"text": ["This is an example sentence for signature inference."]}
            )

            wrapper_module_name = "generation_detector.serving_model_wrapper"
            code_paths = [
                str(project_root / "generation_detector" / "generation_detector")
            ]

            with mlflow.start_run(run_id=mlflow_run_id, nested=True) as run:
                print(f"Active MLflow run for PyFunc logging: {run.info.run_id}")
                mlflow.pyfunc.log_model(
                    artifact_path=pyfunc_model_artifact_path_in_mlflow,
                    loader_module=wrapper_module_name,
                    code_paths=code_paths,
                    data_path=str(data_for_pyfunc_model_path),
                    input_example=input_example_df,
                    registered_model_name=cfg.model.name + "_serving_pipeline",
                )
                print(
                    f"PyFunc model logged to MLflow under artifact path: {pyfunc_model_artifact_path_in_mlflow}"
                )

        # with mlflow.start_run(run_id=mlflow_run_id, nested=True) as run:
        #     print(f"Active MLflow run for ONNX logging: {run.info.run_id}")

        #     loaded_onnx_model = onnx.load(str(onnx_output_path))
        #     onnx_model_artifact_path_in_mlflow = "onnx_model_for_serving"
        #     model_info = mlflow.onnx.log_model(
        #         onnx_model=loaded_onnx_model,
        #         artifact_path=onnx_model_artifact_path_in_mlflow,
        #         registered_model_name=cfg.model.name + "_onnx",
        #     )

        #     print(
        #         f"ONNX model logged to MLflow under artifact path: {onnx_model_artifact_path_in_mlflow}"
        #     )

        #     registered_model_name = cfg.model.name + "_onnx"

        #     client = MlflowClient(tracking_uri=cfg.logging.tracking_uri)
        #     model_version = client.get_latest_versions(
        #         name=registered_model_name, stages=["None"]
        #     )[0].version

        #     original_run_id = model_info.run_id

        #     tokenizer_artifact_subpath = (
        #         f"tokenizers_for_models/{registered_model_name}/{model_version}"
        #     )

        #     with tempfile.TemporaryDirectory() as tmp_tokenizer_dir:
        #         tokenizer.save_pretrained(tmp_tokenizer_dir)
        #         mlflow.log_artifacts(
        #             tmp_tokenizer_dir, artifact_path=tokenizer_artifact_subpath
        #         )
        #         print(
        #             f"Tokenizer for '{registered_model_name}' v{model_version} logged to run '{original_run_id}' under artifact path: '{tokenizer_artifact_subpath}'"
        #         )

    else:
        print(
            "`export.mlflow_run_id_for_onnx_log` not provided. Skipping MLflow ONNX model logging."
        )


if __name__ == "__main__":
    export_to_onnx()
