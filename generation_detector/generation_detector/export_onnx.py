# src/text_detector/export_onnx.py
import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from .model import DetectionModelPL
from .utils import get_project_root


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def export_to_onnx(cfg: DictConfig) -> None:
    print("Starting ONNX export process...")

    project_root = get_project_root()

    # --- Load Model from Checkpoint ---
    checkpoint_path_str = cfg.model.checkpoint_path
    if not checkpoint_path_str:
        raise (
            ValueError,
            "`model.checkpoint_path` not specified in config. Cannot export.",
        )

    # Check if checkpoint_path is absolute or relative to hydra's output dir or project root
    checkpoint_path = Path(checkpoint_path_str)
    if not checkpoint_path.is_absolute():
        # Try relative to current working directory (Hydra's output dir)
        # Or, more robustly, relative to project root if it's a common pattern
        potential_path_cwd = Path(os.getcwd()) / checkpoint_path_str
        potential_path_root = project_root / checkpoint_path_str

        if potential_path_cwd.exists():
            checkpoint_path = potential_path_cwd
        elif potential_path_root.exists():
            checkpoint_path = potential_path_root
        else:
            raise (
                ValueError,
                f"Checkpoint not found at {checkpoint_path_str} (tried CWD and project root).",
            )

    trained_model = DetectionModelPL.load_from_checkpoint(
        checkpoint_path,
        model_config=cfg.model,  # Pass the model config, hparams might be picked from checkpoint
    )

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
        input_names=["input_ids", "attention_mask"],  # Optional: names for inputs
        output_names=["output_logits"],  # Optional: names for outputs
        dynamic_axes={  # Optional: if you want dynamic batch size or sequence length
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "output_logits": {0: "batch_size"},
        },
    )
    print(f"Model successfully exported to ONNX: {onnx_output_path}")
    print(
        "Remember to add the ONNX model to DVC: `dvc add models/onnx/detector_model.onnx`"
    )


if __name__ == "__main__":
    export_to_onnx()
