import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from .model import DetectionModelPL
from .utils import get_project_root


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def export_to_onnx(cfg: DictConfig) -> None:
    print("Starting ONNX export process...")

    project_root = get_project_root()

    # --- Load Model from Checkpoint ---
    checkpoint_path = Path(os.getcwd()) / "checkpoints" / "best_checkpoint.ckpt"

    print(f"\n\n{checkpoint_path}\n\n")

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


if __name__ == "__main__":
    export_to_onnx()
