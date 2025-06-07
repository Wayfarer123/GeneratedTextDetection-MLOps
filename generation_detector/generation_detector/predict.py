from pathlib import Path

import hydra
import numpy as np
import onnxruntime
from omegaconf import DictConfig
from transformers import AutoTokenizer

from .utils import get_project_root, pull_dvc_data


def preprocess_text(text: str, tokenizer_name: str, max_len: int):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    encoding = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        padding="max_length",
        max_length=max_len,
        add_special_tokens=True,
    )

    return {
        "input_ids": encoding["input_ids"].astype(
            np.int64
        ),  # Ensure correct type for ONNX
        "attention_mask": encoding["attention_mask"].astype(np.int64),
    }


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(
        x - np.max(x, axis=-1, keepdims=True)
    )  # Subtract max for numerical stability
    return e_x / e_x.sum(axis=-1, keepdims=True)


def perform_prediction(
    text_to_predict: str,
    onnx_model_path: Path,
    tokenizer_name: str,
    max_len: int,
    class_names: dict | None = None,
) -> dict:
    """
    Performs prediction on a single text using a loaded ONNX model.

    Args:
        text_to_predict (str): The input text.
        onnx_model_path (Path): Path to the ONNX model file.
        tokenizer_name (str): Name of the tokenizer.
        max_len (int): Maximum sequence length for the tokenizer.
        class_names (dict): Mapping from class index to class label.

    Returns:
        dict: A dictionary containing the predicted class index, label, and probabilities.
              Example: {"predicted_index": 1, "predicted_label": "AI-generated", "probabilities": [0.1, 0.9]}
    """
    if class_names is None:
        class_names = {
            0: "Human-written",
            1: "AI-generated",
        }
    if not onnx_model_path.exists():
        raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}") from None

    ort_session = onnxruntime.InferenceSession(str(onnx_model_path))

    processed_input = preprocess_text(text_to_predict, tokenizer_name, max_len)
    ort_inputs = {
        "input_ids": processed_input["input_ids"],
        "attention_mask": processed_input["attention_mask"],
    }

    ort_outs = ort_session.run(None, ort_inputs)

    logits = ort_outs[0]
    probabilities = softmax(logits)[0]
    predicted_class_idx = int(np.argmax(probabilities))
    predicted_label = class_names.get(predicted_class_idx, "Unknown Class")

    return {
        "predicted_index": predicted_class_idx,
        "predicted_label": predicted_label,
        "probabilities": probabilities.tolist(),
    }


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def cli_predict_entrypoint(cfg: DictConfig) -> None:
    """
    CLI entry point for making predictions.
    This function handles DVC pull, calls perform_prediction, and prints results.
    It does not return a value, suitable for command-line usage.
    """
    cfg_inference = cfg.inference

    project_root = get_project_root()

    # --- DVC Pull for ONNX model ---
    onnx_model_path_str = cfg_inference.onnx_model_path
    onnx_model_path = Path(onnx_model_path_str)
    # if not onnx_model_path.is_absolute():
    #     onnx_model_path = project_root / onnx_model_path

    # if onnx_model_path.exists():
    #     relative_dvc_target_path = str(onnx_model_path.relative_to(project_root))
    pull_dvc_data(onnx_model_path, dvc_root_path=project_root)

    if not onnx_model_path.is_absolute():
        onnx_model_path = project_root / onnx_model_path_str

    text_to_predict = cfg_inference.text_to_predict
    tokenizer_name = cfg_inference.tokenizer_name
    max_len = cfg_inference.max_len
    class_names_map = {0: "Human-written", 1: "AI-generated"}

    try:
        prediction_result = perform_prediction(
            text_to_predict=text_to_predict,
            onnx_model_path=onnx_model_path,
            tokenizer_name=tokenizer_name,
            max_len=max_len,
            class_names=class_names_map,
        )

        print("\n--- Inference Results ---")
        print(f"Input Text: '{text_to_predict}'")
        print(
            f"Predicted Label: {prediction_result['predicted_label']} (Class {prediction_result['predicted_index']})"
        )
        print("Probabilities:")
        for i, prob in enumerate(prediction_result["probabilities"]):
            print(f"  Class {i} ({class_names_map.get(i, 'Unknown')}): {prob:.4f}")
        print("-------------------------\n")

    except Exception as e:
        raise RuntimeError(
            f"An unexpected error occurred during prediction: {e}"
        ) from e


if __name__ == "__main__":
    cli_predict_entrypoint()
