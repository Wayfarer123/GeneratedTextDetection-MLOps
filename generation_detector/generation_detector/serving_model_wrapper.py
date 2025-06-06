import json
from pathlib import Path

import numpy as np
import onnxruntime
import pandas as pd
from transformers import AutoTokenizer


class TextClassificationModel:
    def __init__(self, tokenizer, ort_session, max_len, input_names, output_names):
        self.tokenizer = tokenizer
        self.ort_session = ort_session
        self.max_len = max_len
        self.input_names = input_names
        self.output_names = output_names
        print("TextClassificationModel initialized.")
        print(f"  Tokenizer: {type(self.tokenizer)}")
        print(f"  ORT Session: {type(self.ort_session)}")
        print(f"  Max len: {self.max_len}")
        print(f"  Input names: {self.input_names}")
        print(f"  Output names: {self.output_names}")

    def _preprocess(self, model_input_df: pd.DataFrame):
        if "text" not in model_input_df.columns:
            raise ValueError("Input DataFrame must contain a 'text' column.")
        texts = model_input_df["text"].tolist()

        inputs = self.tokenizer(
            texts,
            return_tensors="np",
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )

        ort_inputs = {}
        if "input_ids" in self.input_names:
            ort_inputs["input_ids"] = inputs["input_ids"]
        if "attention_mask" in self.input_names:
            ort_inputs["attention_mask"] = inputs["attention_mask"]
        return ort_inputs

    def _postprocess(self, onnx_outputs):
        logits = onnx_outputs[0]
        predicted_indices = np.argmax(logits, axis=1)
        return pd.DataFrame(predicted_indices, columns=["prediction"])

    def predict(self, model_input, params=None):
        print(
            f"TextClassificationModel.predict: Received model_input of type: {type(model_input)}"
        )
        if isinstance(model_input, dict) and "dataframe_records" in model_input:
            model_input_df = pd.DataFrame(model_input["dataframe_records"])
        elif isinstance(model_input, dict) and "dataframe_split" in model_input:
            model_input_df = pd.DataFrame(
                model_input["dataframe_split"]["data"],
                columns=model_input["dataframe_split"]["columns"],
            )
        elif isinstance(model_input, pd.DataFrame):
            model_input_df = model_input
        else:
            if isinstance(model_input, list) and all(
                isinstance(i, str) for i in model_input
            ):
                model_input_df = pd.DataFrame({"text": model_input})
            elif isinstance(model_input, dict) and "text" in model_input:
                model_input_df = pd.DataFrame(model_input)
            else:
                raise ValueError(
                    "Expected Pandas DataFrame, list of strings, or dict with 'text' or "
                    "'dataframe_records'/'dataframe_split' keys as input, "
                    f"got {type(model_input)}"
                )

        if "text" not in model_input_df.columns:
            raise ValueError(
                "Input data after conversion must contain a 'text' column."
            )

        processed_input = self._preprocess(model_input_df)
        onnx_outputs = self.ort_session.run(self.output_names, processed_input)
        predictions_df = self._postprocess(onnx_outputs)
        return predictions_df


def _load_pyfunc(model_uri: str):
    print(f"_load_pyfunc called with model_uri: {model_uri}")

    onnx_model_file_path = str(Path(model_uri) / "model.onnx")
    tokenizer_dir_path = str(Path(model_uri) / "tokenizer_files")

    print(f"  Attempting to load ONNX model from: {onnx_model_file_path}")
    print(f"  Attempting to load tokenizer from: {tokenizer_dir_path}")

    hparams_file = Path(tokenizer_dir_path) / "hparams.json"
    with open(hparams_file) as f:
        hparams = json.load(f)
    max_len = hparams["max_len"]

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir_path)
    ort_session = onnxruntime.InferenceSession(onnx_model_file_path)

    input_names = [inp.name for inp in ort_session.get_inputs()]
    output_names = [out.name for out in ort_session.get_outputs()]

    return TextClassificationModel(
        tokenizer=tokenizer,
        ort_session=ort_session,
        max_len=max_len,
        input_names=input_names,
        output_names=output_names,
    )
