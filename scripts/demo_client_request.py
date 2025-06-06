import json

import pandas as pd
import requests

MLFLOW_SERVING_URL = "http://127.0.0.1:5001/invocations"


def send_prediction_request(text_list):
    df = pd.DataFrame({"text": text_list})

    payload = {"dataframe_split": df.to_dict(orient="split")}

    if "index" in payload["dataframe_split"]:
        del payload["dataframe_split"]["index"]

    headers = {"Content-Type": "application/json"}

    print(
        f"Sending request to {MLFLOW_SERVING_URL} with payload: {json.dumps(payload, indent=2)}"
    )
    try:
        response = requests.post(
            MLFLOW_SERVING_URL, data=json.dumps(payload), headers=headers
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response content: {e.response.text}")
        return None

    try:
        response_data = response.json()
        print(f"Raw response from server: {response_data}")
        if isinstance(response_data, dict) and "predictions" in response_data:
            predictions = response_data["predictions"]
            return predictions
        elif isinstance(response_data, list):
            return response_data
        else:
            print(
                f"Unexpected response format. Could not extract predictions. Response: {response_data}"
            )
            return None

    except json.JSONDecodeError:
        print(f"Failed to decode JSON response: {response.text}")
        return None
    except Exception as e:
        print(f"Error processing response: {e}")
        return None


if __name__ == "__main__":
    texts_to_classify = [
        "This is a sample textfor prediction. Let's see if the model thinks it's AI-generated.",
        "Another sample just to check correct batch processing.",
    ]

    predictions = send_prediction_request(texts_to_classify)

    if predictions is not None:
        print("\n--- Predictions ---")
        for text, pred in zip(texts_to_classify, predictions, strict=False):
            label = "AI-generated" if pred == 1 else "Human-written"
            print(f'Text: "{text}" \n  -> Prediction: {pred} ({label})\n')
