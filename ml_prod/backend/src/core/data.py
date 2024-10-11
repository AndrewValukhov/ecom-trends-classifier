from typing import Any, Tuple, Dict, List
import pandas as pd
import numpy as np

# from sklearn.pipeline import Pipeline
from transformers import pipeline

def preprocess(data: Any)-> str:
    return str(data)


def predict(data: str, model: pipeline) -> np.ndarray:

    prediction = model(data)[0]
    preds = np.array([d['score'] for d in prediction]) > 0.55
    prob_dict = {d['label']: d['score'] for d in prediction}
    print(prediction)
    print(preds)
    return preds  #, prob_dict

def explain_prediction(mapping: Dict[str, List[str]], prediction: np.ndarray)-> List[List[str]]:
    result = []
    try:
        indices = np.where(prediction == 1)
        for idx in indices[1].tolist():
            result.append(mapping[str(idx)])
    except IndexError:
        pass
    return result