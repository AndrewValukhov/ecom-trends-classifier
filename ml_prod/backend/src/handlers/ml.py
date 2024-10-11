from typing import Dict

from fastapi import Request
# from sklearn.pipeline import Pipeline
from transformers import pipeline

from core.data import preprocess, predict, explain_prediction
from core.logger import JSONLogger
from schemas.predict_trends import PredictTrendsRequest, PredictTrendsResponse


logger = JSONLogger(__name__)

model = pipeline("text-classification",
                                             'Maldopast/bge-ecom-trends-classifier',
                                             device='cpu',
                                             batch_size=16,
                                             return_all_scores=True,
                                             )

async def predict_trends(request: Request, body: PredictTrendsRequest) -> PredictTrendsResponse:
    # model: pipeline = request.app.state.model
    mapping: Dict = request.app.state.mapping

    logger.info(f"A request has been received with body: {body.data}")

    data = preprocess(body.data)

    prediction = predict(data, model)
    prediction_explains = explain_prediction(mapping, prediction)

    return PredictTrendsResponse(trends_list = prediction_explains)
