from typing import List, Dict

from schemas.base_schema import BaseSchema


class PredictTrendsRequest(BaseSchema):
    data: str

class PredictTrendsResponse(BaseSchema):
    trends_list: List[List[str]]
    # prob_dict: Dict