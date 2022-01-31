from pydantic import BaseModel
from typing import Literal, Optional

# Schema for api input
class ClassifierInput(BaseModel):
    text: str

## schema for Sentiment Classifier Output
class Probabilities(BaseModel):
    pos: Optional[float]
    neg: Optional[float]

class ClassifierOutput(BaseModel):
    model: str
    probs: Probabilities
    label: Literal['positive', 'negative']