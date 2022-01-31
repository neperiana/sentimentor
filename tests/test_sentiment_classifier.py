import pytest
import numpy as np
from app.sentiment_classifier import SentimentClassifier
from app import schemas


@pytest.mark.parametrize(
    "text, expected_sentiment",
    [
        ('happy', 'positive'),
        ('sad', 'negative'),
    ],
)
def test_expected_label(text, expected_sentiment):
    # using classifier
    clsfr = SentimentClassifier()
    clsfr.predict(text)
    preds = clsfr.predictions

    # output schema is as expected
    posts = [schemas.ClassifierOutput(**dct) for dct in preds]

    # all labels are correct
    return np.all(np.array([dct['label'] == expected_sentiment for dct in preds]))
