from fastapi import FastAPI
from typing import List
from .sentiment_classifier import SentimentClassifier
from .schemas import ClassifierInput, ClassifierOutput


# fast api object
app = FastAPI()

# POST /predictions
@app.post('/predictions', response_model=List[ClassifierOutput])
def classify_text(
        input: ClassifierInput,
        #db: Session = Depends(get_db),
    ):
    # Classify text
    clsfr = SentimentClassifier()
    clsfr.predict(input.text)
    preds = clsfr.predictions

    # Save to database
    # new_pred = models.Prediction(**preds)
    # db.add(new_post)
    # db.commit()
    # db.refresh(new_post) # extracts refreshed post

    # Return classification
    return preds

# GET /predictions
