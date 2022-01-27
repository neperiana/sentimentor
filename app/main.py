from sentiment_classifier import SentimentClassifier

clsfr = SentimentClassifier()

clsfr.predict("Been feeling quite miserable for a bit. Although I am also happy.")
print(clsfr.predictions)