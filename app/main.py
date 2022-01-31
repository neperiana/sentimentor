from sentiment_classifier import SentimentClassifier

clsfr = SentimentClassifier()

clsfr.predict("I am soooo sad!")
print(clsfr.predictions)