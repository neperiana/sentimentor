import logging
import flair 
import nltk
from nltk.sentiment import vader

nltk.download('vader_lexicon')


class SentimentClassifier:
    """Classifier implementing sentiment analysis. Includes open source models 
    like flair and vader alongside a homemade one.
    
    Parameters
    ----------
    incl_vader : bool, default=True
        Include VADER sentiment classifier in results? Yes/No.

    incl_flair : bool, default=True
        Include FLAIR sentiment classifier in results? Yes/No.

    incl_homemade : bool, default=True
        Include homemade sentiment classifier in results? Yes/No.

    Attributes
    ----------
    status : str
        Indicates the status of the classifier.
        Possible status include:
            - 

    text : str
        The text to be classified. 

    predictions : list of Dict
        Each dictionary includes the probability of the text bearing negative 
        sentiment and the classification label for a specific model. 
    """

    def __init__(
            self, 
            incl_vader: bool=True, 
            incl_flair: bool=True, 
            incl_homemade: bool=True,
        ):
        # at least one classifier?
        assert incl_vader or incl_flair or incl_homemade, \
            "SentimentClassifier initialised without any models. Include at least one model."

        # settings 
        self._vader = incl_vader
        self._flair = incl_flair
        self._homemade = incl_homemade

        # attributes
        self.status = 'initialised'
        self.text = None
        self.predictions = None


    def _decide_label(self, pos_prob, neg_prob):
        # if one is missing?
        if pos_prob is None:
            return 'negative'
        elif neg_prob is None:
            return 'positive'

        # if not, return bigger
        return 'positive' if pos_prob > neg_prob else 'negative'


    def _update_predictions(self, model_name, pos_prob, neg_prob):
        # updates self.results
        this_prediction = {
            'model': model_name,
            'probs': {'pos': pos_prob, 'neg': neg_prob},
            'label': self._decide_label(pos_prob, neg_prob),
        }

        # is it first model in results?
        if self.predictions is None:
            self.predictions = [this_prediction]
        else:
            self.predictions.append(this_prediction)


    def _predict_vader(self):
        # Predictd sentiment of self.text using VADER

        # loading flair
        vader_clfr = vader.SentimentIntensityAnalyzer()

        # using vader, replace eol and predict sentiment
        clean_text = self.text.replace('\n',' ')
        values = vader_clfr.polarity_scores(clean_text)

        # update results
        self._update_predictions(
            'VADER',
            values['pos'],
            values['neg'],
        )


    def _predict_flair(self):
        # Predictd sentiment of self.text using FLAIR

        # loading flair
        flair_clfr = flair.models.TextClassifier.load('en-sentiment')

        # using flair, convert to a sentence object and predict sentiment
        sentence = flair.data.Sentence(self.text)
        flair_clfr.predict(sentence)
        
        # extract results
        pos_prob = None
        neg_prob = None
        for label in sentence.labels:
            if label.value == 'POSITIVE':
                pos_prob = label.score
            elif label.value == 'NEGATIVE':
                neg_prob = label.score

        # update results
        self._update_predictions(
            'FLAIR',
            pos_prob,
            neg_prob,
        )


    def predict(self, text: str) -> None:
        """Runs sentiment predictions for all models.

        Parameters
        ----------
        text : str
            Text to be classified.
        """
        # validate text
        assert isinstance(text, str), 'Text should be of type string.'

        # throw warning if overwritting, then overwrite
        if self.text:
            logging.warning("Overwritting text to be analysed.")
            self.predictions = None
        self.text = text

        # call models
        self._predict_vader()
        self._predict_flair()