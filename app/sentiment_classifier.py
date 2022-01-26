import vader, flair 

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


    def predict(text: str) -> None:
        """Runs sentiment predictions for all models.

        Parameters
        ----------
        text : str
            Text to be classified.
        """
        pass

