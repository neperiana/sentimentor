import spacy

nlp = spacy.load('en_core_web_sm')

def create_data(dataset, is_train=True):
    # Getting all text into a python list
    texts = list(dataset['text'].values)
                 
    # Put the list into the nlp pipeline and converting the output into a list
    preprocessed_texts = list(nlp.pipe(texts))

    # Getting vectors for all texts 
    X = [string.vector  for string in preprocessed_texts]


    if is_train:
        # Labels for the corrosponding texts 
        y = dataset['label'].tolist()

        return X, y

    else:
        return X