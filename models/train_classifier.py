import time
import sys
from sqlalchemy import create_engine
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,TfidfTransformer
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
import pickle

# Detect the language of only the first comment
nltk.download('stopwords')
nltk.download('punkt')
language = "english"
STOPWORDS = set(stopwords.words(language))
PUNCT_TO_REMOVE = string.punctuation



def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))


# Stemmer
stemmer = PorterStemmer()
def stem_word(word):
    """ stem words with Porter """
    try:
        return stemmer.stem(word)
    except RecursionError:
        return ""


def load_data(database_filepath):
    """load data from database"""
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table("emergency", engine)
    X = df.message.values
    y = df.loc[:,"related":].values
    return X,y


def tokenize(text):
    """ Tokenize text to words and clean it"""
    word_list = word_tokenize(text)
    word_list = [word.lower() for word in word_list]
    word_list = [word for word in word_list if word not in STOPWORDS]
    word_list = [remove_punctuation(word) for word in word_list if len(word)>1]
    word_list = [stem_word(word) for word in word_list if len(word)>1]
    # text = " ".join(word_list)
    return word_list


def build_model():
    """ model pipeline with grid search"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ("clf", MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))
    ])

    param_grid = {
        'vect__max_df':[1.0],
        'vect__min_df':[1],
        'vect__max_features':[1000],
#        'tfidf__use_idf':(True,False),
    }

    model = GridSearchCV(
        pipeline,
        param_grid,
        cv=2,
        n_jobs=-1,
        verbose=2,
    )

    return model

def evaluate_model(model, X_test, Y_test):
    """ print out the classification report for every parameter"""
    y_pred = model.predict(X_test)
    for i in range(0, len(Y_test[1, :])):
        print(classification_report(Y_test[:, i], y_pred[:, i]))


def save_model(model, model_filepath):
    """ save models"""
    with open(f'{model_filepath}', 'wb') as handle:
        pickle.dump(model, handle)

def main():
    """ execute the training"""
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
