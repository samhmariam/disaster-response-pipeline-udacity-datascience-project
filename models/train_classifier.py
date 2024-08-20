import sys
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import re

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


def load_data(database_filepath):
    engine = create_engine(f"sqlite:///{database_filepath}")
    df = pd.read_sql_table("classified_msgs", con=engine)
    X = df["message"]
    Y = df.drop(columns=["id", "message", "original", "genre"])
    category_names = Y.columns.tolist()

    return X, Y, category_names


def tokenize(text):
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stopwords.words("english")]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return clean_tokens


def build_model():
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            ("clf", MultiOutputClassifier(RandomForestClassifier(random_state=42))),
        ]
    )

    parameters = {
        "tfidf__use_idf": [False],
        "clf__estimator__max_depth": [None]
    }

    model = GridSearchCV(pipeline, param_grid=parameters, verbose=1, n_jobs=-1, cv=2)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    for i, category in enumerate(category_names):
        print(f"Category: {category}")
        print(classification_report(Y_test[category], Y_pred[:, i]))
        print("\n")


def save_model(model, model_filepath):
    with open(model_filepath, "wb") as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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