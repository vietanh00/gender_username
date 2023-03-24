import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

data_path = './data/gender.csv'


def train(save=False):
    data = pd.read_csv(data_path)
    data['name'] = data['name'].apply(lambda s: str(s).lower())
    data['gender'] = np.where(data['gender'] == 'male', 1, 0)
    data = data[['name', 'gender']].dropna().reset_index(drop=True)
    x_train, x_valid, y_train, y_valid = train_test_split(data['name'],
                                                          data['gender'],
                                                          test_size=0.2,
                                                          random_state=1)
    vectorizer = CountVectorizer(stop_words=['nguyễn', 'trần'],
                                 max_features=1000)
    vectorizer.fit(x_train)
    x_train_vect = vectorizer.transform(x_train)
    x_valid_vect = vectorizer.transform(x_valid)

    rf_clf = RandomForestClassifier()
    rf_clf.fit(x_train_vect, y_train)
    probs = rf_clf.predict_proba(x_valid_vect)[:, 1]
    predi = np.where(probs >= 0.5, 1, 0)

    print(classification_report(y_valid, predi))
    print("AUC: ", roc_auc_score(y_valid, probs))

    if save:
        with open('models/rf_clf.pickle', 'wb') as rf:
            pickle.dump(rf_clf, rf)

        with open('models/vectorizer.pickle', 'wb') as vc:
            pickle.dump(vectorizer, vc)


if __name__ == '__main__':
    train(save=True)
