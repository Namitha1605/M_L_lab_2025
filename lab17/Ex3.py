#Implement twitter sentiment prediction using SVM -  Try different kernel functions and compare the results. 



import pandas as pd
import numpy as np
from mlxtend.evaluate import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

def load_data():
    data = pd.read_csv('Tweets.csv')

    # Target variable
    y = data['airline_sentiment'].map({'positive': 1, 'negative': 0, 'neutral': 2})

    # Text vectorization
    vectorizer = TfidfVectorizer(max_features=2000)
    x_textvect = vectorizer.fit_transform(data['text'])

    # Encode categorical columns
    le_airline = LabelEncoder()
    le_reason = LabelEncoder()

    airline = le_airline.fit_transform(data['airline'].astype(str))
    negativereason = le_reason.fit_transform(data['negativereason'].fillna('None'))

    # Numerical columns
    confidence = data['airline_sentiment_confidence'].fillna(0).values
    retweet_count = data['retweet_count'].fillna(0).values

    # Combine all features
    x_other = np.vstack([airline, negativereason, confidence, retweet_count]).T
    x_data = hstack([x_textvect, x_other])

    return x_data, y

def train_test(x_data, y):
    return train_test_split(x_data, y, test_size=0.3, random_state=42)

def svm_for_all_kernal(x_train, x_test, y_train, y_test):
    kernels = ['poly', 'rbf', 'linear']
    accuracy = []
    for kernel in kernels:
        print(f"\nTraining SVM with kernel: {kernel}")
        model = SVC(kernel=kernel)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy.append(acc)
        print(f"Accuracy: {acc:.4f}")

# Main execution
x_data, y = load_data()
x_train, x_test, y_train, y_test = train_test(x_data, y)
svm_for_all_kernal(x_train, x_test, y_train, y_test)
