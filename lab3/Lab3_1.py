import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.datasets import fetch_california_housing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier

#"age","BMI","BP","blood_sugar","Gender","disease_score","disease_score_fluct"
def load_data():
    data = pd.read_csv("data_ML.csv")
    x = data.drop(columns=["disease_score","disease_score_fluct"])
    y = data["disease_score_fluct"]
    print(y)
    print(x)
    return x,y

def main():
    # load california housing data
    [x, y] = load_data()

    # split data - train = 70%, test = 30%
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=999)

    # scale the data
    scaler = StandardScaler()
    scaler = scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # train a model
    print("-----TRAINING----")
    # training a linear regression
    model = LinearRegression()
    # train the model
    model.fit(X_train_scaled, y_train)

    # prediction on a test set
    y_pred = model.predict(X_test_scaled)

    # compute the r2 score (r2 score closer to 1 is considered to be good)
    r2 = r2_score(y_test, y_pred)
    plt.scatter(y_test,y_pred)
    plt.show()

    print("r2 score is %0.2f" % r2)
    print("Successfully done!")


if __name__ == "__main__":
    main()




