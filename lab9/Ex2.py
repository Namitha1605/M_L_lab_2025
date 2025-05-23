import matplotlib.pyplot
import pandas as pd
import numpy as np
from matplotlib.pyplot import figure
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv("data_ML.csv")
print(data.columns)
X = data[["BP","blood_sugar","BMI","age"]]
Y = data[['disease_score']]
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=50)
r2_scores = []
for fold,(train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    print(f"x_train {X_train} and {X_test}")
    y_train, y_test = Y.iloc[train_index], Y.iloc[test_index]
    print(f"y_train{y_train} and y_test{y_test}")

    model  = DecisionTreeRegressor(random_state=50,max_depth=3)

    model.fit(X_train,y_train)
    y_prd = model.predict(X_test)
    print(f"y_pred {y_prd}")
    r_score = r2_score(y_test,y_prd)
    r2_scores.append(r_score)
    plt.figure(figsize=(15,12))
    plot_tree(model, filled=True, feature_names=X.columns, fontsize=8)
    plt.show()



print(f"rescore: {r2_scores}")
mean = np.mean(r2_scores)
print(f" r2 score: {round(mean,5)}")
sd_deviation = np.std(r2_scores)
print(f"std_deviation: {round(sd_deviation,5)}")
