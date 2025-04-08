#bagging regressor using scikit-learn Using diabetes

#Implement bagging regressor and classifier using scikit-learn. Use diabetes and iris datasets.

from sklearn.datasets import load_diabetes
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor



def load_data():
    X,y = load_diabetes(return_X_y=True,as_frame=True)
    print(f"x: {X.head()}")
    print(f"x: {X.columns.tolist()}")
    print(f"Y_shape: {y.shape}")
    print(f"X_describe: {X,y.describe()}")
    print(f" missing_values: {X,y.isnull().sum()}")
    return X,y

def split_data(X,y):
    X_train,X_test,Y_train, Y_test = train_test_split(X,y, random_state=30,test_size=0.3)

    return X_train,X_test,Y_train,Y_test


def K_fold(X_train,Y_train):
    k=10
    Kf = KFold(shuffle=True,n_splits=k,random_state=30)
    mean_squared_E=[]
    r2_scores =[]

    for train_index,test_index in Kf.split(X_train):
        X_train_1,x_val_set  = X_train.iloc[train_index],X_train.iloc[test_index]
        Y_train_1,y_val_set = Y_train.iloc[train_index], Y_train.iloc[test_index]

        #scaling data
        scalar = StandardScaler()
        x_train_scaled = scalar.fit_transform(X_train_1)
        X_val_scaled = scalar.transform(x_val_set)

        model = BaggingRegressor(estimator=DecisionTreeRegressor(),n_estimators=50,max_samples=0.6,random_state=40)
        model.fit(x_train_scaled,Y_train_1)

        y_pred = model.predict(X_val_scaled)
        mse = mean_squared_error(y_val_set,y_pred)
        mean_squared_E.append(mse)
        r2 = r2_score(y_val_set,y_pred)
        r2_scores.append(r2)

    mean_acc =  np.mean(mean_squared_E)
    std_acc = np.std(mean_squared_E)
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    print(f" overall mean_acc of mse: {mean_acc} and overall std_dev of mse: {std_acc}")
    print(f" overall mean_acc of r2: {mean_r2} and overall std_dev of r2: {std_r2}")

    return mean_acc, std_acc


def main():
    X,y = load_data()
    X_train, X_test, Y_train, Y_test = split_data(X,y)
    mean_acc, std_acc = K_fold(X_train,Y_train)

    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(X_train)
    Xval_scaled = scaler.transform(X_test)

    model = BaggingRegressor(estimator=None, n_estimators=50, max_samples=0.6, random_state=40)
    model.fit(xtrain_scaled, Y_train)

    # Evaluate on test set
    y_pred = model.predict(Xval_scaled)
    mse = mean_squared_error(Y_test, y_pred)
    r2 = r2_score(Y_test,y_pred)
    print(f"\nFinal Test MSE: {mse:.4f} and r2_score: {r2}")


if __name__ == '__main__':
    main()



