Implement bagging classifier using scikit-learn with iris datasets.

from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import mean_squared_error,accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def load_data():
    X,y = load_iris(return_X_y=True,as_frame=True)
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
    accuarcy_scores =[]

    for train_index,test_index in Kf.split(X_train):
        X_train_1,x_val_set  = X_train.iloc[train_index],X_train.iloc[test_index]
        Y_train_1,y_val_set = Y_train.iloc[train_index], Y_train.iloc[test_index]

        #scaling data
        scalar = StandardScaler()
        x_train_scaled = scalar.fit_transform(X_train_1)
        X_val_scaled = scalar.transform(x_val_set)

        model = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=50,max_samples=0.6,random_state=40)
        model.fit(x_train_scaled,Y_train_1)

        y_pred = model.predict(X_val_scaled)
        acc = accuracy_score(y_val_set,y_pred)
        accuarcy_scores.append(acc)

    mean_acc = np.mean(accuarcy_scores)
    std_acc = np.std(accuarcy_scores)
    print(f" overall mean_acc for k fold  : {mean_acc} and overall std_dev : {std_acc}")

    return mean_acc, std_acc


def main():
    X,y = load_data()
    X_train, X_test, Y_train, Y_test = split_data(X,y)
    mean_acc, std_acc = K_fold(X_train,Y_train)

    scaler = StandardScaler()
    xtrain_scaled = scaler.fit_transform(X_train)
    Xval_scaled = scaler.transform(X_test)

    model = BaggingClassifier(estimator=None, n_estimators=50, max_samples=0.6, random_state=40)
    model.fit(xtrain_scaled, Y_train)

    # Evaluate on test set
    y_pred = model.predict(Xval_scaled)
    accc = accuracy_score(Y_test,y_pred)
    print(f"\nFinal Test accuracy: {accc:.4f}")


if __name__ == '__main__':
    main()



