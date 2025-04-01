Implement Random Forest algorithm for classification using scikit-learn, using iris datasets.


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np


def load_data():
    """Load the Iris dataset and return features and target values."""
    x, y = load_iris(return_X_y=True, as_frame=True)
    print(f"Dataset shape: {x.shape}, Target shape: {y.shape}")
    print("Missing values per column:\n", x.isnull().sum())
    return x, y


def split_data(x, y):
    """Split the dataset into training and test sets and scale the data."""
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=30)

    # Standardization
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test


def kfold(x_train, y_train):
    """Perform K-Fold cross-validation using Random Forest Classifier."""
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=30)
    accuracy_scores = []

    for train_index, val_index in kf.split(x_train):
        x_train1, x_val1 = x_train[train_index], x_train[val_index]
        y_train1, y_val1 = y_train.iloc[train_index], y_train.iloc[val_index]

        model = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=4, random_state=30)
        model.fit(x_train1, y_train1)

        y_pred = model.predict(x_val1)
        accuracy_scores.append(accuracy_score(y_val1, y_pred))

    mean_acc = np.mean(accuracy_scores)
    std_dev = np.std(accuracy_scores)
    print(f"K-Fold Cross-Validation -> Mean Accuracy: {mean_acc:.4f}, Std Dev: {std_dev:.4f}")

    return mean_acc, std_dev


def train_and_evaluate(x_train, x_test, y_train, y_test):
    """Train the Random Forest model and evaluate on test set."""
    model = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=4, random_state=30)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Set Accuracy (70:30 split): {accuracy:.4f}")


def main():
    x, y = load_data()
    x_train, x_test, y_train, y_test = split_data(x, y)

    kfold(x_train, y_train)
    train_and_evaluate(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()
