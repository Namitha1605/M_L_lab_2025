# Random Forest algorithm for regression 

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data():
    """Load the Diabetes dataset and return features and target values."""
    x, y = load_diabetes(return_X_y=True, as_frame=True)
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
    """Perform K-Fold cross-validation using Random Forest Regressor."""
    k = 10
    kf = KFold(n_splits=k, shuffle=True, random_state=30)
    r2_scores = []
    mse_values = []

    for train_index, val_index in kf.split(x_train):
        x_train1, x_val1 = x_train[train_index], x_train[val_index]
        y_train1, y_val1 = y_train.iloc[train_index], y_train.iloc[val_index]

        model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=30)
        model.fit(x_train1, y_train1)

        y_pred = model.predict(x_val1)
        mse_values.append(mean_squared_error(y_val1, y_pred))
        r2_scores.append(r2_score(y_val1, y_pred))

    mean_mse = np.mean(mse_values)
    std_dev_mse = np.std(mse_values)
    print(f"K-Fold Cross-Validation -> Mean MSE: {mean_mse:.4f}, Std Dev: {std_dev_mse:.4f}")

    mean_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    print(f"K-Fold Cross-Validation -> Mean R²: {mean_r2:.4f}, Std Dev: {std_dev_r2:.4f}")

    return mean_r2, std_dev_r2

def train_and_evaluate(x_train, x_test, y_train, y_test):
    """Train the Random Forest model and evaluate on the test set."""
    model = RandomForestRegressor(n_estimators=50, max_depth=4, random_state=30)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test,y_pred)
    r2_score_value = r2_score(y_test, y_pred)
    print(f"Test Set R² Score (70:30 split): {r2_score_value:.4f}")
    print(f"Test Set mse Score (70:30 split): {mse:.4f}")

def main():
    x, y = load_data()
    x_train, x_test, y_train, y_test = split_data(x, y)

    kfold(x_train, y_train)
    train_and_evaluate(x_train, x_test, y_train, y_test)

if __name__ == '__main__':
    main()
