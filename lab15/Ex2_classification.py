from ISLP import load_data
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler


def load_data2():
    weekly_data = load_data('Weekly')
    print(weekly_data.head(10))
    x = weekly_data.drop(columns='Direction')
    y = weekly_data['Direction']
    print(f"X shape: {x.shape}, Y shape: {y.shape}")
    print("Missing values:\n", x.isnull().sum())
    return x, y


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=40)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test


def train_evaluate(x_train_scaled, x_test_scaled, y_train, y_test):
    model = GradientBoostingClassifier(max_depth=3, learning_rate=0.1, loss='log_loss')
    model.fit(x_train_scaled, y_train)

    y_pred = model.predict(x_test_scaled)
    y_proba = model.predict_proba(x_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_proba)

    print(f"Accuracy Score: {acc:.4f}")
    print(f"🔍 Log Loss: {logloss:.4f}")


def main():
    x, y = load_data2()
    x_train_scaled, x_test_scaled, y_train, y_test = split_data(x, y)
    train_evaluate(x_train_scaled, x_test_scaled, y_train, y_test)


if __name__ == '__main__':
    main()
