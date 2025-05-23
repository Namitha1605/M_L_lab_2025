
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
data = pd.read_csv('data_BC.csv')
print(data.info())

data = data.drop(columns=['id', 'Unnamed: 32'])
data['diagnosis'] = data['diagnosis'].map({'B': 0, 'M': 1})
#print(data['diagnosis'])
X = data.drop(columns=['diagnosis'])
y = data['diagnosis']  # Target (now numeric)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
plt.figure(figsize=(12, 8))
plot_tree(classifier)
plt.show()
