#For the heart.csv dataset, build a logistic regression classifier to predict the risk of heart disease.  
#Vary the threshold to generate multiple confusion matrices.  Implement a python code to calculate the following metrics


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from matplotlib.pyplot import xlabel, ylabel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as cm, accuracy_score, recall_score, precision_score, f1_score, roc_curve,auc

#load the data
def load_data():
    data = pd.read_csv("heart.csv")
    print(f"print the data{data.head(10)}")
    print(f"columns: {data}")
    print(f"{data.isnull().sum()}")
    X = data.drop(columns='target')
    Y = data['target']
    print(f" print the data: {X}")
    print(f"print the data: {Y}")
    return data, X ,Y

def train_test_splits(X,Y):
    x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=50)
    return  x_train,x_test,y_train,y_test
#evaluate and find the confusion matrix
def confusion_matrix(x_train,x_test,y_train,y_test):
    # standard scalar
    scalar = StandardScaler()
    x_scaled_train = scalar.fit_transform(x_train)
    x_scaled_test = scalar.transform(x_test)

    model = LogisticRegression(random_state=50)
    model.fit(x_scaled_train,y_train)
    y_proba = model.predict_proba(x_scaled_test)[:,1]

    thresholds = [0.3, 0.5, 0.7]

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        print(f" Confusion Matrix at Threshold = {threshold} ---")
        cfusion_mtrix = cm(y_test, y_pred)       # cm is used for calculating confusion matrix
        accuracy_scor = accuracy_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        sensetivity = precision_score(y_test,y_pred)
        specificity = cfusion_mtrix[0,0]/cfusion_mtrix[0,0] + cfusion_mtrix[0,1]  # as specificity is not present we need to calculate manualy 
        f1_Score = f1_score(y_test,y_pred)  
        print(f"confusion matrix: {cfusion_mtrix}")
        print(f"Size of accuaracy_score: {accuracy_scor}")
        print(f"Size of recall: {recall}")
        print(f"size of sensetivity: {sensetivity}")
        print(f"the specificity: {specificity}")
        print(f"the f1_score: {f1_Score}")
    frp,trp,threshold = roc_curve(y_test, y_proba)  # use prob to draw the roc curve
    roc_cur = auc(frp,trp)  
    plt.figure(figsize= (9,7))
    plt.plot(frp,trp, label=f'ROC Curve (AUC = {roc_cur})', lw=2, color= 'black')
    plt.plot([0,1] ,[0,1], color = 'red', linestyle= '--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()


def main():
    data,X, Y = load_data()
    x_train, x_test, y_train, y_test = train_test_splits(X, Y)
    confusion_matrix(x_train, x_test, y_train, y_test)


if __name__ == '__main__':
    main()

























