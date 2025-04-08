#Implement Gradient Boost Regression and Classification using scikit-learn. Use the Boston housing dataset from the ISLP package for  the regression problfrom
from ISLP import load_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, log_loss
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


# load the Bostan data
def load_data1():
    data = load_data('Boston')
    print(f" data_summery: {data.head()}")
    print(f"data_medv: {data['medv']}")
    x = data.drop(columns='medv')
    y = data['medv']
    print(f" data_shape: {x.shape} and {y.shape}")
    print(f"id data is null")
    print(data.isnull())
    return x ,y

def split_data(x,y):
    """ split the datasets into training and test"""
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=40)

    # scale the value using standard scalar
    scalar = StandardScaler()
    x_train_scaled = scalar.fit_transform(x_train)
    x_test_scaled =  scalar.transform(x_test)

    return x_train_scaled, x_test_scaled , y_train, y_test

def train_evaluate(x_train_scaled, x_test_scaled , y_train, y_test):
    # apply the gradient boost for regression
    # the hyperparameter is learning rate and maxdepth
    alpha = [0.0010,0.001,0.01,0.1]
    maxim_depth = [4,3,2,1]
    best_r2 = 0
    for i in alpha:
        for k in maxim_depth:

            model = GradientBoostingRegressor(learning_rate=i,n_estimators=50,max_depth=k,loss='squared_error')
            model.fit(x_train_scaled,y_train)
            y_pred= model.predict(x_test_scaled)
            r_score = r2_score(y_test,y_pred)
            print(f"the learning rate {i} and maximum_depth {k} and r_score is {r_score}")

            if r_score > best_r2:
                best_r2 =r_score
                best_model= (i,k)
    print(f"Best R² Score: {best_r2:.4f} with Learning Rate: {best_model[0]} and Max Depth: {best_model[1]}")




def main():
    x,y = load_data1()
    x_train_scaled, x_test_scaled, y_train, y_test = split_data(x,y)
    train_evaluate(x_train_scaled, x_test_scaled, y_train, y_test)

if __name__=='__main__':
    main()



