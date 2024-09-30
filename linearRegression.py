#supervised learning
#Model: linear regression

# AREA DATASET

import numpy as np # good for linear algebra and for eigen values
import pandas as pd # supports 2 datatypes series(1D) and dataframes (2D)

#common imports in most models
#scikit learn library
from sklearn.linear_model import LinearRegression # formula of linear regression is imported
from sklearn.model_selection import train_test_split # trained data and test data for pareto principle 80:20 proportion
from sklearn.metrics import mean_squared_error # to determine the difference between expected outcome and observed outcome

#sample data
data = {
    'Squarefoot': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200],
    'Price': [50000, 52000, 60000, 60600, 70600, 70890, 71500, 71800]
}

df = pd.DataFrame(data)
print(df)

# set the target column
# X will be input always
# y will be output always
X = df[['Squarefoot']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) # X is input array, y is output array, testing data is 20% therefore 0.2, select random 80 values for training those should remain constant that is fixed for all iterations therefore 0

#construct the model
model = LinearRegression()
model.fit(X_train, y_train) # training parameters

#predictions
y_pred = model.predict(X_test) # based on training data

#evluation of our model
mse = mean_squared_error(y_test, y_pred)

#output
print(f'Mean Squared Error: {mse}') # f is for format
print(f'Predicted prices: {y_pred}')
print(f'Actual prices: {y_test}') 

#------------------------------------------------------------------------------------------------------
# the code has some error
# user_input = float(input("Enter a square footage whose price you wish to predict"))

# user_prediction = model.predict([[user_input]])

# print(f'Predicted price for {user_input} square footage: {user_prediction[0]}')

#------------------------------------------------------------------------------------------------------




# LIKES AND COMMENTS DATASET

import numpy as np # good for linear algebra and for eigen values
import pandas as pd # supports 2 datatypes series(1D) and dataframes (2D)
from math import sqrt

#common imports in most models
#scikit learn library
from sklearn.linear_model import LinearRegression # formula of linear regression is imported
from sklearn.model_selection import train_test_split # trained data and test data for pareto principle 80:20 proportion
from sklearn.metrics import mean_squared_error # to determine the difference between expected outcome and observed outcome

#sample data
data = {
    'Likes': [100, 150, 200, 250, 300, 350],
    'Comments': [20, 30, 40, 50, 60, 70],
    'Views': [1000, 1500, 2000,  2500, 3000, 3500]
}

df = pd.DataFrame(data)
print(df)

X = df[['Likes', 'Comments']]
y = df['Views']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# After your predictions
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

#output
print(f'Mean Squared Error: {mse}') # f is for format
print(f'Root mean squared error is: {rmse}')
print(f'Predicted views: {y_pred}')
print(f'Actual views: {y_test}')

# incorrect code
# user_input = float(input("Enter likes and comments you wish to predict"))

# user_prediction = model.predict([[user_input]])

# print(f'Predicted price for {user_input} square footage: {user_prediction[0]}')



#--------------------------------------------------------------------------------------------------------



#TV ADVERTISEMENT AND SALES DATASET

import numpy as np # good for linear algebra and for eigen values
import pandas as pd # supports 2 datatypes series(1D) and dataframes (2D)

#common imports in most models
#scikit learn library
from sklearn.linear_model import LinearRegression # formula of linear regression is imported
from sklearn.model_selection import train_test_split # trained data and test data for pareto principle 80:20 proportion
from sklearn.metrics import mean_squared_error # to determine the difference between expected outcome and observed outcome

#sample data
data = {
    'tv_advt_invested': [10000, 15000, 20000, 25000, 30000, 35000],
    'radio_ad_spend': [2000, 3000, 4000, 5000, 6000, 7000],
    'Profit': [1000, 1500, 2000,  2500, 3000, 3500]
}

df = pd.DataFrame(data)
print(df)

X = df[['tv_advt_invested', 'radio_ad_spend']]
y = df['Profit']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

#output
print(f'Mean Squared Error: {mse}') # f is for format
print(f'Predicted profit: {y_pred}')
print(f'Actual profit: {y_test}')

