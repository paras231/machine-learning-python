import numpy
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

#X represents the size of a tumor in centimeters.
X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)

#Note: X has to be reshaped into a column from a row for the LogisticRegression() function to work.
#y represents whether or not the tumor is cancerous (0 for "No", 1 for "Yes").
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logr = linear_model.LogisticRegression()
logr.fit(X,y)


#predict if tumor is cancerous where the size is 3.46mm:
predicted = logr.predict(numpy.array([3.46]).reshape(-1,1))

# print(predicted)

# refer this article for more info about logistic regression -> https://www.w3schools.com/python/python_ml_logistic_regression.asp



# pridict what age person will purchase which product category
df =  pd.read_csv("customer_purchase_data.csv")

# take x as the age 
x1 = df[['Age']].values
#  take y as product category
# which will be calculated using logistic regression
y1 = df['ProductCategory'].values

x_ages =  numpy.array(x1)

logistic_r = linear_model.LogisticRegression()
logistic_r.fit(x1,y1)

predicted_value =  logistic_r.predict([[70]])


print(predicted_value)

# log_odds = logistic_r.coef_

# calculate probablity of which product category by what age
scaler = StandardScaler()
def calculate_probablity(x):
    log_odds = logr.coef_ * x + logr.intercept_
    print("Coefficients:", logistic_r.coef_)
    print("Intercept:", logistic_r.intercept_)
    # print("log_odds:", log_odds)
    odds = numpy.exp(log_odds)
    print(odds,'odds values')
    probability = odds / (1 + odds)
    # print(probability,'prob')
    return(probability)


probability =  calculate_probablity(x_ages)

# print(probability)