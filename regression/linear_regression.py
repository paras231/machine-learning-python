# understand linear regression using python

# draw a scatter plot/chart

import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)


def draw_regression(x):
    return slope * x + intercept 


mymodel = list(map(draw_regression, x))

plt.scatter(x, y)
plt.plot(x,mymodel)
# plt.show()


# here r is the coefficient of correlation , which is very important for linear regression
#  The r value ranges from -1 to 1, where 0 means no relationship, and 1 (and -1) means 100% related.
# This linear regression can be used to calculate future predictions

#  learn more about linear regression ->  https://www.w3schools.com/python/python_ml_linear_regression.asp


# Now let's predict the future values 

x1 = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y1 = [99,86,87,88,111,86,103,87,94,78,77,85,86]

#  here x is the age of car , y is the speed of car

slope, intercept, r, p, std_err = stats.linregress(x1, y1)


# calculate the speed of 10 years old car

# def calcualte_speed(x):
#     return slope * x + intercept

# speed = calcualte_speed(10)

# # print(speed)

df = pd.read_csv('diabetes.csv')

# Extract X (Age) and y (Production) as arrays
X = df['Age'].values # Reshape to make it a column vector (if needed)
y = df['BloodPressure'].values



slope, intercept, r, p, std_err = stats.linregress(X, y)

print(r,'value of r')