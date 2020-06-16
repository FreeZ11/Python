import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
data1 = pd.read_csv("ratings.csv")
data = data1[data1["movieId"] == 1]
Y = data["rating"]
Y = list(Y)
ycopy = []

for i in Y:
    list1 = [i]
    i = list1
    ycopy.append(i)
Y = ycopy
Y = np.array(Y)

IDs = list(data["userId"])

boolean_series = data1.userId.isin(IDs)
filtered_data = data1[boolean_series]
filtered_data = filtered_data[filtered_data["movieId"] != 1]


for m in [10, 1000, 10000]:
    prev = 1
    X = [] 
    mylist = [0.0]*m
    for index, rows in filtered_data.iterrows(): 
    
        if(rows.userId != prev):
            X.append(mylist)
            mylist = [0.0]*m
            prev = rows.userId
        if(rows.movieId < m):
            mylist[int(rows.movieId)-2] = rows.rating
        if(rows.userId == filtered_data["userId"].iloc[-1] and rows.movieId == filtered_data["movieId"].iloc[-1]):
            X.append(mylist)

    X = np.array(X)

#X = PolynomialFeatures(degree=m, include_bias=False).fit_transform(X)
    model = LinearRegression().fit(X, Y)


#r_sq = model.score(X, Y)
#intercept, coefficients = model.intercept_, model.coef_


    y_pred = model.predict(X)
    helper = []
    for i in range(1,216):
        helper.append(i)
    helper = np.array(helper)
    plt.plot(helper, Y, 'ro', label='real')
    plt.plot(helper, y_pred, 'go', label='predicted')
    k, b = np.polyfit(helper,y_pred,1)
    plt.plot(helper,k*helper + b, color = "y", label='best fit')
    plt.title('linear regression m = {}'.format(m))
    plt.legend()
    plt.show()

for m in [10, 100, 200, 500, 1000, 10000]:
    prev = 1
    X = [] 
    mylist = [0.0]*m
    for index, rows in filtered_data.iterrows(): 
    
        if(rows.userId != prev):
            X.append(mylist)
            mylist = [0.0]*m
            prev = rows.userId
        if(rows.movieId < m):
            mylist[int(rows.movieId)-2] = rows.rating
        if(rows.userId == filtered_data["userId"].iloc[-1] and rows.movieId == filtered_data["movieId"].iloc[-1]):
            X.append(mylist)

    X = np.array(X)

    X_cut = X[:200]
    Y_cut = Y[:200]

    model = LinearRegression().fit(X_cut, Y_cut)
    pred = model.predict(X)

    plt.plot(helper[200:], Y[200:], 'ro', label='real')
    plt.plot(helper[200:], pred[200:], 'go', label='predicted')
    k, b = np.polyfit(helper[200:],pred[200:],1)
    plt.plot(helper[200:],k*helper[200:] + b, color = "y", label='best fit')
    plt.title('linear regression v2 m = {}'.format(m))
    plt.legend()
    plt.show()
