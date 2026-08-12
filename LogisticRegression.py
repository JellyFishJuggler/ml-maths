import pandas as py
import numpy as np
from sympy import symbols,diff
import math

# goal is to create a logistic function using gradient ascent and newton's method
# general idea is that we need a sigmoid function and a function for increasing the likelyhood of feature (theta)


class LogisticRegression:

    def __init__(self, alfa=0.000001, iterations=100000, tolerance=0.0000001):

        self.n = 0
        self.slope = np.array([])
        self.intercept = 0
        self.alfa = alfa
        self.iterations = iterations
        self.tolerance = tolerance
        self.theta = 0

    def _predict(self, input):

        z = np.dot(self.theta, input)
        y_predicted  = self._sigmoid(z)
        return y_predicted
        
    def sum(self, input, theta):

        value = 0
        for i in input:
            value += i
        return 

    def multiple(self, param1, param2):

        value = 1
        for i,j in zip(param1,param2):
            value += i*j
        return value

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # def _error(self,y):
    #     return y - self.slope


    #formula : 
    #   j(theta) = -1/m * summition of (from 1 to m) : y(i)log(predicted_value(i)) + (1-y(i))log(1-predicted_value(i))
    def _costFunction(self,y,y_preditcted,iterations):

        for i in range(iterations):
            j = (-1/i) * (self.sum(self.multiple(math.log(y_preditcted), y)) + self.sum(self.self.multiple(math.log(1-y_preditcted),(1-y))))
        return j

    def _gradientCalculation(self):

        theta = symbols('theta')
        f = self._costFunction(y,y_preditcted=self._predict(X))
        f_prime = diff(f,theta)

        return f_prime

    # forumula : 
        # theta := theta + alfa * (y - ypredicted) * x
    def _updateGradient(self):

        self.theta = self.theta + (self.alfa * self._gradientCalculation())
        self.intercept = self.intercept + (self.alfa * self._gradientCalculation())

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        # Number of input features
        self.n = X.shape[1]

        # One weight for every feature
        self.m = np.zeros(self.n)

        # Initial intercept
        self.c = 0

        previous_cost = float("inf")

        for i in range(self.iterations):

            y_bar = self._predict(X)
            error = self._errorCalculation(y, y_bar)
            cost = np.mean(error ** 2)

            slope, intercept = self._gradientDescentCalculation(X, y, y_bar)
            self.updateParameters(slope, intercept)

            # Stop when cost is no longer changing significantly
            if abs(previous_cost - cost) < self.tolerance:
                break

            previous_cost = cost

        return self

   #fit transform function is required
    def _fitTransform(self,target):

        target_processed = np.unique(target)

        target_map = {
            val: i for i, val in enumerate(target_processed)
        }

        target_encoded = target.map(target_map)

        return target_encoded

        # l = []
        # j = 0
        # t = target.values
        # for i in range(0,5): 
        #     if t[i] != t[j]:
        #         l.append(t[i])
        #     j = j +1
        #     if j == 5:
        #         break
        # return l

        # X = np.asarray(input)
        # y = np.asarray(target)

        # self.n = X.shape[0]

        # return X.shape[0]



df = py.DataFrame({
    "Name" : ["Vanshika","Srijan","Utkarsh","Shreya","Kasak"],
    "Status" : ["Relationship","Relationship","Single","Relationship","Single"]
})

X = df['Name']
y = df['Status']

# print(y.dtype)

print(df)

model = LogisticRegression()
y = model._fitTransform(y)
X = model._fitTransform(X)

print(y)
print(X)   



# z = df.nunique()
# print(z.dtype)
