import pandas as py
import numpy as np

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

    def _predict(self, X, val):

        new_val = 0 - val
        sig = 1 + (2.72) ** np.dot(new_val, X)
        y_predicted = (float)(1/sig)
        return y_predicted
    
    def sum(self, input):

        value = 0
        for i in input:
            value += i
        return value

    def _sigmoid(self, theta, input):
        pass

    def _error(self,y):
        return y - self.slope

    def _costFunction(self,alfa,error,input):
        pass


    def _recalculate(self,input,target,df=py.DataFrame):
        pass

    #     count = 0
    #     t = target.values
    #     print('t assigned')
    #     uni = df.nunique()
    #     print(uni.dtype)
    #     print('unique assigned successfully')
    #     for i in [0,uni]:
    #         if t[i+1] == t[i]:
    #             count += 1
    #     return count

    #fit transform function is required
    def _fitTransform(self,input,target):

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

        X = np.asarray(input)
        y = np.asarray(target)

        self.n = X.shape[0]

        return X.shape[0]



df = py.DataFrame({
    "Name" : ["Vanshika","Srijan","Utkarsh","Shreya","Kasak"],
    "Status" : ["Relationship","Relationship","Single","Relationship","Single"]
})

X = df['Name']
y = df['Status']

y = y.replace('Relationship',1)
y = y.replace('Single',0)

# print(y.dtype)

model = LogisticRegression()

print(df)
# print(model._error(y))
print(model._predict(X,""))
# print(model._fitTransform(y,df))
# print(model._recalculate(X,y))

# z = df.nunique()
# print(z.dtype)
