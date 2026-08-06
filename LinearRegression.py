# tsk is to make a function and do prediction over a dataset using simple linear regression model
# basic funda is we need 2 variables - y and x, and a few functions for calculating intercept and slope of the curve.
# m = summition of (x-x_mean).(y-y_mean) / summition of (x-x_mean)^2
# c = y_mean - m.x_mean

import pandas as py
import numpy as np


#function to calculate mean
# input -> list hogi
# now let's make it work for dataframes aka input.values()
def _calculateMean(input):

    sum = 0
    avg = len(input)
    for i in input.values: 
        sum += i
    x_mean = sum/avg
    return x_mean

# now calculate the difference of entry with mean
# sum(x - x_mean)

def sum(input):
    value = 0
    for i in input:
        value += i
    return value


def _calculateMeanDifference(input):

    element_mean = _calculateMean(input)
    _differ = 0
    # x_difference_x_mean_summition = 0
    print(element_mean)

    _diff_List = []

    # print('=====')
    for i in input.values:
        _differ = i - element_mean
        # print(i)
        # print('------')
        # print(_differ)
        # print('========')
        # print(sum)
        _diff_List.append(_differ)
        # x_difference_x_mean_summition += _differ
    return sum(_diff_List)

# summition of (x-x_mean)^2
def _calculateMeanDifferenceSquare(input):
    
    element_mean = _calculateMean(input)
    _differ_sq = 0
    # x_difference_x_mean_summition = 0

    x_difference_x_mean_summition = []

    print(element_mean)
    # print('=====')
    for i in input.values:
        _differ_sq = (i - element_mean) ** 2
        # print(i)
        # print('------')
        # print(_differ_sq)
        # print('========')
        # print(sum)
        # x_difference_x_mean_summition += _differ_sq
        x_difference_x_mean_summition.append(_differ_sq)
    return sum(x_difference_x_mean_summition)


# calculates the difference(difference(x, x_mean) , difference(y,y_mean))
def _calculateProductMeanofVariables(input, target):

    ip = input.values
    op = target.values

    ip_mean = _calculateMean(input)
    op_mean = _calculateMean(target)

    inputDiff = 0
    outputDiff = 0

    paramDifference = []

    for i,j in zip(ip,op):

        inputDiff = i - ip_mean
        outputDiff = j - op_mean

        paramDifference.append(inputDiff*outputDiff)
    return sum(paramDifference)

#calculating the slope of the line
def _fnSlope(input, target):

    ip = _calculateProductMeanofVariables(input,target)
    op = _calculateMeanDifferenceSquare(input)

    return ip/op

#calculating the intercept
def _fnIntercept(input,target):

    ip = _calculateMean(input)
    op = _calculateMean(target)
    m = _fnSlope(input,target)

    return op - (m * ip)


# fit function (we already implementated it but giving the right convention)
def _fit(input,target):

    slope = _fnSlope(input,target)
    intercept = _fnIntercept(input,target)

    # return f'Line of Equation based on provided dataset is y = {slope} * x + {intercept}'
    return slope, intercept

def _predict(new_x,input,target):

    x = new_x
    # m,c = _fit(input,target)
    m = _fnSlope(input,target)
    c = _fnIntercept(input,target)

    predicted_value = m*x + c
    return predicted_value

# df = py.DataFrame({
#     "Marks" : [20,40,60,70,90],
#     "Hours" : [2,4,6,8,9]
# })

# X = df[['Hours']]
# y = df['Marks']

# print(X)
# print(y)
# print(_calculateMean([1,2,3,4]))
# print(_calculateMeanDifferenceSquare(X))
# print(_calculateProductMeanofVariables(X,y))
# print( _fnSlope(X,y))
# print(_fnIntercept(X,y))
# print(_fit(X,y))
# x = 10
# print(_predict(x,X,y))
# print(type(X.values()))

# for i in X:
#     print(i)
#     print(type(i))

n = int(input('Number of columns: '))

dc = dict()

for _ in range(n):
    key = input('Enter column name: ')
    value = input(f'Enter values for {key} (comma separated): ')
    dc[key] = [float(i.strip()) for i in value.split(",")]

df = py.DataFrame(dc)

print("Provided Dataset")
print(df)
print(df.columns)

input_column = input('Which column is independent variable: ')
target_coumn = input('Which column is dependent variable: ')


X = df[input_column]
y = df[target_coumn]


x_test = int(input('Enter the value on which you want to predict: '))

print(f'Predicted value at {x_test} Hours is {_predict(x_test,X,y)}')