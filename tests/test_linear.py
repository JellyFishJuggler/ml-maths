import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as py
from LinearRegression import _predict, _fnSlope, _fnIntercept

np.random.seed(42)
n = 60
x = np.random.uniform(0, 20, n)
y = 3 * x + 2 + np.random.normal(0, 2, n)

split = int(0.7 * n)
X_train, X_test = py.Series(x[:split]), py.Series(x[split:])
y_train, y_test = py.Series(y[:split]), py.Series(y[split:])

m = _fnSlope(X_train, y_train)
c = _fnIntercept(X_train, y_train)

y_pred = np.array([_predict(x_c, X_train, y_train) for x_c in X_test.values])

ss_res = np.sum((y_test.values - y_pred) ** 2)
ss_tot = np.sum((y_test.values - y_test.mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print("LinearRegression (simple, synthetic y = 3x + 2 + noise)")
print("Learnt slope: " + str(round(m, 3)) + "  intercept: " + str(round(c, 3)))
print("True target : slope 3.0  intercept 2.0")
print("Test R-squared: " + str(round(r2, 4)))