import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as py
from GradientDescent import GradientDescent

np.random.seed(42)
n = 80
X1 = np.random.uniform(0, 10, n)
X2 = np.random.uniform(0, 10, n)
y = 2 + 3 * X1 - 4 * X2 + np.random.normal(0, 1.5, n)

df = py.DataFrame({"x1": X1, "x2": X2, "target": y})
split = int(0.7 * n)

train = df.iloc[:split].reset_index(drop=True)
test = df.iloc[split:].reset_index(drop=True)

model = GradientDescent()
model.fit(train[["x1", "x2"]], train["target"])
y_pred = model.predict(test[["x1", "x2"]])

ss_res = np.sum((test["target"].values - y_pred) ** 2)
ss_tot = np.sum((test["target"].values - test["target"].mean()) ** 2)
r2 = 1 - ss_res / ss_tot

print("GradientDescent (multi-linear via GD, synthetic y = 2 + 3x1 - 4x2 + noise)")
print("Learnt weights (x1, x2): " + str(model.m.round(3)) + "  intercept: " + str(round(model.c, 3)))
print("True        (x1, x2): [3.0, -4.0]  intercept: 2.0")
print("Test R-squared: " + str(round(r2, 4)))