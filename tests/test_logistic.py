import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as py
from LogisticRegression import LogisticRegression

np.random.seed(42)
n = 80
X_vals = np.random.uniform(-6, 6, n)
y_vals = (X_vals > 0).astype(int)
y_vals = np.where(np.random.random(n) < 0.05, 1 - y_vals, y_vals)  # 5% noise

split = int(0.7 * n)
X_train, X_test = X_vals[:split], X_vals[split:]
y_train, y_test = y_vals[:split], y_vals[split:]

df_train = py.DataFrame({"Feature": X_train, "Status": y_train})

model = LogisticRegression()
model.fit(df_train["Feature"], y_train)

train_pred = model.predict_class(X_train.reshape(-1, 1)).flatten()
train_acc = (train_pred == y_train).mean()

test_pred = model.predict_class(X_test.reshape(-1, 1)).flatten()
test_acc = (test_pred == y_test).mean()

print("LogisticRegression (synthetic binary, single feature)")
print("theta:", model.theta, "intercept:", model.intercept)
print("Train accuracy: " + str(round(train_acc, 4)))
print("Test accuracy:  " + str(round(test_acc, 4)))