import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as py
from iD3_decision_tree import ID3

np.random.seed(42)

def rule(outlook, humidity, wind):
    if outlook == "Sunny":
        return "Yes" if humidity == "Normal" else "No"
    if outlook == "Overcast":
        return "Yes"
    if outlook == "Rain":
        return "Yes" if wind == "Weak" else "No"

def make_data(n):
    rows = []
    for _ in range(n):
        outlook = np.random.choice(["Sunny", "Overcast", "Rain"])
        humidity = np.random.choice(["High", "Normal"])
        wind = np.random.choice(["Weak", "Strong"])
        rows.append({
            "Outlook": outlook,
            "Temp": np.random.choice(["Hot", "Mild", "Cool"]),
            "Humidity": humidity,
            "Wind": wind,
            "Play": rule(outlook, humidity, wind),
        })
    return py.DataFrame(rows)

df = make_data(200)
X = df.drop(columns=["Play"])
y = df["Play"]

model = ID3(X, y)
tree = model.fit()
target_class = y.value_counts().idxmax()

train_pred = df.apply(lambda r: model.predict(tree, r), axis=1)
train_acc = (train_pred == y).mean()

test_df = make_data(100)
test_acc = test_df.apply(lambda r: model.predict(tree, r), axis=1)
test_acc = (test_df.apply(lambda r: model.predict(tree, r), axis=1) == test_df["Play"]).mean()

print("iD3_decision_tree (ID3, synthetic play-golf style data, 200 train / 100 unseen)")
print("Root split feature: " + str(next(iter(tree.keys()))))
print("Train accuracy : " + str(round(train_acc, 4)))
print("Test  accuracy : " + str(round(test_acc, 4)))