# Decision Tree Implementation (ID3 type)

# Required calculations -> Entropy and Information Gain

# In ideal situation, infomation gain will be inversely proportional to entropy

# Entropy = - sigma_i_to_n (p_i * log_2_[p_i])
# Information gain = Entropy - sigma_i_to_n (|Entropy of value of selected class| / |Total Entropy|) * |Sv|

import pandas as py
import numpy as np
import matplotlib.pyplot as mp

class ID3:
    def __init__(self, X: py.DataFrame, y: py.Series):

        self.X = X
        self.y = y
        self.n = len(y)

    def entropy(self, x):

        counts = x.value_counts()
        probabilities = counts / len(x)

        s = 0
        for p in probabilities:
            s += p * np.log2(p)

        return -s

    def informationGain(self, x):

        w_e = 0
        t_e = self.entropy(self.y)

        for i in x.unique():
            subset_y = self.y[x == i]
            e = self.entropy(subset_y)
            w_e += e * len(subset_y) / self.n

        return t_e - w_e

    def featureSelection(self):

        ig_max = -1
        root = None

        for i in self.X:

            ig_curr = self.informationGain(self.X[i])
            if ig_curr > ig_max:
                ig_max = ig_curr
                root = i

        return root
    
    def fit(self):

        if len(self.y.unique()) == 1:
            return self.y.iloc[0]

        if len(self.X.columns) == 0:
            return self.y.value_counts().idxmax()

        root = self.featureSelection()

        tree = {root: {}}

        for value in self.X[root].unique():

            mask = self.X[root] == value

            subset_X = self.X[mask].drop(columns=[root])
            subset_y = self.y[mask]

            child = ID3(subset_X, subset_y).fit()

            tree[root][value] = child

        return tree


    def predict(self, tree, row):

        for feature in tree:
            value = row[feature]
            nextnode = tree[feature][value]

            if isinstance(nextnode, dict):
                return self.predict(nextnode, row)
            else:
                return nextnode

    def plotTree(self):
    
        tree = self.fit()
    
        fig, ax = mp.subplots(figsize=(12, 7))
        ax.axis("off")
    
        def draw(node, x, y, dx, parent=None, edge_label=""):
        
            # Leaf node
            if not isinstance(node, dict):
                ax.text(
                    x, y, str(node),
                    ha="center", va="center",
                    bbox=dict(boxstyle="round", facecolor="lightgreen")
                )
    
                if parent is not None:
                    ax.plot([parent[0], x], [parent[1], y], "k-")
                    ax.text(
                        (parent[0] + x) / 2,
                        (parent[1] + y) / 2,
                        edge_label
                    )
    
                return
    
            # Current feature
            feature = list(node.keys())[0]
    
            ax.text(
                x, y, feature,
                ha="center", va="center",
                bbox=dict(boxstyle="round", facecolor="lightblue")
            )
    
            if parent is not None:
                ax.plot([parent[0], x], [parent[1], y], "k-")
                ax.text(
                    (parent[0] + x) / 2,
                    (parent[1] + y) / 2,
                    edge_label
                )
    
            branches = node[feature]
            values = list(branches.keys())
    
            for i, value in enumerate(values):
            
                child = branches[value]
    
                child_x = x + (i - (len(values) - 1) / 2) * dx
                child_y = y - 1
    
                draw(
                    child,
                    child_x,
                    child_y,
                    dx / 2,
                    parent=(x, y),
                    edge_label=str(value)
                )
    
        draw(tree, 0, 0, 4)
    
        mp.show()
            

# data = {
#     "Outlook" : ["Sunny", "Sunny", "Overcast", "Rain", "Rain", "Rain", "Overcast", "Sunny", "Sunny"],
#     "Temp" : ["Hot","Hot","Hot","Mild","Cool","Cool","Cool","Mild","Cool"],
#     "Humidity" : ["High","High","High","High","Normal","Normal","Normal","Normal","High"],
#     "Wind" : ["Weak","Strong","Weak","Weak","Weak","Strong","Strong","Weak","Weak"],
#     "Play" : ["N","N","Y","Y","Y","N","Y","N","Y"]
# }

data = {
    "Age": [
        "Young", "Young", "Middle", "Middle", "Senior",
        "Senior", "Middle", "Young", "Senior", "Middle"
    ],
    "Income": [
        "Low", "High", "High", "Low", "High",
        "Low", "High", "Low", "High", "High"
    ],
    "CreditScore": [
        "Bad", "Good", "Good", "Bad", "Good",
        "Bad", "Good", "Bad", "Good", "Bad"
    ],
    "Employment": [
        "No", "Yes", "Yes", "No", "Yes",
        "No", "Yes", "No", "Yes", "Yes"
    ],
    "LoanApproved": [
        "No", "Yes", "Yes", "No", "Yes",
        "No", "Yes", "No", "Yes", "No"
    ]
}

df = py.DataFrame(data)
# print(df)

# X = df.drop(columns=["Play"])
# y = df["Play"]

X = df.drop(columns=["LoanApproved"])
y = df["LoanApproved"]


model = ID3(X,y)
print(f'Entropy: {model.entropy(y)}')
# print(f'Information Gain of wind: {model.informationGain(X["Wind"])}')
print(model.fit())
# row = model.X.iloc[0]
row = py.Series({
    "Age": "Young",
    "Income": "High",
    "CreditScore": "Good",
    "Employment": "Yes"
})

print(model.predict(model.fit(), row))
print(model.predict(model.fit(), row))
model.plotTree()
# print(y)
# print(y.value_counts())
# print(y[y == 'Y'].value_counts())
# print(y[y == 'N'].value_counts())
# print(y.loc[0])