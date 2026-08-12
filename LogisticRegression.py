import pandas as py
import numpy as np

# goal is to create a logistic function using gradient ascent and newton's method (later)
# general idea is that we need a sigmoid function and a function for increasing the likelyhood of feature (theta)


class LogisticRegression:

    def __init__(self, alfa=0.000001, iterations=100000, tolerance=0.0000001):

        self.n = 0
        self.theta = np.array([])
        self.intercept = 0
        self.alfa = alfa
        self.iterations = iterations
        self.tolerance = tolerance

    def _predict(self, input):

        z = np.dot(input, self.theta) + self.intercept
        y_predicted = self._sigmoid(z)
        return y_predicted

    def predict_class(self, input):

        probability = self._predict(input)
        return (probability >= 0.5).astype(int)

    def sum(self, input):

        value = 0
        for i in input:
            value += i
        return value

    def multiple(self, param1, param2):

        value = 0
        for i, j in zip(param1, param2):
            value += i * j
        return value

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # log-likelihood -- gradient ASCENT means we MAXIMIZE this, not minimize it
    #   l(theta) = 1/m * summition of (from 1 to m) : y(i)log(predicted_value(i)) + (1-y(i))log(1-predicted_value(i))
    def _logLikelihood(self, y, y_predicted):

        m = len(y)
        term1 = self.multiple(y, np.log(y_predicted))
        term2 = self.multiple((1 - y), np.log(1 - y_predicted))
        likelihood = (1 / m) * (term1 + term2)
        return likelihood

    # gradient of the log-likelihood w.r.t theta and intercept
    #   d/dtheta = 1/m * X.T . (y - y_predicted)
    def _gradientCalculation(self, X, y, y_predicted):

        m = len(y)
        error = y - y_predicted
        grad_theta = (1 / m) * np.dot(X.T, error)
        grad_intercept = (1 / m) * self.sum(error)
        return grad_theta, grad_intercept

    # gradient ASCENT: move theta in the direction that increases likelihood
    #   theta := theta + alfa * gradient
    def _updateGradient(self, grad_theta, grad_intercept):

        self.theta = self.theta + (self.alfa * grad_theta)
        self.intercept = self.intercept + (self.alfa * grad_intercept)

    def fit(self, X, y):

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Number of input features
        self.n = X.shape[1]

        # One weight for every feature
        self.theta = np.zeros(self.n)
        self.intercept = 0

        previous_likelihood = float("-inf")

        for i in range(self.iterations):

            y_bar = self._predict(X)
            likelihood = self._logLikelihood(y, y_bar)

            grad_theta, grad_intercept = self._gradientCalculation(X, y, y_bar)
            self._updateGradient(grad_theta, grad_intercept)

            # stop once likelihood stops increasing meaningfully
            if abs(likelihood - previous_likelihood) < self.tolerance:
                break

            previous_likelihood = likelihood

        return self

    #fit transform function is required
    def _fitTransform(self, target):

        target_processed = np.unique(target)

        target_map = {
            val: i for i, val in enumerate(target_processed)
        }

        target_encoded = target.map(target_map)

        return target_encoded


def collect_training_data():

    n = int(input("Kitne training samples dalne hain? "))
    feature_vals = []
    label_vals = []

    for i in range(n):
        feature_vals.append(float(input(f"Sample {i+1} - numeric feature value: ")))
        label_vals.append(input(f"Sample {i+1} - class label (e.g. Relationship/Single): "))

    df = py.DataFrame({
        "Feature": feature_vals,
        "Status": label_vals
    })

    return df


df = collect_training_data()
print(df)

X = df['Feature']
y = df['Status']

model = LogisticRegression()
y = model._fitTransform(y)          # Status is categorical -> encode to 0/1
X = np.asarray(X, dtype=float)       # Feature is already numeric -> no encoding needed

print(y)
print(X)

model.fit(X, y)
print("theta:", model.theta)
print("intercept:", model.intercept)
print("predictions (probability):", model._predict(X.reshape(-1, 1)))
print("predictions (class):", model.predict_class(X.reshape(-1, 1)))

# classify a new, unseen point using the trained model
new_val = float(input("\nNaya data point classify karne ke liye numeric feature value: "))
new_point = np.array([[new_val]])
probability = model._predict(new_point)[0]
predicted_class = model.predict_class(new_point)[0]
print(f"probability: {probability:.4f}  ->  predicted class: {predicted_class}")


# Relationship/Single are only example labels.
# You can use any two class labels; the model automatically encodes them into 0 and 1.