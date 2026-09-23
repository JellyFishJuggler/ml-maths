# Python ML Models (Custom Implementations)

Machine learning models implemented from scratch in NumPy + Pandas — no sklearn used for the algorithms. Each file is self-contained: the model class/functions, a built-in demo, and a corresponding test script that evaluates it on a new/unseen dataset.

## Models & Results

### 1. NaiveBayes.py — Multinomial Naive Bayes (Spam Detection)

Classifies text as **spam** or **ham** using word likelihoods with Laplace smoothing and a log-sum score.

- Trained on 15 spam + 15 ham sample emails (inline `SPAM_TRAIN` / `HAM_TRAIN`).
- Trained via `fit()`, predicts via `predict()`. Run directly for an interactive message loop.

**Evaluation** (`test_naivebayes.py`) — 50 brand-new messages (25 spam + 25 ham):

| Metric    | Value |
| --------- | ----- |
| Accuracy  | 86%   |
| Precision | 100%  |
| Recall    | 72%   |
| F1        | 0.84  |

### 2. LogisticRegression.py — Logistic Regression (Gradient Ascent)

Binary classifier built with a sigmoid and gradient **ascent** (maximizes log-likelihood).

- Numeric feature(s) + categorical 2-class label; labels auto-encoded to 0/1.
- Run directly to enter training samples interactively, or import the class.

**Evaluation** (`test_logistic.py`) — synthetic single-feature dataset (80 points, 70/30 split, with noise):

| Metric         | Value |
| -------------- | ----- |
| Train accuracy | 94.6% |
| Test accuracy  | 95.8% |

### 3. LinearRegression.py — Simple Linear Regression (Closed Form)

Predicts `y = m*x + c` using the least-squares slope/intercept formulas.

- Takes a DataFrame, fits slope + intercept, predicts on a new x.
- Run directly for interactive column input, or import the functions.

**Evaluation** (`test_linear.py`) — synthetic `y = 3x + 2 + noise` (60 points, 70/30 split):

| Metric           | Value           |
| ---------------- | --------------- |
| Learnt slope     | 2.93 (true 3.0) |
| Learnt intercept | 2.55 (true 2.0) |
| Test R²         | 0.9945          |

### 4. MultiLinearRegression.py — Multiple Linear Regression (Normal Equation)

Fits multiple features with the closed-form normal equation `theta = (XᵀX)⁻¹ Xᵀy`.

**Evaluation** (`test_multi.py`) — synthetic `y = 2 + 3x1 - 4x2 + noise` (80 points, 70/30 split):

| Metric       | Value                             |
| ------------ | --------------------------------- |
| Learnt theta | [2.57, 2.99, -4.09] vs [2, 3, -4] |
| Test R²     | 0.9912                            |

### 5. GradientDescent.py — Gradient Descent for Linear Regression

Minimizes MSE by iteratively updating weights/intercept, with convergence tolerance and optional cost history tracking.

> Note: with the default learning rate (`alfa=0.000001`) convergence is slow, so the learnt weights don't fully reach the true values and R² is the lowest of the three regression models. It improves if `alfa` is increased.

**Evaluation** (`test_gradient.py`) — same synthetic dataset as MultiLinearRegression (80 points, 70/30 split):

| Metric         | Value                            |
| -------------- | -------------------------------- |
| Learnt weights | [2.53, -3.17] (true [3.0, -4.0]) |
| Test R²       | 0.9499                           |

### 6. iD3_decision_tree.py — ID3 Decision Tree (Categorical Classification)

Builds a classification tree by choosing, at each node, the feature with the highest **information gain** (lowest entropy). Categorical features only.

- `fit()` grows the tree recursively; `predict(tree, row)` walks a row down to a leaf, `entropy()`/`informationGain()` back the splits.
- Run directly for the built-in loan-approval demo, which also draws the tree via `plotTree()`.
- Note: `plotTree()` is **AI-generated** code.

**Evaluation** (`test_id3.py`) — synthetic play-golf-style data generated from a deterministic rule (200 train / 100 brand-new rows):

| Metric         | Value   |
| -------------- | ------- |
| Root split     | Outlook |
| Train accuracy | 100%    |
| Test accuracy  | 100%    |

> Note: train and test accuracy are both 100% because the noise-free synthetic data follows a fixed rule, so the learned tree recovers that rule exactly. On real noisier data expect lower scores.

## Summary

| Model                 | Type           | Metric   | Unseen-data Score |
| --------------------- | -------------- | -------- | ----------------- |
| NaiveBayes            | Classification | Accuracy | 86%               |
| LogisticRegression    | Classification | Accuracy | 95.8%             |
| LinearRegression      | Regression     | R²      | 0.9945            |
| MultiLinearRegression | Regression     | R²      | 0.9912            |
| GradientDescent       | Regression     | R²      | 0.9499            |
| iD3_decision_tree     | Classification | Accuracy | 100%              |

> Note: results are on small synthetic/hand-made datasets. The NaiveBayes accuracy (86%) is true test accuracy from 50 unseen messages; the whole train set's accuracy (100%) is memorization and not a real score.

## How to run

```bash
# Model demo (interactive where applicable)
python NaiveBayes.py
python LogisticRegression.py
python LinearRegression.py
python MultiLinearRegression.py
python GradientDescent.py
python iD3_decision_tree.py

# Automated evaluation on unseen data (tests are in the tests/ folder)
python tests/test_naivebayes.py
python tests/test_logistic.py
python tests/test_linear.py
python tests/test_multi.py
python tests/test_gradient.py
python tests/test_id3.py
```

## Dependencies

- Python 3
- NumPy
- Pandas
- Matplotlib (only needed for the ID3 `plotTree()` demo)
