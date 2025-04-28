import pandas as pd
import numpy as np

# READ DATA
df = pd.read_csv("Football_players.csv", encoding="ISO-8859-1")
age = df["Age"].to_numpy()
height = df["Height"].to_numpy()
mental = df["Mental"].to_numpy()
skill = df["Skill"].to_numpy()
salary = df["Salary"].to_numpy()

bias = np.ones(len(age))
X = np.column_stack((bias, age, height, mental, skill))
y = salary

# FUNCTIONS
def compute_beta(X, y):
    XT = X.T
    beta = np.linalg.inv(XT @ X) @ XT @ y
    return beta

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def cross_validation_mse(X, y, k):
    fold_size = len(X) // k
    mses = []

    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i != k - 1 else len(X)

        X_test = X[start:end]
        y_test = y[start:end]

        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        beta = compute_beta(X_train, y_train)
        y_pred = X_test @ beta
        mse = mean_squared_error(y_test, y_pred)
        mses.append(mse)

    return np.mean(mses)

# PRINT HEADER
print(f"{'Validation MSE':<20}{'8-fold CV MSE':<20}")
print(f"{'-'*20}{'-'*20}")

# 10 ITERATIONS
for _ in range(10):
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    # 80-20 Split
    split_idx = int(len(X) * 0.8)
    X_train = X_shuffled[:split_idx]
    y_train = y_shuffled[:split_idx]
    X_test = X_shuffled[split_idx:]
    y_test = y_shuffled[split_idx:]

    # Validation MSE
    beta_val = compute_beta(X_train, y_train)
    y_pred_val = X_test @ beta_val
    mse_val = mean_squared_error(y_test, y_pred_val)

    # Cross-Validation MSE
    mse_cv = cross_validation_mse(X_shuffled, y_shuffled, k=8)

    # PRINT RESULT
    print(f"{mse_val:<20.2f}{mse_cv:<20.2f}")
