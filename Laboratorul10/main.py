import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

N = 1000
t = np.linspace(0, N, N)
trend = (3 * t ** 2 + 2 * t + 5) * 1e-5  # trb sa inmultesc cu asta ca altfel da urat pt ca e f mare
season = 2 * np.sin(t * np.pi * 0.02) + 3 * np.sin(t * np.pi * 0.07)
noise = np.random.normal(0, 1, N)
series = trend + season + noise


def AR(series, p):
    train_data = series[:-p]
    test_data = series[-p:]

    y = train_data[p:]
    k = len(y)
    Y = np.zeros((k, p))

    # zic ca ult elem din y este o combinatie a primelor p elemente, d aia in y iau de la p incolo
    for t in range(k):
        for lag in range(p):
            Y[t, lag] = train_data[t + p - lag - 1]  # sper sa fi nimerit coef
    big_gamma = Y.T @ Y
    small_gamma = Y.T @ y
    x_star = np.linalg.inv(big_gamma) @ small_gamma

    predictions = []
    last_values = train_data[-p:]

    for i in range(p):
        pred = x_star @ last_values[::-1]
        predictions.append(pred)
        last_values = np.append(last_values[1:], pred)

    return train_data, test_data, predictions, x_star


# train_data, test_data, pred = AR(series, 100)
# x_pred = np.arange(len(train_data), len(train_data) + len(pred))
# plt.figure(figsize=(14, 6))
# plt.plot(train_data, label="Train data")
# plt.plot(x_pred, pred, "r-", label="Prediction", alpha=0.7)
# plt.plot(x_pred, test_data, "g-", label="Real", alpha=0.7)
# plt.legend()
# plt.show()


# c)
def AR_greedy(series, p, features):
    train_data = series[:-p]
    test_data = series[-p:]
    y = train_data[p:]
    k = len(y)
    Y = np.zeros((k, p))

    for t in range(k):
        for lag in range(p):
            Y[t, lag] = train_data[t + p - lag - 1]

    selected_indices = []
    remaining_indices = list(range(p))
    for i in range(features):
        best_mse = float('inf')
        best_lag = -1

        for lag in remaining_indices:
            current_subset = selected_indices + [lag]
            Y_subset = Y[:, current_subset]

            coeffs, _, _, _ = np.linalg.lstsq(Y_subset, y)

            prediction = Y_subset @ coeffs
            mse = np.mean((y - prediction) ** 2)

            if mse < best_mse:
                best_mse = mse
                best_lag = lag

        selected_indices.append(best_lag)
        remaining_indices.remove(best_lag)
    Y = Y[:, selected_indices]
    final_coeffs_subset, _, _, _ = np.linalg.lstsq(Y, y)

    x_star = np.zeros(p)
    for idx, lag_idx in enumerate(selected_indices):
        x_star[lag_idx] = final_coeffs_subset[idx]

    predictions = []
    last_values = train_data[-p:]

    for i in range(p):
        pred = x_star @ last_values[::-1]
        predictions.append(pred)
        last_values = np.append(last_values[1:], pred)

    return train_data, test_data, predictions, x_star


def AR_regulartization(series, p):
    train_data = series[:-p]
    test_data = series[-p:]
    y = train_data[p:]
    k = len(y)
    Y = np.zeros((k, p))

    for t in range(k):
        for lag in range(p):
            Y[t, lag] = train_data[t + p - lag - 1]
    P_matrix = np.zeros((2 * p, 2 * p))
    YY = Y.T @ Y
    P_matrix[:p, :p] = YY
    P_matrix[:p, p:] = -YY
    P_matrix[p:, :p] = -YY
    P_matrix[p:, p:] = YY
    P = matrix(P_matrix)

    Yy = Y.T @ y
    q_vec = np.concatenate([-Yy + 1, Yy + 1])
    q = matrix(q_vec)

    G = matrix(-np.eye(2 * p))
    h = matrix(np.zeros(2 * p))

    sol = solvers.qp(P, q, G, h)
    z = np.array(sol['x']).flatten()

    x_star = z[:p] - z[p:]

    x_star[np.abs(x_star) < 1e-4] = 0

    predictions = []
    last_values = train_data[-p:]
    for i in range(p):
        pred = np.dot(x_star, last_values[::-1])
        predictions.append(pred)
        last_values = np.append(last_values[1:], pred)

    return train_data, test_data, predictions, x_star


p = 100
_, _, pred, x_star = AR(series, p)
_, _, pred_reg, x_star_reg = AR_regulartization(series, p)
train_data, test_data, pred_greedy, x_star_greedy = AR_greedy(series, p, 5)
x_pred = np.arange(len(train_data), len(train_data) + len(pred))
fig, axs = plt.subplots(3, figsize=(14, 6))
axs[0].plot(train_data, label="Train data")
axs[0].plot(x_pred, pred, "r-", label="Prediction", alpha=0.7)
axs[0].plot(x_pred, test_data, "g-", label="Real", alpha=0.7)
axs[1].plot(train_data, label="Train data")
axs[1].plot(x_pred, pred_greedy, "c-", label="Prediction", alpha=0.7)
axs[1].plot(x_pred, test_data, "g-", label="Real", alpha=0.7)
axs[2].plot(train_data, label="Train data")
axs[2].plot(x_pred, pred_reg, "m-", label="Prediction", alpha=0.7)
axs[2].plot(x_pred, test_data, "g-", label="Real", alpha=0.7)
axs[0].legend()
axs[1].legend()
axs[2].legend()
plt.tight_layout()
plt.show()


# d)
def find_roots(coefs):
    n = len(coefs) - 1
    if n < 1:
        return np.array([])

    C = np.zeros((n, n))
    c = coefs[1:] / coefs[0]  # sa am polinomul monic
    if n > 1:
        C[1:, :-1] = np.eye(n - 1)
    C[:, -1] = -c[::-1]
    roots = np.linalg.eigvals(C)
    return roots


print(find_roots(np.array([1, 0, -5, 0, 4, 0])))


# f
# Modific sa primesc si x_starurile din fiecare AR
coefs = np.concatenate(([1], -x_star))
coefs_reg = np.concatenate(([1], -x_star_reg))
coefs_greedy = np.concatenate(([1], -x_star_greedy))

magnitudes = np.abs(find_roots(coefs))
is_stationary = np.all(magnitudes > 1.0)
print(f"Normal Ar is stationary: {is_stationary}")

magnitudes = np.abs(find_roots(coefs_reg))
is_stationary = np.all(magnitudes > 1.0)
print(f"Regulation l1 Ar is stationary: {is_stationary}")

magnitudes = np.abs(find_roots(coefs_greedy))
is_stationary = np.all(magnitudes > 1.0)
print(f"Greedy Ar is stationary: {is_stationary}")