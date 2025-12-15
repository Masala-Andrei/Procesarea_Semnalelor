import itertools
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import scipy

# 1a)
N = 1000
t = np.linspace(0, N, N)
trend = (3 * t ** 2 + 2 * t + 5) * 1e-5  # trb sa inmultesc cu asta ca altfel da urat pt ca e f mare
season = 2 * np.sin(t * np.pi * 0.02) + 3 * np.sin(t * np.pi * 0.07)
noise = np.random.normal(0, 1, N)
series = trend + season + noise
#
# fig, axs = plt.subplots(4)
# axs[0].plot(trend)
# axs[0].set_title("Trend")
# axs[1].plot(season)
# axs[1].set_title("Season")
# axs[2].plot(noise)
# axs[2].set_title("Noise")
# axs[3].plot(series)
# axs[3].set_title("Series")
#
# plt.tight_layout()
# plt.show()
#
#
# # b
# def mediere(series, alpha):
#     series_med = np.zeros(len(series))
#     series_med[0] = series[0]
#     for i in range(1, len(series)):
#         series_med[i] = alpha * series[i] + (1 - alpha) * series_med[i - 1]
#     return series_med
#
# alpha = 0.1
# series_med = mediere(series, alpha)
# plt.plot(series, label="Original")
# plt.plot(series_med, label="Mediere exp alpha = " + str(alpha), linestyle="--")
# plt.legend()
# plt.show()
#
# err = 1
# best_alpha = 0.1
# alphas = np.linspace(0, 1, 100)
# for i in alphas:
#     curr_err = np.mean((series - mediere(series, i)) ** 2)
#     if curr_err < err:
#         curr_err = err
#         best_alpha = i
# best_mediere = mediere(series, best_alpha)
# plt.plot(series, label="Original")
# plt.plot(best_mediere, label="Mediere exp alpha = " + str(best_alpha), linestyle="--")
# plt.legend()
# plt.show()
#
#
# c
# q = 10
# theta = 1 / 4  # estimez un theta1 = 1 / 4 ca in ex de la curs (desi ar trb sa fac lstsq)
# moving_average_series = np.full(N, np.nan)
# err = np.zeros(N)
# for i in range(q, N):
#     window = series[i - q:i]
#     current_mean = np.mean(window)
#     err[i] = series[i] - current_mean
#     moving_average_series[i] = theta * err[i - 1] + current_mean
#
# plt.plot(series, label="Series")
# plt.plot(moving_average_series, alpha=0.7, label="Moving average")
# plt.legend()
# plt.show()

# d
p = range(1, 20)
q = range(1, 20)
pq_combinations = list(itertools.product(p, q))


def ARMA(series, p, q):
    train_data = series[:-p]

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

    predictionsAR = []
    last_values = train_data[-p:]

    for i in range(p):
        pred = x_star @ last_values[::-1]
        predictionsAR.append(pred)
        last_values = np.append(last_values[1:], pred)

    theta = 1 / 4
    moving_average_series = np.full(N, 0)
    err = np.zeros(N)
    for i in range(q, N):
        window = series[i - q:i]
        current_mean = np.mean(window)
        err[i] = series[i] - current_mean
        moving_average_series[i] = theta * err[i - 1] + current_mean

    pred_arma = last_values + moving_average_series[]
    return pred_arma


best_mse = np.inf
best_p, best_q = 0, 0
for p, q in pq_combinations:
    mse = mean_squared_error(series, ARMA(series, p, q))
    if mse < best_mse:
        best_mse = mse
        best_p = p
        best_q = q

print(f"Best p is {best_p} and best q is {best_q}")

best_arma = ARMA(series, best_p, best_q)
plt.plot(series, label="Series")
plt.plot(best_arma, alpha=0.7, label="Moving average")
plt.legend()
plt.show()

