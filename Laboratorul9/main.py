import itertools

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
# # c
# q = 10
# theta = 1 / 4
# err = np.random.normal(0, 1, N)
# moving_average_series = np.full(N, np.mean(series))  # pun miu in fiecare termen
# for i in range(q, N):
#     window = series[i - q:i]
#     current_mean = np.mean(window)
#     moving_average_series[i] += theta * err[i - q] + err[i]
#
# plt.plot(moving_average_series)
# plt.show()



# d
p = range(0, 20)
q = range(0, 20)
pq_combinations = list(itertools.product(p, q))
