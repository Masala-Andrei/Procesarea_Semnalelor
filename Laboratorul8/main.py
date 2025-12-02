import numpy as np
import matplotlib.pyplot as plt
import scipy

# 1a
N = 1000
t = np.linspace(0, N, N)
trend = (3 * t ** 2 + 2 * t + 5) * 1e-5  # trb sa inmultesc cu asta ca altfel da urat pt ca e f mare
season = 2 * np.sin(t * np.pi * 0.02) + 3 * np.sin(t * np.pi * 0.07)
noise = np.random.normal(0, 1, N)
series = trend + season + noise

fig, axs = plt.subplots(4)
axs[0].plot(trend)
axs[0].set_title("Trend")
axs[1].plot(season)
axs[1].set_title("Season")
axs[2].plot(noise)
axs[2].set_title("Noise")
axs[3].plot(series)
axs[3].set_title("Series")

plt.tight_layout()
plt.show()

# b
cseries = series - np.mean(series)
np_autocorr = np.correlate(cseries, cseries, "full")
np_autocorr = np_autocorr[np_autocorr.size // 2:]
np_autocorr /= np_autocorr[0]


# pk = sum((series[t+1:n] - mean) * (series[1:t+1] - mean))
def acv(series, n):
    mean = np.mean(series)
    d = np.sum((series - mean) ** 2)
    acv = []
    for k in range(n + 1):
        temp = 0.0
        for t in range(k, n):
            temp += (series[t] - mean) * (series[t - k] - mean)
        acv.append(temp / d)
    return acv


hand_autocorr = acv(series, N)

fig, axs = plt.subplots(2, figsize=(10, 8))
axs[0].plot(np_autocorr)
axs[0].set_title("Numpy")
axs[1].plot(hand_autocorr)
axs[1].set_title("De mana")
plt.tight_layout()
plt.show()


# c
def autoreg(series, p, m=None):
    train_data = series[:-p]
    test_data = series[-p:]
    y = train_data[p:p+m] if m is not None else train_data[p:]  # iau ultimele p elem
    if m is None:
        m = len(y)
    Y = np.zeros((m, p))
    # zic ca ult elem din y este o combinatie a primelor p elemente, d aia in y iau de la p incolo
    for t in range(m):
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

    return train_data, test_data, predictions


train_data, test_data, pred = autoreg(series, 100)
x_pred = np.arange(len(train_data), len(train_data) + len(pred))
plt.figure(figsize=(14, 6))
plt.plot(train_data, label="Train data")
plt.plot(x_pred, pred, "r-", label="Prediction", alpha=0.7)
plt.plot(x_pred, test_data, "g-", label="Real", alpha=0.7)
plt.legend()
plt.show()


# d

def tune_mp(series, m_range):
    best_error = np.inf
    best_p = None
    best_m = None
    for m in range(2, m_range + 1):
        for p in range(2, m):
            if m + p > N:
                break
            train_data = series[:N-m]
            test_data = series[N-m:]
            y = train_data[p:]

            Y = np.zeros((m, p))
            for t in range(m):
                for lag in range(p):
                    Y[t, lag] = train_data[t + p - lag - 1]
            big_gamma = Y.T @ Y
            small_gamma = Y.T @ y
            x_star = np.linalg.inv(big_gamma) @ small_gamma
            pred = x_star @ train_data[-p:][::-1]
            error = np.sum((pred - test_data[-1]) ** 2)
            if error < best_error:
                print(f"New best error: {error}, p: {p}, m: {m}")
                best_error = error
                best_p = p
                best_m = m
    return best_p, best_m


best_p, best_m = tune_mp(series, 100)

print(f"Cei mai buni hiperparametri:")
print(f"p (ordinul AR) = {best_p}")
print(f"m (orizontul de test) = {best_m}")

train_data, test_data, pred = autoreg(series, best_p, best_m)
x_pred = np.arange(len(train_data), len(train_data) + len(pred))
plt.figure(figsize=(14, 6))
plt.plot(train_data, label="Train data")
plt.plot(x_pred, pred, "r-", label="Prediction", alpha=0.7)
plt.plot(x_pred, test_data, "g-", label="Real", alpha=0.7)
plt.legend()
plt.show()


