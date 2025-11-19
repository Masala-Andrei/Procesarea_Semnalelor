import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

#
# t = np.linspace(-3, 3, 6 * 1000)
# B = 0.75
# x = np.sinc(B * t) * np.sinc(B * t)
#
# fig, axs = plt.subplots(2, 2, figsize=(12, 6))
# # 1
# # Imi pare rau pentru ce urmeaza sa scriu e putin abominabil, dar mi a fost lene sa fac un for
# fs = 1
# Ts = 1 / fs
# n = np.arange(np.ceil(-3 / Ts), np.floor(3 / Ts) + 1)
# ts = n * Ts
# xn = np.sinc(B * ts) ** 2
# xp = np.sum(xn[:, None] * np.sinc((t[None, :] - ts[:, None]) / Ts), axis=0)
# axs[0][0].plot(t, x)
# axs[0][0].plot(t, xp, "--")
# axs[0][0].stem(ts, xn)
# axs[0][0].set_xlim(right=-3, left=3)
#
# fs = 1.5
# Ts = 1 / fs
# n = np.arange(np.ceil(-3 / Ts), np.floor(3 / Ts) + 1)
# ts = n * Ts
# xn = np.sinc(B * ts) ** 2
# xp = np.sum(xn[:, None] * np.sinc((t[None, :] - ts[:, None]) / Ts), axis=0)
# axs[0][1].plot(t, x)
# axs[0][1].plot(t, xp, "--")
# axs[0][1].stem(ts, xn)
# axs[0][1].set_xlim(right=-3, left=3)
#
# fs = 2
# Ts = 1 / fs
# n = np.arange(np.ceil(-3 / Ts), np.floor(3 / Ts) + 1)
# ts = n * Ts
# xn = np.sinc(B * ts) ** 2
# xp = np.sum(xn[:, None] * np.sinc((t[None, :] - ts[:, None]) / Ts), axis=0)
# axs[1][0].plot(t, x)
# axs[1][0].plot(t, xp, "--")
# axs[1][0].stem(ts, xn)
# axs[1][0].set_xlim(right=-3, left=3)
#
# fs = 4
# Ts = 1 / fs
# n = np.arange(np.ceil(-3 / Ts), np.floor(3 / Ts) + 1)
# ts = n * Ts
# xn = np.sinc(B * ts) ** 2
# xp = np.sum(xn[:, None] * np.sinc((t[None, :] - ts[:, None]) / Ts), axis=0)
# axs[1][1].plot(t, x)
# axs[1][1].plot(t, xp, "--")
# axs[1][1].stem(ts, xn)
# axs[1][1].set_xlim(right=-3, left=3)
#
# plt.tight_layout()
# plt.show()
# # Cand frecventa de esantionare este mai mare decat 2 * B, atunci graficul se suprapune
# # cu xp (adica x caciula)
#
# # 2
# N = 100
# x = np.random.rand(N)
# fig, axs = plt.subplots(4, figsize=(12, 8))
# for ax in axs:
#     ax.plot(x)
#     x = np.convolve(x, x)
# plt.tight_layout()
# plt.show()
#
#
# # 3
# # Aici nu voi ma repeta codul, voi face ca omul
# def generate_poly(order):
#     coef = np.random.randint(-7, 7, order)
#     coef[0] = np.random.randint(1, 7)
#     return np.poly1d(coef)
#
#
# ordp = 7
# ordq = 6
# p = generate_poly(ordp)
# q = generate_poly(ordq)
#
# orderf = p.order + q.order + 1
# # Direct
# result = np.zeros(orderf)
# start = time.perf_counter()
# for i in range(p.order + 1):
#     for j in range(q.order + 1):
#         result[i + j] += p.c[i] * q.c[j]
# end = time.perf_counter()
# print("Direct:\n" + str(np.poly1d(result)))
# print("Time: " + str(end - start) + " s")
#
# print("------------------------------------")
# # FFT
# start = time.perf_counter()
# pfft = np.fft.fft(p.c, orderf)
# qfft = np.fft.fft(q.c, orderf)
# result = np.fft.ifft(pfft * qfft).real
# end = time.perf_counter()
# print("FFT:\n" + str(np.poly1d(result)))
# print("Time: " + str(end - start) + " s")

# 4
# t = np.linspace(0, 1, 20)
# x = np.sin(2 * np.pi * t * 3)
# print(x)
# d = 8
# y = np.roll(x, d)
# print(y)
#
# m1 = np.fft.ifft(np.dot(np.fft.fft(x), np.fft.fft(y)))
# print(m1)
#
# m2 = np.fft.ifft(np.fft.fft(y) / np.fft.fft(x))
# print(m2)

# # 5
# N = 200
# t = np.linspace(0, 1, 1000)
# x = np.sin(2 * np.pi * 100 * t)
# wH = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))
# wR = np.ones(N)
#
# rect_window = x[:N] * wR
# hann_window = x[:N] * wH
#
# fig, axs = plt.subplots(2, figsize=(12, 8))
# axs[0].plot(t[:N], rect_window)
# axs[0].set_title("Rectangular Window")
# axs[1].plot(t[:N], hann_window)
# axs[1].set_title("Hanning Window")
# plt.tight_layout()
# plt.show()

# 6
# a,b)
index = 3480
x = np.genfromtxt("Train.csv", delimiter=",", skip_header=1)
x = x[index: index + 24 * 3][:, 2]
#
# plt.plot(x, label="Semnal brut")
# plt.title("Semnal brut si semnalele filtrate running average")
#
# window_sizes = [5, 9, 13, 17]
# op = [0.5, 0.6, 0.7, 0.8]
# for w, opacity in zip(window_sizes, op):
#     filtered_signal = np.convolve(x, np.ones(w) / w)
#     plt.plot(filtered_signal, label=f"w = {w}", alpha=opacity)
#
# plt.tight_layout()
# plt.show()

# c)
sample_freq = 1 / 3600
# filtrez freq mai mari de 12 ore
Wn = sample_freq / 12
f_nyq = sample_freq / 2
Wn_norm = Wn / f_nyq

# d)
order = 5
rp = 5

bb, ab = scipy.signal.butter(order, Wn_norm)
bc, ac = scipy.signal.cheby1(order, rp, Wn_norm)

wb, hb = scipy.signal.freqz(bb, ab)
wc, hc = scipy.signal.freqz(bc, ac)

plt.plot(wb, 20 * np.log10(np.abs(hb)), label="Butterworth")
plt.plot(wc, 20 * np.log10(np.abs(hc)), label="Chebyshev")
plt.legend()
plt.show()

# e)
x_butter = scipy.signal.filtfilt(bb, ab, x)
x_cheby1 = scipy.signal.filtfilt(bc, ac, x)
plt.plot(x, label="Semnal brut")
plt.plot(x_butter, label="Butterworth")
plt.plot(x_cheby1, label="Chebyshev")
plt.legend()
plt.show()

# f)

