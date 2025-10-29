import matplotlib.pyplot as plt
import scipy
import numpy as np
import math

# 1
# N = 8
# F = np.zeros((N, N), dtype=np.complex128)
# for k in range(N):
#     for j in range(N):
#         F[k][j] = np.power(np.e, -2 * np.pi * 1j * j * k / N)
#
# fig, axs = plt.subplots(N, figsize=(10, 6), sharex=True, sharey=True)
# for k in range(N):
#     axs[k].plot([j + 1 for j in range(N)], F[k].real)
#     axs[k].plot([j + 1 for j in range(N)], F[k].imag)
#     axs[k].set_ylabel(f"Linia {k}")
#
# plt.show()
#
# dft = scipy.linalg.dft(N)  # matricea fourier discreta pt n = 8
# print("Este unitara? - " + str(np.allclose(F, dft)))  # Trebuie sa declar ca elementele din F vor fi complexe upsi
#
# # 2
#
# axis = np.linspace(0, 1, 1000)
# signal = np.sin(2 * np.pi * 7 * axis + np.pi / 6)
#
# fig, axs = plt.subplots(1, 2, figsize=(10, 4))
# complexSig = signal * np.power(np.e, -2 * np.pi * 1j * axis)
#
# axs[0].scatter(axis, signal, c=np.abs(signal), cmap="viridis")
# axs[1].scatter(complexSig.real, complexSig.imag, c=np.abs(complexSig), cmap="viridis")
# plt.tight_layout()
# plt.show()
#
# fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
# omegas = [3, 4, 6, 7, 9, 10]
# for p, ax in enumerate(axs.flat):
#     omega = omegas[p]
#     z = signal * np.power(np.e, -2 * np.pi * omega * 1j * axis)
#     center = np.mean(z)
#     ax.scatter(z.real, z.imag, c=abs(z), cmap="plasma")
#     ax.scatter(center.real, center.imag, color='black')
#     ax.set_title(f"omega = {omega:.2f}")
#     ax.set_xlim([-1.1, 1.1])
#     ax.set_ylim([-1.1, 1.1])
#     ax.axhline(0, color="black", linewidth=0.8)
#     ax.axvline(0, color="black", linewidth=0.8)
#
# plt.tight_layout()
# plt.show()

# MDA, nu prea mi place ce face scatter la sinusoida si in general la semnale
# TODO: sa vad daca gasesc ceva mai bun decat scatter

# 3

axis = np.linspace(0, 1, 1000)
N = len(axis)

f1 = 13
f2 = 58
f3 = 123

# semnal1 + semnal2 + semnal3
signal = (np.sin(2 * np.pi * axis * f1) +
          np.sin(2 * np.pi * axis * f2 + np.pi / 6) +
          np.sin(2 * np.pi * axis * f3 + np.pi / 4))

X = np.zeros(N, dtype=np.complex128)
for omega in range(N):
    for n in range(N):
        X[omega] += signal[n] * np.exp(-2 * np.pi * 1j * omega * n / N)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))

axs[0].plot(axis, signal)
axs[0].set_xlabel("Timp (s)")
axs[0].set_ylabel("x(t)")


freq = np.arange(0, 1000, 1)
markerline, stemlines, baseline = axs[1].stem(
    freq, np.abs(X[0:1000]), linefmt="k-", markerfmt="ko"
)
markerline.set_markerfacecolor("none")
stemlines.set_linewidth(0.5)
baseline.set_color("k")
axs[1].set_xlabel("Frecventa (Hz)")
axs[1].set_ylabel("|X(omega)|")
axs[1].set_xlim([0, 1000 / 2])

plt.tight_layout()
plt.show()
