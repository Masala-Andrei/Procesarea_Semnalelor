import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, datasets
import pooch

# 1
# n1 = 64
# n2 = 64
# # x1
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# i, j = np.indices((n1, n2))
# X = np.sin(2 * np.pi * i + 3 * np.pi * j)
# axs[0].imshow(X.real)
# axs[0].set_title("Imagine")
# fig.colorbar(axs[0].imshow(np.real(X)), ax=axs[0])
# Y = 20 * np.log10(abs(np.fft.fft2(X)))
# axs[1].imshow(Y)
# axs[1].set_title("Spectru")
# fig.colorbar(axs[1].imshow(Y), ax=axs[1])
# plt.show()
#
# # x2
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# i, j = np.indices((n1, n2))
# X = np.sin(4 * np.pi * i) + np.cos(6 * np.pi * j)
# axs[0].imshow(X.real)
# axs[0].set_title("Imagine")
# fig.colorbar(axs[0].imshow(np.real(X)), ax=axs[0])
# Y = 20 * np.log10(abs(np.fft.fft2(X)) + 1e-20)
# axs[1].imshow(Y)
# axs[1].set_title("Spectru")
# fig.colorbar(axs[1].imshow(Y), ax=axs[1])
# plt.show()
#
# # y1
# Y = np.zeros((n1, n2))
# Y[0][5] = Y[0][n2 - 5] = 1
# X = np.fft.ifft2(Y)
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# axs[0].imshow(X.real)
# axs[0].set_title("Imagine")
# fig.colorbar(axs[0].imshow(np.real(X)), ax=axs[0])
# axs[1].imshow(Y)
# axs[1].set_title("Spectru")
# fig.colorbar(axs[1].imshow(Y), ax=axs[1])
# plt.show()
#
# # y2
# Y = np.zeros((n1, n2))
# Y[5][0] = Y[n1 - 5][0] = 1
# X = np.fft.ifft2(Y)
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# axs[0].imshow(X.real)
# axs[0].set_title("Imagine")
# fig.colorbar(axs[0].imshow(np.real(X)), ax=axs[0])
# axs[1].imshow(Y)
# axs[1].set_title("Spectru")
# fig.colorbar(axs[1].imshow(Y), ax=axs[1])
# plt.show()
#
# # y3
# Y = np.zeros((n1, n2))
# Y[5][5] = Y[n1 - 5][n2 - 5] = 1
# X = np.fft.ifft2(Y)
# fig, axs = plt.subplots(1, 2, figsize=(12, 6))
# axs[0].imshow(X.real)
# axs[0].set_title("Imagine")
# fig.colorbar(axs[0].imshow(np.real(X)), ax=axs[0])
# axs[1].imshow(Y)
# axs[1].set_title("Spectru")
# fig.colorbar(axs[1].imshow(Y), ax=axs[1])
# plt.show()

# 2
# nu mi a mers misc, zice ca e deprecated
X = datasets.face(gray=True)
# snr = 0.0075
#
#
# # snr = msq(s) / msq(n), unde msq este mean squares de s (semnal) resp n (noise)
# def calc_snr(orig, comp):
#     noise = orig - comp
#     noise_mean_squares = np.mean(noise ** 2)
#     if noise_mean_squares == 0:
#         return np.inf
#     signal_mean_squares = np.mean(orig ** 2)
#     return signal_mean_squares / noise_mean_squares
#
#
# X_cutoff = X
# # Magnitudini sortate (de la mici la mari)
# for i in range(200):
#     curr_snr = calc_snr(X, X_cutoff)
#     print(f"SNR = {curr_snr}")
#     if curr_snr <= snr:
#         break
#     Y = np.fft.fft2(X_cutoff)
#     freq_cutoff = np.max(np.abs(Y)) * 0.85
#     Y[np.abs(Y) >= freq_cutoff] = 0
#     X_cutoff = np.fft.ifft2(Y).real
#
#
# fig, axs = plt.subplots(1, 2, figsize=(10, 5))
# axs[0].imshow(X, cmap="gray")
# axs[0].set_title("Original")
# axs[1].imshow(X_cutoff, cmap="gray")
# axs[1].set_title("Compressed")
# plt.show()

# 3
pixel_noise = 200

noise = np.random.randint(-pixel_noise, high=pixel_noise + 1, size=X.shape)
X_noisy = X + noise
X_denoised = X_noisy

# for i in range(1, X.shape[0] - 1):
#     for j in range(1, X.shape[1] - 1):
#         X_denoised[i][j] = np.median(
#             [X_denoised[i - 1][j - 1], X_denoised[i - 1][j], X_denoised[i - 1][j + 1], X_denoised[i][j - 1], X_denoised[i][j + 1],
#              X_denoised[i + 1][j - 1], X_denoised[i + 1][j], X_denoised[i + 1][j + 1]]
#         )
plt.imshow(X, cmap=plt.cm.gray)
plt.title('Original')
plt.show()
plt.imshow(X_noisy, cmap=plt.cm.gray)
plt.title('Noisy')
plt.show()
snr = np.mean(X ** 2) / np.mean(X_noisy ** 2)
print("SNR initial: " + str(snr))
# plt.imshow(X_denoised, cmap=plt.cm.gray)
# plt.title('Median')
# plt.show()
X_denoised = ndimage.median_filter(X_noisy, size=5)
plt.imshow(X_denoised, cmap=plt.cm.gray)
plt.show()
snr = np.mean(X ** 2) / np.mean(X_denoised ** 2)
print("SNR using median: " + str(snr))
X_gaussian = ndimage.gaussian_filter(X_noisy, 7)
plt.imshow(X_gaussian, cmap=plt.cm.gray)
plt.show()
snr = np.mean(X ** 2) / np.mean(X_gaussian ** 2)
print("SNR using gaussian filter: " + str(snr))

