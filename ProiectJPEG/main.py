import pickle
import struct
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot
import matplotlib.image
from scipy import misc, ndimage
from scipy.fft import dctn, idctn
import cv2
from concurrent.futures import ThreadPoolExecutor
import heapq
import sys

# def huffman_coding(option):
#     if option == 1:
#
#     else:


Q_jpeg = [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 28, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

zigzag_indices = [
    0, 1, 5, 6, 14, 15, 27, 28, 2, 4, 7, 13, 16, 26, 29, 42,
    3, 8, 12, 17, 25, 30, 41, 43, 9, 11, 18, 24, 31, 40, 44, 53,
    10, 19, 23, 32, 39, 45, 52, 54, 20, 22, 33, 38, 46, 51, 55, 60,
    21, 34, 37, 47, 50, 56, 59, 61, 35, 36, 48, 49, 57, 58, 62, 63
]

# https://www.geeksforgeeks.org/dsa/huffman-coding-in-python/
sys.setrecursionlimit(2000)


class HuffmanNode:
    def __init__(self, freq, symbol=None):
        self.freq = freq
        self.symbol = symbol
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def build_huff_tree(chars, freqs):
    priority_queue = [HuffmanNode(f, char) for char, f in zip(chars, freqs)]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        # Iau primele 2 noduri cu frecv cea mai mica si le combin facand alt nod
        left_child = heapq.heappop(priority_queue)
        right_child = heapq.heappop(priority_queue)
        merged_node = HuffmanNode(freq=left_child.freq + right_child.freq)
        merged_node.left = left_child
        merged_node.right = right_child
        heapq.heappush(priority_queue, merged_node)

    return priority_queue[0]  # returnez varful


def build_huff_codes(node, code="", huffman_codes=None):
    if huffman_codes is None:
        huffman_codes = {}

    if node is not None:
        if node.symbol is not None:
            huffman_codes[node.symbol] = code
        build_huff_codes(node.left, code + "0", huffman_codes)
        build_huff_codes(node.right, code + "1", huffman_codes)

    return huffman_codes


# Fac conversie ycbcr pt ca ochiul uman percepe mai bine intensitatea
# luminii decat culori in sine si e mai usor de comprimat

img = matplotlib.image.imread("sample3.bmp")

# Vad daca imaginea se poate imparti in blocuri de 8x8, daca nu ii fac padding cu 0
print(img[:, :, 0].shape)
h, w, _ = img.shape
pad_h = (8 - (h % 8)) % 8
pad_w = (8 - (w % 8)) % 8
img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

imgYCC = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
h, w, _ = img.shape
B = np.zeros_like(imgYCC, dtype=np.float32)
A = np.zeros_like(imgYCC, dtype=np.float32)


def worker(i):
    for j in range(0, w, 8):
        for c in range(3):
            x = imgYCC[i:i + 8, j:j + 8, c]
            # Scot media din semnal
            mean = np.mean(x)
            x = x - mean
            means[i // 8, j // 8, c] = mean
            b = dctn(x, norm='ortho')
            b_jpeg = np.round(b / Q_jpeg)
            temp = b_jpeg * Q_jpeg
            temp = idctn(temp, norm='ortho')
            temp = temp + mean
            A[i:i + 8, j:j + 8, c] = temp
            B[i:i + 8, j:j + 8, c] = b_jpeg


# M am enervat ca era f slow pe o poza f mare, asa ca am incercat sa fac paralel dar nu merge cu MULT mai bine
means = np.zeros((h // 8, w // 8, 3))  # salvez mediile pt decompresie
with ThreadPoolExecutor(max_workers=24) as executor:
    for i in range(0, h, 8):
        executor.submit(worker, i)

all_values_to_encode = []
# Comprimam
for i in range(0, h, 8):
    for j in range(0, w, 8):
        for c in range(3):
            bloc = B[i:i + 8, j:j + 8, c]
            flat_block = bloc.flatten()
            zigzag_vector = [int(flat_block[idx]) for idx in zigzag_indices]
            all_values_to_encode.extend(zigzag_vector)

x = Counter(all_values_to_encode)
symbols = list(x.keys())
freqs = list(x.values())
root = build_huff_tree(symbols, freqs)
huffman_dict = build_huff_codes(root, "", None)
print(huffman_dict)


def pack_to_binary(all_values, huff_dict):
    bit_string = "".join(huff_dict[val] for val in all_values)

    # Calculez padding-ul (bitii extra pana la un octet complet)
    padding_len = (8 - len(bit_string) % 8) % 8
    bit_string += "0" * padding_len

    byte_arr = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_arr.append(int(bit_string[i:i + 8], 2))

    return bytes(byte_arr), padding_len


binary_data, padding = pack_to_binary(all_values_to_encode, huffman_dict)
l = []
for i in huffman_dict.keys():
    l.append((i, huffman_dict[i]))
print(l)

with open("compressed_image.bin", "wb") as f:
    # Salvam Header-ul: Latime(I), Inaltime(I), Padding(B) -> 9 octeți
    f.write(struct.pack('IIB', w, h, padding))

    # Salvam dicționarul (serializat)
    dict_bytes = pickle.dumps(l)
    f.write(struct.pack('I', len(l)))  # Scriem lungimea dicționarului
    f.write(dict_bytes)

    # Scriem bitii imaginii
    f.write(binary_data)


fig, ax = plt.subplots(1, 2, figsize=(15, 7))
A_clipped = np.clip(A, 0, 255)
A_final = A_clipped.astype(np.uint8)
A_final = cv2.cvtColor(A_final, cv2.COLOR_YCrCb2RGB)
# y_jpeg = idctn(B)
ax[0].imshow(A_final)
ax[0].set_title("Fake JPEG Compressed")
ax[1].imshow(img)
ax[1].set_title("Original")
plt.show()
