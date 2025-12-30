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

sys.setrecursionlimit(2000)


# https://www.geeksforgeeks.org/dsa/huffman-coding-in-python/ de unde m am inspirat
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


def pack_to_binary(all_values, huff_dict):
    bit_string = "".join(huff_dict[val] for val in all_values)

    # Calculez padding-ul pana la urm byte
    padding_len = 8 - (len(bit_string) % 8)
    bit_string += "0" * padding_len

    byte_arr = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_arr.append(int(bit_string[i:i + 8], 2))

    return bytes(byte_arr), padding_len




def comprimareJPEG(path):
    img = matplotlib.image.imread(path)

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
                B[i:i + 8, j:j + 8, c] = b_jpeg

    # Vad daca imaginea se poate imparti in blocuri de 8x8, daca nu ii fac padding cu 0
    print(img[:, :, 0].shape)
    h, w, _ = img.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Fac conversie ycbcr pt ca ochiul uman percepe mai bine intensitatea
    # luminii decat culori in sine si e mai usor de comprimat
    imgYCC = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    h, w, _ = img.shape
    B = np.zeros_like(imgYCC, dtype=np.float32)
    # M am enervat ca era f slow pe o poza f mare, asa ca am incercat sa fac paralel dar nu merge cu MULT mai bine
    means = np.zeros((h // 8, w // 8, 3))  # salvez mediile pt decompresie
    with ThreadPoolExecutor(max_workers=48) as executor:
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

    binary_data, padding = pack_to_binary(all_values_to_encode, huffman_dict)

    with open("compressed_image.bin", "wb") as f:
        f.write(struct.pack('IIB', w, h, padding))

        dict_bytes = pickle.dumps(huffman_dict)
        dict_size_bytes = len(dict_bytes)

        f.write(struct.pack('I', dict_size_bytes))
        f.write(dict_bytes)

        f.write(binary_data)

    return means


def decomprimare_jpeg(path, means):
    with open(path, "rb") as f:
        w, h, padding = struct.unpack('IIB', f.read(9))

        dict_size_bytes = struct.unpack('I', f.read(4))[0]

        dict_data = f.read(dict_size_bytes)
        huff_dict = pickle.loads(dict_data)

        binary_data = f.read()

    bit_string = "".join(f"{byte:08b}" for byte in binary_data)
    if padding > 0:
        bit_string = bit_string[:-padding]

    reverse_huff_dict = {v: k for k, v in huff_dict.items()}

    decoded_values = []
    current_code = ""

    for bit in bit_string:
        current_code += bit
        if current_code in reverse_huff_dict:
            decoded_values.append(reverse_huff_dict[current_code])
            current_code = ""

    img_reconstructed = np.zeros((h, w, 3), dtype=np.float32)

    def inverse_zigzag(flat_vector_64):
        block_flat = np.zeros(64, dtype=np.float32)
        for i, idx in enumerate(zigzag_indices):
            block_flat[idx] = flat_vector_64[i]
        return block_flat.reshape((8, 8))

    idx_counter = 0
    B_recovered = np.zeros((h, w, 3), dtype=np.float32)

    for i in range(0, h, 8):
        for j in range(0, w, 8):
            for c in range(3):
                vector_zigzag = decoded_values[idx_counter: idx_counter + 64]
                idx_counter += 64

                block_quantized = inverse_zigzag(vector_zigzag)
                B_recovered[i:i + 8, j:j + 8, c] = block_quantized

    def worker_decomp(i):
        for j in range(0, w, 8):
            for c in range(3):
                b_jpeg = B_recovered[i:i + 8, j:j + 8, c]

                b = b_jpeg * Q_jpeg

                x_reconst = idctn(b, norm='ortho')
                x_final = x_reconst + means[i // 8, j // 8, c]

                img_reconstructed[i:i + 8, j:j + 8, c] = x_final

    with ThreadPoolExecutor(max_workers=48) as executor:
        for i in range(0, h, 8):
            executor.submit(worker_decomp, i)

    img_final_uint8 = np.clip(img_reconstructed, 0, 255).astype(np.uint8)

    img_rgb = cv2.cvtColor(img_final_uint8, cv2.COLOR_YCrCb2RGB)

    return img_rgb


path = "sample3.bmp"
means = comprimareJPEG(path)
img_rgb = decomprimare_jpeg("compressed_image.bin", means)
fig, ax = plt.subplots(1, 2, figsize=(15, 7))
img = matplotlib.image.imread(path)
ax[0].imshow(img_rgb)
ax[0].set_title("Fake JPEG Compressed")
ax[1].imshow(img)
ax[1].set_title("Original")
plt.show()
