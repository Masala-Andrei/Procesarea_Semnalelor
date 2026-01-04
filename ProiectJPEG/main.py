import pickle
import struct
import time
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
Q_jpeg_np = np.array(Q_jpeg, dtype=np.float32)
zigzag_indices_np = np.array(zigzag_indices, dtype=np.int32)


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
    bit_list = [huff_dict[val] for val in all_values]
    bit_string = "".join(bit_list)

    # Calculez padding-ul pana la urm byte
    padding_len = 8 - (len(bit_string) % 8)
    bit_string += "0" * padding_len

    byte_arr = bytearray()
    for i in range(0, len(bit_string), 8):
        byte_arr.append(int(bit_string[i:i + 8], 2))

    return bytes(byte_arr), padding_len


def comprimareJPEG(path):
    img = matplotlib.image.imread(path)

    # Vad daca imaginea se poate imparti in blocuri de 8x8, daca nu ii fac padding cu 0
    # print(img[:, :, 0].shape)
    h1, w1, _ = img.shape
    pad_h = (8 - (h1 % 8)) % 8
    pad_w = (8 - (w1 % 8)) % 8
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Fac conversie ycbcr pt ca ochiul uman percepe mai bine intensitatea
    # luminii decat culori in sine si e mai usor de comprimat
    imgYCC = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    h, w, _ = img.shape

    # Voi schimba abordarea si o sa dau reshape la matrice ca sa nu fac totul in py ca se misca greu
    blocks = imgYCC.reshape(h // 8, 8, w // 8, 8, 3).swapaxes(1, 2).reshape(-1, 8, 8, 3)
    # Pe scurt, in imgYCC aveam 3 matrici de h x w, acum transform totul intr o matrice mare de matrici de
    # 8 x 8 x 3 ca sa pot aplica totul in blocuri decat secvential cu for-uri

    # Calculez mediile pentru fiecare block si las np sa fac in spate
    means = np.mean(blocks, axis=(1, 2), keepdims=True)
    blocks_centered = blocks - means

    bs = dctn(blocks_centered, axes=(1, 2), norm='ortho')
    # Trb sa extind qjpeg sa aratae ca bs
    b_jpegs = np.round(bs / Q_jpeg_np[None, :, :, None])

    # Dau flat in vector
    flat_blocks = b_jpegs.reshape(-1, 64, 3)

    zigzag_vectors = flat_blocks[:, zigzag_indices_np, :]

    all_values_to_encode = zigzag_vectors.flatten()

    means = means.reshape(h // 8, w // 8, 3)

    # Sunt de a dreptul uimit cat de prost am putut eu sa scriu cod si cat de repede face numpy-ul
    # Voi lasa si codul vechi mai jos just for reference ca mi se pare funny
    #################################################################################3
    # means = np.zeros((h // 8, w // 8, 3))  # salvez mediile pt decompresie
    # all_values_to_encode = np.zeros(h * w * 3)
    # zigzag_indices_np = np.array(zigzag_indices)
    # B = np.zeros_like(imgYCC, dtype=np.float32)

    # start = time.time_ns()
    # start_pos = 0
    # for i in range(0, h, 8):
    #     for j in range(0, w, 8):
    #         for c in range(3):
    #             x = imgYCC[i:i + 8, j:j + 8, c]
    #             # Scot media din semnal
    #             mean = np.mean(x)
    #             x = x - mean
    #             means[i // 8, j // 8, c] = mean
    #             b = dctn(x, norm='ortho')
    #             b_jpeg = np.round(b / Q_jpeg_np)
    #             flat_block = b_jpeg.ravel()
    #             zigzag_vector = flat_block[zigzag_indices_np]
    #             all_values_to_encode[start_pos: start_pos + 64] = zigzag_vector
    #             start_pos += 64
    #
    # end = time.time_ns()
    # print(end - start)
    # Comprimam
    # Am incercat sa pun chestia asta in paralel, aparent imi da cu o secunda mai prost
    # poate pentru ca ii ia mai mult sa tot calculeze pozitia in vector unde trebuie sa dea append
    # decat doar sa calculeze si sa ii dea paste
    # Ori asta, ori e din cauza GIL-ului (global interpreter lock) bcz e cod de py pur
    # for i in range(0, h, 8):
    #     for j in range(0, w, 8):
    #         for c in range(3):
    #             bloc = B[i:i + 8, j:j + 8, c]
    #             flat_block = bloc.flatten()
    #             zigzag_vector = [int(flat_block[idx]) for idx in zigzag_indices]
    #             all_values_to_encode.extend(zigzag_vector)
    ############################################################################################

    x = Counter(all_values_to_encode)
    symbols = list(x.keys())
    freqs = list(x.values())

    root = build_huff_tree(symbols, freqs)
    huffman_dict = build_huff_codes(root, "", None)

    binary_data, padding = pack_to_binary(all_values_to_encode, huffman_dict)

    # Pierd f mult timp pe packing si scriere in fisier, chiar am incercat sa optimizez mai mult dar n am reusit

    with open("compressed_image.bin", "wb") as f:
        f.write(struct.pack('II', w1, h1))
        f.write(struct.pack('IIB', w, h, padding))


        dict_bytes = pickle.dumps(huffman_dict)
        dict_size_bytes = len(dict_bytes)

        f.write(struct.pack('I', dict_size_bytes))
        f.write(dict_bytes)

        f.write(binary_data)

    return means


def decomprimare_jpeg(path, means):
    with open(path, "rb") as f:
        w1, h1 = struct.unpack('II', f.read(8))
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

    decoded_array = np.array(decoded_values, dtype=np.float32)
    flat_blocks = decoded_array.reshape(-1, 64, 3)
    blocks = np.zeros_like(flat_blocks)
    blocks[:, zigzag_indices_np, :] = flat_blocks
    blocks_quantized = blocks.reshape(-1, 8, 8, 3)
    bs = blocks_quantized * Q_jpeg_np[None, :, :, None]

    blocks_reconstructed = idctn(bs, axes=(1, 2), norm='ortho')

    h_blocks, w_blocks = means.shape[0], means.shape[1]
    means_reshaped = means.reshape(-1, 1, 1, 3)
    img_blocks = blocks_reconstructed + means_reshaped

    imgYCC = img_blocks.reshape(h_blocks, w_blocks, 8, 8, 3).swapaxes(1, 2).reshape(h_blocks * 8, w_blocks * 8, 3)
    imgYCC = np.clip(imgYCC, 0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(imgYCC, cv2.COLOR_YCrCb2RGB)
    img_rgb = img_rgb[:h1, :w1]

    return img_rgb


###########################################################################
# c)

# Voi cauta un alpha la care voi imparti matricea q_jpeg astfel incat sa afecteze mse dintre imaginea originala
# Si cea comprimata

def reconstruct(blocks_centered, means, alpha):
    Q_scaled = Q_jpeg_np * alpha
    bs = dctn(blocks_centered, axes=(1, 2), norm='ortho')
    b_quantized = np.round(bs / Q_scaled[None, :, :, None])
    b_recovered = b_quantized * Q_scaled[None, :, :, None]
    blocks_reconstructed = idctn(b_recovered, axes=(1, 2), norm='ortho')
    return blocks_reconstructed + means


def find_mse(img_ycc, target_mse):
    h, w, _ = img_ycc.shape

    blocks = img_ycc.reshape(h // 8, 8, w // 8, 8, 3).swapaxes(1, 2).reshape(-1, 8, 8, 3)
    means = np.mean(blocks, axis=(1, 2), keepdims=True)
    blocks_centered = blocks - means

    # Cautare binara pentru alpha
    low = 0.01
    high = 100.0
    alpha_optim = 1.0
    tolerance = 0.5

    while low < high:
        mid = (low + high) / 2
        img_reconst_blocks = reconstruct(blocks_centered, means, mid)
        current_mse = np.mean((blocks - img_reconst_blocks) ** 2)

        if abs(current_mse - target_mse) < tolerance:
            alpha_optim = mid
            break

        if current_mse < target_mse:
            low = mid
        else:
            high = mid
    else:
        alpha_optim = low

    return alpha_optim


def comprimareJPEG_MSE(path, target_mse):
    img = matplotlib.image.imread(path)

    # Vad daca imaginea se poate imparti in blocuri de 8x8, daca nu ii fac padding cu 0
    # print(img[:, :, 0].shape)
    h1, w1, _ = img.shape
    pad_h = (8 - (h1 % 8)) % 8
    pad_w = (8 - (w1 % 8)) % 8
    img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Fac conversie ycbcr pt ca ochiul uman percepe mai bine intensitatea
    # luminii decat culori in sine si e mai usor de comprimat
    imgYCC = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    alpha = find_mse(imgYCC, target_mse)
    Q_jpeg_np_fake = Q_jpeg_np * alpha
    h, w, _ = img.shape

    # Voi schimba abordarea si o sa dau reshape la matrice ca sa nu fac totul in py ca se misca greu
    blocks = imgYCC.reshape(h // 8, 8, w // 8, 8, 3).swapaxes(1, 2).reshape(-1, 8, 8, 3)
    # Pe scurt, in imgYCC aveam 3 matrici de h x w, acum transform totul intr o matrice mare de matrici de
    # 8 x 8 x 3 ca sa pot aplica totul in blocuri decat secvential cu for-uri

    # Calculez mediile pentru fiecare block si las np sa fac in spate
    means = np.mean(blocks, axis=(1, 2), keepdims=True)
    blocks_centered = blocks - means

    bs = dctn(blocks_centered, axes=(1, 2), norm='ortho')
    # Trb sa extind qjpeg sa aratae ca bs
    b_jpegs = np.round(bs / Q_jpeg_np_fake[None, :, :, None])

    # Dau flat in vector
    flat_blocks = b_jpegs.reshape(-1, 64, 3)

    zigzag_vectors = flat_blocks[:, zigzag_indices_np, :]

    all_values_to_encode = zigzag_vectors.flatten()

    means = means.reshape(h // 8, w // 8, 3)

    x = Counter(all_values_to_encode)
    symbols = list(x.keys())
    freqs = list(x.values())

    root = build_huff_tree(symbols, freqs)
    huffman_dict = build_huff_codes(root, "", None)

    binary_data, padding = pack_to_binary(all_values_to_encode, huffman_dict)

    # Pierd f mult timp pe packing si scriere in fisier, chiar am incercat sa optimizez mai mult dar n am reusit

    with open("compressed_image_mse.bin", "wb") as f:
        f.write(struct.pack('II', w1, h1))
        f.write(struct.pack('IIB', w, h, padding))

        dict_bytes = pickle.dumps(huffman_dict)
        dict_size_bytes = len(dict_bytes)

        f.write(struct.pack('I', dict_size_bytes))
        f.write(dict_bytes)

        f.write(binary_data)

    return means, alpha

def decomprimare_jpeg_MSE(path, means, alpha):
    Q_jpeg_np_fake = Q_jpeg_np * alpha
    with open(path, "rb") as f:
        w1, h1 = struct.unpack('II', f.read(8))
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

    decoded_array = np.array(decoded_values, dtype=np.float32)
    flat_blocks = decoded_array.reshape(-1, 64, 3)
    blocks = np.zeros_like(flat_blocks)
    blocks[:, zigzag_indices_np, :] = flat_blocks
    blocks_quantized = blocks.reshape(-1, 8, 8, 3)
    bs = blocks_quantized * Q_jpeg_np_fake[None, :, :, None]

    blocks_reconstructed = idctn(bs, axes=(1, 2), norm='ortho')

    h_blocks, w_blocks = means.shape[0], means.shape[1]
    means_reshaped = means.reshape(-1, 1, 1, 3)
    img_blocks = blocks_reconstructed + means_reshaped

    imgYCC = img_blocks.reshape(h_blocks, w_blocks, 8, 8, 3).swapaxes(1, 2).reshape(h_blocks * 8, w_blocks * 8, 3)
    imgYCC = np.clip(imgYCC, 0, 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(imgYCC, cv2.COLOR_YCrCb2RGB)
    img_rgb = img_rgb[:h1, :w1]

    return img_rgb

###########################################################################
# d)
# M am decis sa schimb putin logica de cum comprim si decomprim asa ca am rescris putin functiile
# in cazul in care nu mi merge sa le copiez pe alea de mai sus care stiu sigur ca merg
def compress_frame(img_rgb):
    h, w, _ = img_rgb.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    img = np.pad(img_rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)

    # Fac conversie ycbcr pt ca ochiul uman percepe mai bine intensitatea
    # luminii decat culori in sine si e mai usor de comprimat
    imgYCC = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    h, w, _ = img.shape
    blocks = imgYCC.reshape(h // 8, 8, w // 8, 8, 3).swapaxes(1, 2).reshape(-1, 8, 8, 3)
    means = np.mean(blocks, axis=(1, 2), keepdims=True)
    blocks_centered = blocks - means
    bs = dctn(blocks_centered, axes=(1, 2), norm='ortho')
    b_jpegs = np.round(bs / Q_jpeg_np[None, :, :, None])
    flat_blocks = b_jpegs.reshape(-1, 64, 3)
    zigzag_vectors = flat_blocks[:, zigzag_indices_np, :]
    all_values_to_encode = zigzag_vectors.flatten()
    means = means.reshape(h // 8, w // 8, 3)

    x = Counter(all_values_to_encode)
    symbols = list(x.keys())
    freqs = list(x.values())

    root = build_huff_tree(symbols, freqs)
    huffman_dict = build_huff_codes(root, "", None)

    binary_data, padding = pack_to_binary(all_values_to_encode, huffman_dict)

    frame_metadata = {
        'huff': huffman_dict,
        'means': means,
        'pad': padding
    }

    metadata_bytes = pickle.dumps(frame_metadata)

    return metadata_bytes, binary_data


def decompress_frame(metadata_bytes, binary_data):
    meta = pickle.loads(metadata_bytes)
    huff_dict = meta['huff']
    means = meta['means']
    padding = meta['pad']

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

    decoded_array = np.array(decoded_values, dtype=np.float32)
    flat_blocks = decoded_array.reshape(-1, 64, 3)

    blocks = np.zeros_like(flat_blocks)
    blocks[:, zigzag_indices_np, :] = flat_blocks
    blocks_quantized = blocks.reshape(-1, 8, 8, 3)

    b_coefficients = blocks_quantized * Q_jpeg_np[None, :, :, None]
    blocks_reconstructed = idctn(b_coefficients, axes=(1, 2), norm='ortho')
    h_blocks, w_blocks = means.shape[0], means.shape[1]
    means_reshaped = means.reshape(-1, 1, 1, 3)
    img_blocks = blocks_reconstructed + means_reshaped
    imgYCC = img_blocks.reshape(h_blocks, w_blocks, 8, 8, 3).swapaxes(1, 2).reshape(h_blocks * 8, w_blocks * 8, 3)
    img_rgb = cv2.cvtColor(np.clip(imgYCC, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2RGB)

    return img_rgb


def compress_video(path, output):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with open(output, "wb") as f:
        f.write(struct.pack('III', w, h, fps))

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            meta, data = compress_frame(frame_rgb)

            f.write(struct.pack('II', len(meta), len(data)))
            f.write(meta)
            f.write(data)

            frame_index += 1
            if frame_index % 10 == 0:
                print(f"Progres: {frame_index}/{total_frames} cadre comprimate.")
    cap.release()


def decompress_video(input, output):
    with open(input, "rb") as f:
        w, h, fps = struct.unpack('III', f.read(12))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_vid = cv2.VideoWriter(output, fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            chunk_header = f.read(8)
            if not chunk_header:
                break

            l_meta, l_data = struct.unpack('II', chunk_header)
            meta = f.read(l_meta)
            data = f.read(l_data)

            img_rgb = decompress_frame(meta, data)
            # Scot paddingul
            img_bgr = cv2.cvtColor(img_rgb[:h, :w], cv2.COLOR_RGB2BGR)

            out_vid.write(img_bgr)
            frame_idx += 1
            if frame_idx % 10 == 0:
                print(f"Progres: {frame_idx} cadre decomprimate.")

        out_vid.release()


option = input("Choose option:"
               "\n1) Comprimare JPEG"
               "\n2) Comprimare JPEG cu MSE"
               "\n3) Comprimare video\n")
match option:
    case "1":
        path = "sample2.bmp"
        means = comprimareJPEG(path)
        img_rgb = decomprimare_jpeg("compressed_image.bin", means)
        fig, ax = plt.subplots(1, 2, figsize=(15, 7))
        img = matplotlib.image.imread(path)
        ax[0].imshow(img_rgb)
        ax[0].set_title("Fake JPEG Compressed")
        ax[1].imshow(img)
        ax[1].set_title("Original")
        plt.show()
    case "2":
        mse = int(input("Please provide a mse threshold: "))
        path = "sample1.bmp"
        means_mse, alpha = comprimareJPEG_MSE(path, mse)
        img_rgb_mse = decomprimare_jpeg_MSE("compressed_image_mse.bin", means_mse, alpha)
        means = comprimareJPEG(path)
        img_rgb = decomprimare_jpeg("compressed_image.bin", means)
        fig, ax = plt.subplots(1, 3, figsize=(15, 7))
        img = matplotlib.image.imread(path)
        ax[0].imshow(img_rgb)
        ax[0].set_title("Fake JPEG Compressed")
        ax[1].imshow(img_rgb_mse)
        ax[1].set_title(f"Fake JPEG with MSE of {mse}")
        ax[2].imshow(img)
        ax[2].set_title("Original")
        plt.tight_layout()
        plt.show()
    case "3":
        compress_video("video1.mp4", "compressed_video1.bin")
        decompress_video("compressed_video1.bin", "decompressed_video1.mp4")
    case _:
        print("Wrong option")
