import math
import time
import numpy as np
import bitarray
import os
import struct
import matplotlib.pyplot as plt

WINDOW_SIZE = 2 ** 15  # size-ul sliding window-ului
LOOKAHEAD = 15
TOKEN_FORMAT = '>HHB'  # distanta, lungime, byte
TOKEN_SIZE = 5


def findLongestMatch(data, cursor, window_size=WINDOW_SIZE, lookahead=LOOKAHEAD):
    data_len = len(data)

    # Daca nu mai avem destule caractere pentru un match, opresc
    if cursor + 3 > data_len:
        return 0, 0

    # Lookahead buffer-ul este portiunea de date din fata cursorului
    lookahead_limit = min(lookahead, data_len - cursor - 1)

    # Verific doar de la al 3-lea caracter in fata
    seed = data[cursor: cursor + 3]

    best_len = 0
    best_dist = 0

    search_start = max(0, cursor - window_size)
    search_buffer = data[search_start: cursor]

    pos = search_buffer.find(seed)

    while pos != -1:
        real_pos = search_start + pos
        current_len = 0

        # Verific sa se potriveasca cat mai mult din lookahead
        while (current_len < lookahead_limit and
               data[real_pos + current_len] == data[cursor + current_len]):
            current_len += 1

        if current_len > best_len:
            best_len = current_len
            best_dist = cursor - real_pos

        if best_len == lookahead_limit:
            break

        pos = search_buffer.find(seed, pos + 1)

    return best_dist, best_len


def compressLZ77(input, output):
    with open(input, 'rb') as f:
        data = f.read()

    cursor = 0
    data_len = len(data)
    tokens = []
    start = time.time_ns()
    while cursor < data_len:
        dist, length = findLongestMatch(data, cursor)

        # Daca gasesc match, ma duc la urmatorul index
        if length > 0:
            next_index = cursor + length
            c = data[next_index] if next_index < data_len else 0
        else:
            dist = 0
            length = 0
            c = data[cursor]

        tokens.append(struct.pack(TOKEN_FORMAT, dist, length, c))
        cursor += length + 1

    with open(output, 'wb') as f:
        f.write(struct.pack('>I', data_len))  # marime originala (4B)
        for t in tokens:
            f.write(t)

    end = time.time_ns()
    print(f"[LZ77] Gata in {end - start:.2f} secunde.")


def decompressLZ77(input, output_f):
    with open(input, 'rb') as f:
        size = struct.unpack('>I', f.read(4))[0]
        output = bytearray()

        while len(output) < size:
            chunk = f.read(TOKEN_SIZE)
            if not chunk or len(chunk) < TOKEN_SIZE:
                break

            dist, length, char = struct.unpack(TOKEN_FORMAT, chunk)

            if dist > 0:
                start_copy = len(output) - dist
                for i in range(length):
                    output.append(output[start_copy + i])


            if len(output) < size:
                output.append(char)

        with open(output_f, 'wb') as f:
            f.write(output)


def compress_LZ78(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = f.read()

    # Adaugam un marker de sfarsit sub forma de byte
    data += b'$'
    data_len = len(data)

    dictionary = {b'': 0}
    dict_size = 1

    tokens = []
    cursor = 0

    start_time = time.time_ns()

    while cursor < data_len:
        current_string = b''
        match_index = 0

        # Cautam cel mai lung prefix in dictionar
        while cursor < data_len:
            char_byte = data[cursor:cursor+1]
            next_string = current_string + char_byte

            if next_string in dictionary:
                match_index = dictionary[next_string]
                current_string = next_string
                cursor += 1
            else:
                break

        if cursor < data_len:
            next_char = data[cursor]
            cursor += 1
        else:
            next_char = 0

        tokens.append(struct.pack('>HB', match_index, next_char))

        if dict_size < 65535:
            new_phrase = current_string + bytes([next_char])
            dictionary[new_phrase] = dict_size
            dict_size += 1
        else:
            # Dictionarul e plin, il resetam ca sa poata invata fraze noi
            dictionary = {b'': 0}
            dict_size = 1

    with open(output_path, 'wb') as f:
        f.write(b''.join(tokens))

    end_time = time.time_ns()
    print(f"[LZ78] Finalizat in {(end_time - start_time) / 1e9:.2f} secunde.")
    print(f"[LZ78] Original: {data_len} -> Comprimat: {os.path.getsize(output_path)}")


def decompress_LZ78(input_path, output_path):
    with open(input_path, 'rb') as f:
        compressed_data = f.read()

    dictionary = {0: b''}
    dict_size = 1

    decoded_parts = []
    offset = 0

    while offset + 3 <= len(compressed_data):
        idx, char = struct.unpack('>HB', compressed_data[offset:offset + 3])
        offset += 3

        if idx in dictionary:
            prefix = dictionary[idx]
        else:
            break

        phrase = prefix + bytes([char])
        decoded_parts.append(phrase)

        if dict_size < 65535:
            dictionary[dict_size] = phrase
            dict_size += 1
        else:
            dictionary = {0: b''}
            dict_size = 1

    full_output = b''.join(decoded_parts)


    if b'$' in full_output:
        full_output = full_output.rsplit(b'$', 1)[0]
    print(dict_size)

    with open(output_path, 'wb') as f:
        f.write(full_output)


input = "xml.txt"
bin = "bible.bin"
bin_lz78 = "bible_lz78.bin"
decompressed = "bible_dec.txt"
decompressed_lz78 = "bible_dec_lz78.txt"

# compressLZ77(input, bin)
# decompressLZ77(bin, decompressed)

# compress_LZ78(input, bin_lz78)
# decompress_LZ78(bin_lz78, decompressed_lz78)

