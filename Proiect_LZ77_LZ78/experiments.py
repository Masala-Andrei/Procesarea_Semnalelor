import math
import time
import numpy as np
import bitarray
import os
import struct
import matplotlib.pyplot as plt

# =====================================================
# Pentru urmatoarele teste, nu voi salva efectiv fisierele comprimate, doar voi simula ca le salvez, calculand
# dimensiunea ocupată
# =====================================================


WINDOW_SIZE = 2 ** 15  # size-ul sliding window-ului
LOOKAHEAD = 15
TOKEN_FORMAT = '>HHB'  # distanta, lungime, byte
TOKEN_SIZE = 5

# TESTE PT LZ77

# def findLongestMatch(data, cursor, window_size=WINDOW_SIZE, lookahead=LOOKAHEAD):
#     data_len = len(data)
#
#     # Daca nu mai avem destule caractere pentru un match minim (3), ne oprim
#     if cursor + 3 > data_len:
#         return 0, 0
#
#     # Lookahead buffer-ul este portiunea de date din fata cursorului
#     # Limitata de LOOKAHEAD_SIZE sau de sfarsitul fisierului
#     lookahead_limit = min(lookahead, data_len - cursor - 1)
#
#     # "Seed"-ul ramane de 3 caractere pentru a incepe cautarea
#     seed = data[cursor: cursor + 3]
#
#     best_len = 0
#     best_dist = 0
#
#     # Search buffer-ul este portiunea de date din spatele cursorului
#     search_start = max(0, cursor - window_size)
#     search_buffer = data[search_start: cursor]
#
#     # Cautam seed-ul in Search Buffer
#     pos = search_buffer.find(seed)
#
#     while pos != -1:
#         real_pos = search_start + pos
#         current_len = 0
#
#         # Verificam cat de mult se potriveste, dar fara sa depasim Lookahead Buffer-ul
#         while (current_len < lookahead_limit and
#                data[real_pos + current_len] == data[cursor + current_len]):
#             current_len += 1
#
#         if current_len > best_len:
#             best_len = current_len
#             best_dist = cursor - real_pos
#
#         if best_len == lookahead_limit:
#             break
#
#         pos = search_buffer.find(seed, pos + 1)
#
#     return best_dist, best_len
#
#
# def run_experiment(data, win_size, lookahead):
#     cursor = 0
#     data_len = len(data)
#     tokens_count = 0
#     start_time = time.time()
#
#     while cursor < data_len:
#         dist, length = findLongestMatch(data, cursor, win_size, lookahead)
#         cursor += (length + 1)
#         tokens_count += 1
#
#     duration = time.time() - start_time
#     # Calculam marimea: 4 octeti + (numar tokeni * 5 octeti)
#     compressed_size = 4 + (tokens_count * 5)
#     return compressed_size / 1024, duration
#
#
# windows = [1024, 2048, 32768, 65536]  # 1, 2, 32, 64 KB
# lookaheads = [15, 63, 127]
# FISIERE_TEST = ["world192.txt", "xml.txt", "poza.txt"]
#
# for fisier in FISIERE_TEST:
#
#     with open(fisier, 'rb') as f:
#         data = f.read()
#     original_size_kb = len(data) / 1024
#
#     results_size = {lh: [] for lh in lookaheads}
#     results_time = {lh: [] for lh in lookaheads}
#
#     print(f"Incepem testele pe {fisier} ({original_size_kb:.2f} KB)...")
#     for lh in lookaheads:
#         for win in windows:
#             print(f"Rulare: Window={win}, Lookahead={lh}...", end=" ", flush=True)
#             size, duration = run_experiment(data, win, lh)
#             results_size[lh].append(size)
#             results_time[lh].append(duration)
#             print(f"Gata ({duration:.2f}s)")
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
#
#     x_labels = ['1 KB', '2 KB', '32 KB']
#
#     for lh in lookaheads:
#         ax1.plot(windows, results_size[lh], marker='o', linewidth=2, label=f'Lookahead {lh}')
#         ax2.plot(windows, results_time[lh], marker='s', linewidth=2, label=f'Lookahead {lh}')
#
#     # Plot Dimensiune
#     ax1.axhline(y=original_size_kb, color='r', linestyle='--', label='Original Size')
#     ax1.set_title(f'Eficienta Compresiei pe {fisier}', fontsize=12)
#     ax1.set_xlabel('Search Window Size (Bytes)')
#     ax1.set_ylabel('Marime Comprimata (KB)')
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)
#
#     # Plot Timp
#     ax2.set_title('Timp de Executie (Secunde)', fontsize=12)
#     ax2.set_xlabel('Search Window Size (Bytes)')
#     ax2.set_ylabel('Secunde')
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
#
#     plt.tight_layout()
#     plt.show()
#
#
# # TESTE PT LZ78
# def compress_LZ78_variants(input_path, output_base):
#     with open(input_path, 'rb') as f:
#         data = f.read()
#     data += b'$'
#     data_len = len(data)
#
#     results = {}
#
#     # Dict 64k (var 1)
#     print("Rulare Varianta 1: Limita 64k (Static)...")
#     out_v1 = output_base + "_v1_static.bin"
#     dictionary = {b'': 0}
#     dict_size = 1
#     tokens = []
#     cursor = 0
#     while cursor < data_len:
#         current_string = b''
#         match_index = 0
#         while cursor < data_len:
#             next_string = current_string + data[cursor:cursor + 1]
#             if next_string in dictionary:
#                 match_index = dictionary[next_string]
#                 current_string = next_string
#                 cursor += 1
#             else:
#                 break
#         next_char = data[cursor] if cursor < data_len else 0
#         cursor += 1
#         tokens.append(struct.pack('>HB', match_index, next_char))
#         if dict_size < 65535:
#             dictionary[current_string + bytes([next_char])] = dict_size
#             dict_size += 1
#     with open(out_v1, 'wb') as f:
#         f.write(b''.join(tokens))
#     results['64k Static'] = os.path.getsize(out_v1) / 1024
#
#     # Dict 4 giga (var 2)
#     out_v2 = output_base + "_v2_large.bin"
#     dictionary = {b'': 0}
#     dict_size = 1
#     tokens = []
#     cursor = 0
#     while cursor < data_len:
#         current_string = b''
#         match_index = 0
#         while cursor < data_len:
#             next_string = current_string + data[cursor:cursor + 1]
#             if next_string in dictionary:
#                 match_index = dictionary[next_string]
#                 current_string = next_string
#                 cursor += 1
#             else:
#                 break
#         next_char = data[cursor] if cursor < data_len else 0
#         cursor += 1
#         # Folosim >IB (4 octeti pentru index + 1 pentru caracter)
#         tokens.append(struct.pack('>IB', match_index, next_char))
#         if dict_size < 4294967295:
#             dictionary[current_string + bytes([next_char])] = dict_size
#             dict_size += 1
#     with open(out_v2, 'wb') as f:
#         f.write(b''.join(tokens))
#     results['4G Constant'] = os.path.getsize(out_v2) / 1024
#
#     # Reset dict (var 3)
#     out_v3 = output_base + "_v3_reset.bin"
#     dictionary = {b'': 0}
#     dict_size = 1
#     tokens = []
#     cursor = 0
#     while cursor < data_len:
#         current_string = b''
#         match_index = 0
#         while cursor < data_len:
#             next_string = current_string + data[cursor:cursor + 1]
#             if next_string in dictionary:
#                 match_index = dictionary[next_string]
#                 current_string = next_string
#                 cursor += 1
#             else:
#                 break
#         next_char = data[cursor] if cursor < data_len else 0
#         cursor += 1
#         tokens.append(struct.pack('>HB', match_index, next_char))
#         if dict_size < 65535:
#             dictionary[current_string + bytes([next_char])] = dict_size
#             dict_size += 1
#         else:
#             dictionary = {b'': 0}
#             dict_size = 1
#     with open(out_v3, 'wb') as f:
#         f.write(b''.join(tokens))
#     results['64k Reset'] = os.path.getsize(out_v3) / 1024
#
#     return results, data_len / 1024
#
#
# input_file = "mozilla.txt"
# if os.path.exists(input_file):
#     stats, original_size = compress_LZ78_variants(input_file, "comprimat")
#
#     # Plotting
#     names = list(stats.keys())
#     values = list(stats.values())
#
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(names, values, color=['#ff9999', '#66b3ff', '#99ff99'])
#     plt.axhline(y=original_size, color='r', linestyle='--', label=f'Original ({original_size:.0f} KB)')
#
#     plt.title('Comparație Variante LZ78: mozilla.txt', fontsize=14)
#     plt.ylabel('Dimensiune fișier comprimat (KB)')
#     plt.legend()
#
#     for bar in bars:
#         yval = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width() / 2, yval + 100, f'{yval:.0f} KB', ha='center', va='bottom')
#
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()

# CONFRUNTARE LZ77 SI LZ78
LZ77_WIN = 32768
LZ77_LOOKAHEAD = 63
LZ77_FORMAT = '>HHB'
FISIERE_TEST = ["world192.txt", "xml.txt", "poza.txt", "lorem.txt", "log.txt"]


def compressLZ77_best(data):
    cursor = 0
    data_len = len(data)
    tokens_count = 0
    start_time = time.time_ns()

    while cursor < data_len:
        if cursor + 3 > data_len:
            best_dist, best_len = 0, 0
        else:
            lookahead_limit = min(LZ77_LOOKAHEAD, data_len - cursor - 1)
            seed = data[cursor: cursor + 3]
            best_len = 0
            best_dist = 0
            search_start = max(0, cursor - LZ77_WIN)
            search_buffer = data[search_start: cursor]

            pos = search_buffer.find(seed)
            while pos != -1:
                real_pos = search_start + pos
                current_len = 0
                while (current_len < lookahead_limit and
                       data[real_pos + current_len] == data[cursor + current_len]):
                    current_len += 1

                if current_len > best_len:
                    best_len = current_len
                    best_dist = cursor - real_pos
                if best_len == lookahead_limit:
                    break
                pos = search_buffer.find(seed, pos + 1)

        if best_len > 0:
            cursor += best_len + 1
        else:
            cursor += 1
        tokens_count += 1

    duration = (time.time_ns() - start_time) / 1e+9
    # Dimensiune = 4 bytes (header) + tokeni * 5 bytes
    compressed_size = 4 + (tokens_count * 5)
    return compressed_size / 1024, duration


def compressLZ78_best(data):
    data_len = len(data)
    dictionary = {b'': 0}
    dict_size = 1
    tokens_count = 0
    cursor = 0

    start_time = time.time_ns()

    while cursor < data_len:
        current_string = b''
        match_index = 0

        while cursor < data_len:
            char_byte = data[cursor:cursor + 1]
            next_string = current_string + char_byte
            if next_string in dictionary:
                match_index = dictionary[next_string]
                current_string = next_string
                cursor += 1
            else:
                break

        if cursor < data_len:
            cursor += 1
        tokens_count += 1

        if dict_size < 65535:
            char_to_add = data[cursor - 1:cursor] if cursor > 0 else b'\x00'
            dictionary[current_string + char_to_add] = dict_size
            dict_size += 1
        else:
            dictionary = {b'': 0}
            dict_size = 1

    duration = (time.time_ns() - start_time) / 1e+9
    # Dimensiune = token-uri * 3 bytes
    compressed_size = tokens_count * 3
    return compressed_size / 1024, duration


rezultate_dimensiune = {'LZ77': [], 'LZ78': [], 'Original': []}
rezultate_timp = {'LZ77': [], 'LZ78': []}
fisiere_valide = []

for fisier in FISIERE_TEST:
    print(f"\nProcesare: {fisier}")
    with open(fisier, 'rb') as f:
        data = f.read()

    fisiere_valide.append(fisier)
    original_size_kb = len(data) / 1024
    rezultate_dimensiune['Original'].append(original_size_kb)

    # Run LZ77
    size77, time77 = compressLZ77_best(data)
    rezultate_dimensiune['LZ77'].append(size77)
    rezultate_timp['LZ77'].append(time77)
    print(f"[{size77:.3f} KB | {time77:.3f} s]")

    # Run LZ78
    size78, time78 = compressLZ78_best(data)
    rezultate_dimensiune['LZ78'].append(size78)
    rezultate_timp['LZ78'].append(time78)
    print(f"[{size78:.3f} KB | {time78:.3f} s]")

if fisiere_valide:
    x = range(len(fisiere_valide))
    width = 0.25

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    rects1 = ax1.bar([i - width for i in x], rezultate_dimensiune['Original'], width, label='Original', color='gray',
                     alpha=0.5)
    rects2 = ax1.bar(x, rezultate_dimensiune['LZ77'], width, label='LZ77', color='#4a90e2')
    rects3 = ax1.bar([i + width for i in x], rezultate_dimensiune['LZ78'], width, label='LZ78',
                     color='#50c878')

    ax1.set_ylabel('Dimensiune (KB)')
    ax1.set_title('Comparatie Rata de Compresie')
    ax1.set_xticks(x)
    ax1.set_xticklabels(fisiere_valide)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.4)


    def autolabel(rects, ax):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)


    autolabel(rects1, ax1)
    autolabel(rects2, ax1)
    autolabel(rects3, ax1)

    rects_t1 = ax2.bar([i - width / 2 for i in x], rezultate_timp['LZ77'], width, label='LZ77', color='#4a90e2')
    rects_t2 = ax2.bar([i + width / 2 for i in x], rezultate_timp['LZ78'], width, label='LZ78', color='#50c878')

    ax2.set_ylabel('Timp Executie (Secunde)')
    ax2.set_title('Comparatie Performanta')
    ax2.set_xticks(x)
    ax2.set_xticklabels(fisiere_valide)
    ax2.legend()
    ax2.grid(axis='y', linestyle='--', alpha=0.4)

    autolabel(rects_t1, ax2)
    autolabel(rects_t2, ax2)

    plt.tight_layout()
    plt.show()
