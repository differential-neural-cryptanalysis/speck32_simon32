#!/usr/bin/python3
import numpy as np
from os import urandom
from collections import deque
from math import log2

inDIff = []
inDIff.append((0, 0x0040))
inDIff.append([])
inDIff.append((0x0100, 0x0440))
inDIff.append((0x0440, 0x1000))  # 3
inDIff.append((0x1000, 0x4440))  # 4
inDIff.append((0x4000, 0x5101))  # 5


def WORD_SIZE():
    return 16

def ALPHA():
    return(1);

def BETA():
    return(8);

def GAMMA():
    return(2);

z_0 = 0b01100111000011010100100010111110110011100001101010010001011111
MOD_MASK = (2 ** WORD_SIZE()) - 1
c = MOD_MASK ^ 3

def rol(x,k):
    return(((x << k) & MOD_MASK) | (x >> (WORD_SIZE() - k)));

def ror(x,k):
    return((x >> k) | ((x << (WORD_SIZE() - k)) & MOD_MASK));

def expand_keys(key, rounds):
    k_tmp = deque(key)
    keys = []
    for i in range(rounds):
        rs_3 = ((k_tmp[0] << (WORD_SIZE() - 3)) + (k_tmp[0] >> 3)) & MOD_MASK
        # m= 4
        rs_3 = rs_3 ^ k_tmp[2]
        #
        rs_1 = ((rs_3 << (WORD_SIZE() - 1)) + (rs_3 >> 1)) & MOD_MASK
        tmp = rs_3 ^ rs_1
        z = (z_0 >> (i % 62)) & 1
        new_k = z ^ c ^ k_tmp[3] ^ tmp
        keys.append(k_tmp.pop())
        k_tmp.appendleft(new_k)
    return keys

def round_function(x):
    ls_1_x = ((x >> (WORD_SIZE() - 1)) + (x << 1)) & MOD_MASK
    ls_2_x = ((x >> (WORD_SIZE() - 2)) + (x << 2)) & MOD_MASK
    ls_8_x = ((x >> (WORD_SIZE() - 8)) + (x << 8)) & MOD_MASK
    f_value = (ls_1_x & ls_8_x) ^ ls_2_x
    return f_value

def one_round_encrypt(plain, key):
    x, y = plain[0], plain[1]

    f_value = round_function(x)
    xor_1 = f_value ^ y
    y = x
    x = xor_1 ^ key
    return x, y

def encryption(plain_pair, keys):
    x, y = plain_pair[0], plain_pair[1]
    for key in keys:
        x, y = one_round_encrypt((x, y), key)
    return x, y

def one_round_decrypt(cipher, key):
    x, y = cipher[0], cipher[1]
    f_value = round_function(y)
    xor_1 = f_value ^ x
    x = y
    y = xor_1 ^ key
    return x, y

def decryption(cipher_pair, keys):
    x, y = cipher_pair[0], cipher_pair[1]
    for key in reversed(keys):
        x, y = one_round_decrypt((x, y), key)
    return x, y

def dec_one_round_eq(c,k):
    c0, c1 = c[0], c[1];
    t1 = c1 ^ k
    t0 = (rol(t1, ALPHA()) & rol(t1, BETA())) ^ rol(c1, GAMMA());
    c0 = c0 ^ t0;
    return(c1,c0);

def convert_to_bits(arr):
    X = np.zeros((4 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(4 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X

def diff_convert_to_bits(arr):
    X = np.zeros((2 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(2 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X

def value_diff_convert_to_bits(arr):
    X = np.zeros((3 * WORD_SIZE(), len(arr[0])), dtype=np.uint8)
    for i in range(3 * WORD_SIZE()):
        index = i // WORD_SIZE()
        offset = WORD_SIZE() - (i % WORD_SIZE()) - 1
        X[i] = (arr[index] >> offset) & 1
    X = X.transpose()
    return X

def make_datas_diff(n, rounds, diff=(0, 0x0040)):
    print("Make_data Input diff: ", diff)
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_random_samples = np.sum(Y == 0)
    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys = expand_keys(key, rounds)
    plain_01 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_02 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_11 = plain_01 ^ diff[0]
    plain_12 = plain_02 ^ diff[1]
    plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    cipher_01, cipher_02 = encryption((plain_01, plain_02), keys)
    cipher_11, cipher_12 = encryption((plain_11, plain_12), keys)
    delta_x = cipher_01 ^ cipher_11
    delta_y = cipher_02 ^ cipher_12
    X = diff_convert_to_bits([delta_x, delta_y])
    return X, Y


def make_datas(n, rounds, diff=(0, 0x0040)):
    print("Make_data Input diff: ", diff)
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_random_samples = np.sum(Y == 0)
    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys = expand_keys(key, rounds)
    plain_01 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_02 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_11 = plain_01 ^ diff[0]
    plain_12 = plain_02 ^ diff[1]
    plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    cipher_01, cipher_02 = encryption((plain_01, plain_02), keys)
    cipher_11, cipher_12 = encryption((plain_11, plain_12), keys)
    X = convert_to_bits([cipher_01, cipher_02, cipher_11, cipher_12])
    return X, Y

def make_data_value_diff(n, rounds, diff=(0, 0x0040)):
    print("Make_data Input diff: ", diff)
    Y = np.frombuffer(urandom(n), dtype=np.uint8)
    Y = Y & 1
    num_random_samples = np.sum(Y == 0)
    key = np.frombuffer(urandom(8 * n), dtype=np.uint16).reshape(4, -1)
    keys = expand_keys(key, rounds)
    plain_01 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_02 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_11 = plain_01 ^ diff[0]
    plain_12 = plain_02 ^ diff[1]
    plain_11[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    plain_12[Y == 0] = np.frombuffer(urandom(2 * num_random_samples), dtype=np.uint16)
    cipher_01, cipher_02 = encryption((plain_01, plain_02), keys)
    cipher_11, cipher_12 = encryption((plain_11, plain_12), keys)
    delta_y = cipher_02 ^ cipher_12
    X = value_diff_convert_to_bits([cipher_01, cipher_11, delta_y])
    return X, Y
