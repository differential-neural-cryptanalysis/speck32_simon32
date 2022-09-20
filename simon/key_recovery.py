import simon
from os import urandom
import numpy as np
from math import sqrt, log2
import time
import sys, getopt
from datetime import datetime

import multiprocessing as mp
PN_log2 = 0
PN = 1 << PN_log2

np.set_printoptions(linewidth=np.inf)
Z_min = 1e-8

ND_R = 10
PR_R = 1 + 4
AP_R = 1
TOTAL_R = PR_R + ND_R + AP_R
KG = 6
NB_N = 4 + 3
NET_HELP_USE_VALUE_DIFF = 1
NET_USE_VALUE_DIFF = 0
DIRECT_IMPROVE = 0
CAND_KN = 32
UNGUESS = 1
in_diff = (0x1000, 0x4440)
neutral_bit_sets = [[2], [6], [12, 26], [10, 14, 28]]


WORD_SIZE = simon.WORD_SIZE()
mode_mask = simon.MOD_MASK

class P:
    c_s_attempt = True
    keys = None
    local_best_key1_value = -2000
    true_pos = None

def GET1(a, i):
    return ((a & (1 << i)) >> i)

def replace_1bit(a, b, i):
    mask = 0xffff ^ (1 << i)
    a = a & mask
    a = a | (b << i)
    return a

def cost_time(sec):
    hours = int(sec / (60*60))
    min = int(sec/60 -60*hours)
    seconds = int(sec-60*min-3600*hours)
    return "{:0>2d}:{:0>2d}:{:0>2d}".format(hours, min, seconds)

def get_key(rounds):
    key = np.frombuffer(urandom(8), dtype=np.uint16).reshape(4, -1)
    keys = simon.expand_keys(key=key, rounds=rounds)
    return keys


def get_plain_pairs(n):
    global logfile
    plain_1 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    plain_2 = np.frombuffer(urandom(2 * n), dtype=np.uint16)
    return plain_1, plain_2

def make_plain_structure_conditional(
    plain_1,
    plain_2,
    input_diff,
    neutral_bits_CS
    ):
    global logfile

    p0_tmp_A = [np.array(p1i) for p1i in plain_1]
    p1_tmp_A = [np.array(p2i) for p2i in plain_2]
    for pi in range(len(plain_1)):
        for ni in neutral_bits_CS[pi]:
            p0_tmp = np.copy(p0_tmp_A[pi])
            p1_tmp = np.copy(p1_tmp_A[pi])
            for bj in range(len(ni)):
                d = 1 << ni[bj]
                d0 = d >> 16
                d1 = d & 0xffff
                p0_tmp = p0_tmp ^ d0
                p1_tmp = p1_tmp ^ d1
            p0_tmp_A[pi] = np.hstack((p0_tmp_A[pi], p0_tmp))
            p1_tmp_A[pi] = np.hstack((p1_tmp_A[pi], p1_tmp))
    p0 = np.array(p0_tmp_A)
    p1 = np.array(p1_tmp_A)

    p0b = p0 ^ input_diff[0]
    p1b = p1 ^ input_diff[1]
    return p0, p1, p0b, p1b

def make_plain_structure(
    plain_1,
    plain_2,
    input_diff,
    neutral_bits
    ):
    p0 = np.copy(plain_1)
    p1 = np.copy(plain_2)
    p0 = p0.reshape(-1, 1)
    p1 = p1.reshape(-1, 1)
    for i in neutral_bits:
        if len(i) == 1:
            d0 = 1 << i[0]
            d00 = d0 >> 16
            d01 = d0 & 0xffff
            p0 = np.concatenate([p0, p0 ^ d00], axis = 1)
            p1 = np.concatenate([p1, p1 ^ d01], axis = 1)
        elif len(i) == 2:
            d0 = 1 << i[0]
            d00 = d0 >> 16
            d01 = d0 & 0xffff
            d1 = 1 << i[1]
            d10 = d1 >> 16
            d11 = d1 & 0xffff
            p0 = np.concatenate([p0, p0 ^ d00 ^ d10], axis = 1)
            p1 = np.concatenate([p1, p1 ^ d01 ^ d11], axis = 1)
        elif len(i) == 3:
            d0 = 1 << i[0]
            d00 = d0 >> 16
            d01 = d0 & 0xffff
            d1 = 1 << i[1]
            d10 = d1 >> 16
            d11 = d1 & 0xffff
            d2 = 1 << i[2]
            d20 = d2 >> 16
            d21 = d2 & 0xffff
            p0 = np.concatenate([p0, p0 ^ d00 ^ d10 ^ d20], axis = 1)
            p1 = np.concatenate([p1, p1 ^ d01 ^ d11 ^ d21], axis = 1)
        elif len(i) == 4:
            d0 = 1 << i[0]
            d00 = d0 >> 16
            d01 = d0 & 0xffff
            d1 = 1 << i[1]
            d10 = d1 >> 16
            d11 = d1 & 0xffff
            d2 = 1 << i[2]
            d20 = d2 >> 16
            d21 = d2 & 0xffff
            d3 = 1 << i[3]
            d30 = d3 >> 16
            d31 = d3 & 0xffff
            p0 = np.concatenate([p0, p0 ^ d00 ^ d10 ^ d20 ^ d30], axis = 1)
            p1 = np.concatenate([p1, p1 ^ d01 ^ d11 ^ d21 ^ d31], axis = 1)
    p0b = p0 ^ input_diff[0]
    p1b = p1 ^ input_diff[1]
    return p0, p1, p0b, p1b

## Separate neutral bits
def get_cipher_structure(nums, rounds, diff, neutral_bits, keys):
    global logfile
    input_diff=diff
    neutral_bits = neutral_bits

    plain_1, plain_2 = get_plain_pairs(nums)

    if PR_R == (1 + 4):
        # Choose the correct key k0_3 and k0_5, 
        # To require less structures to get correct pos.
        if KG > 0:
            set_bit = GET1(keys[0][0], 5)
            plain_1 = replace_1bit(plain_1, set_bit, 5)
        if KG > 1:
            set_bit = GET1(keys[0][0], 3)
            plain_1 = replace_1bit(plain_1, set_bit, 3)

        k13 = GET1(keys[0][0], 13)
        k15 = GET1(keys[0][0], 15)
        k01 = GET1(keys[0][0],  1)
        k11 = GET1(keys[0][0], 11)
        k07 = GET1(keys[0][0],  7)
        k09 = GET1(keys[0][0],  9)
        neutral_bits_CS = [[[2], [6], [12, 26], [10, 14, 28]] for pi in range(nums)]
        for pi in range(nums):
            c13 = GET1(plain_1[pi], 13) ^ k13
            c15 = GET1(plain_1[pi], 15) ^ k15
            c01 = GET1(plain_1[pi],  1) ^ k01
            c11 = GET1(plain_1[pi], 11) ^ k11
            c09 = GET1(plain_1[pi],  9) ^ k09
            c07 = GET1(plain_1[pi],  7) ^ k07

            if KG > 3:
                NB_13_15 = [8, 6+16]
                if (c13, c15) == (0, 0):
                    NB_13_15 = [8, 6+16]
                elif (c13, c15) == (0, 1):
                    NB_13_15 = [8, 6+16, 7]
                elif (c13, c15) == (1, 0):
                    NB_13_15 = [8, 6+16, 14]
                elif (c13, c15) == (1, 1):
                    NB_13_15 = [8, 6+16, 7, 14]
                neutral_bits_CS[pi].append(NB_13_15)
            if KG > 4:
                NB_13_11 = [4+16]
                if (c13, c11) == (0, 0):
                    NB_13_11 = [4+16]
                elif (c13, c11) == (0, 1):
                    NB_13_11 = [4+16, 12]
                elif (c13, c11) == (1, 0):
                    NB_13_11 = [4+16, 5]
                elif (c13, c11) == (1, 1):
                    NB_13_11 = [4+16, 5, 12]
                neutral_bits_CS[pi].append(NB_13_11)
            if KG > 5:
                NB_09_11 = [2+16, 4]
                if (c09, c11) == (0, 0):
                    NB_09_11 = [2+16, 4]
                elif (c09, c11) == (0, 1):
                    NB_09_11 = [2+16, 4, 3]
                elif (c09, c11) == (1, 0):
                    NB_09_11 = [2+16, 4, 10]
                elif (c09, c11) == (1, 1):
                    NB_09_11 = [2+16, 4, 3, 10]
                neutral_bits_CS[pi].append(NB_09_11)
            if KG > 6:
                NB_01_15 = [10, 8+16]
                if (c01, c15) == (0, 0):
                    NB_01_15 = [10, 8+16]
                elif (c01, c15) == (0, 1):
                    NB_01_15 = [10, 8+16, 0]
                elif (c01, c15) == (1, 0):
                    NB_01_15 = [10, 8+16, 9]
                elif (c01, c15) == (1, 1):
                    NB_01_15 = [10, 8+16, 0, 9]
                neutral_bits_CS[pi].append(NB_01_15)
            if KG > 7:
                NB_07_09 = [0+16]
                if (c07, c09) == (0, 0):
                    NB_07_09 = [0+16, 8]
                elif (c07, c09) == (0, 1):
                    NB_07_09 = [0+16, 8, 1]
                elif (c07, c09) == (1, 0):
                    NB_07_09 = [0+16]
                elif (c07, c09) == (1, 1):
                    NB_07_09 = [0+16, 1]
                neutral_bits_CS[pi].append(NB_07_09)
        plain_01, plain_02, plain_11, plain_12 = \
            make_plain_structure_conditional(plain_1, plain_2, input_diff, neutral_bits_CS)

    if PR_R == (1 + 3):
        # Choose the correct key k0_1 and k0_5, 
        # To require less structures to get correct pos.
        if KG > 0:
            set_bit = GET1(keys[0][0], 1)
            plain_1 = replace_1bit(plain_1, set_bit, 1)
        if KG > 1:
            set_bit = GET1(keys[0][0], 3)
            plain_1 = replace_1bit(plain_1, set_bit, 3)
        plain_01, plain_02, plain_11, plain_12 = \
            make_plain_structure(plain_1, plain_2, input_diff, neutral_bits)

    #
    plain_01, plain_02 = simon.one_round_decrypt(cipher=(plain_01, plain_02), key=0)
    plain_11, plain_12 = simon.one_round_decrypt(cipher=(plain_11, plain_12), key=0)

    cipher_01, cipher_02 = simon.encryption(plain_pair=(plain_01, plain_02), keys=keys)
    cipher_11, cipher_12 = simon.encryption(plain_pair=(plain_11, plain_12), keys=keys)

    # find correct index
    p_01, p_02 = simon.decryption(cipher_pair=(cipher_01, cipher_02), keys=keys[-(ND_R + AP_R):])
    p_11, p_12 = simon.decryption(cipher_pair=(cipher_11, cipher_12), keys=keys[-(ND_R + AP_R):])

    diff0 = p_01 ^ p_11
    diff1 = p_02 ^ p_12

    temp = []
    for i in range(len(diff0)):
        if (diff0[i][0] == 0x0) and (diff1[i][0] == 0x0040):
            temp.append(i)
    print("true pod:", temp, file=logfile, flush=True)
    P.true_pos = temp

    for i in temp:
        diff0_pos = np.where(diff0[i] == 0x0)[0]
        diff1_pos = np.where(diff1[i] == 0x0040)[0]

        right_pairs = set(diff0_pos) & set(diff1_pos)
        if len(list(right_pairs)) > 0:
            print("right message in pod {}: {}".format(i,len(right_pairs) / len(diff0[i])), file=logfile, flush=True)

    return [cipher_01, cipher_02, cipher_11, cipher_12]

#
def hamming_weight(v):
    global logfile
    if isinstance(v, str):
        v = int(v[2:], 16)
        res = 0
        for i in range(16):
            res = res + ((v >> i) & 1)
    else:
        res = np.zeros(v.shape, dtype=np.uint8)
        for i in range(16):
            res = res + ((v >> i) & 1)
    return res

#here, we use some symmetries of the wrong key performance profile
#by performing the optimization step only on the 14 lowest bits and randomizing the others
#on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
#In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here
tmp_br = np.arange(2**(16-UNGUESS), dtype=np.uint16);
tmp_br = np.repeat(tmp_br, CAND_KN).reshape(-1,CAND_KN);
#
def weighted_euclidean_distance(candidate_keys, means, mu, sigma):
    global logfile
    global tmp_br
    n = len(candidate_keys)
    if (tmp_br.shape[1] != n):
        tmp_br = np.arange(2**(16-UNGUESS), dtype=np.uint16);
        tmp_br = np.repeat(tmp_br, n).reshape(-1,n);
    tmp = (tmp_br ^ candidate_keys) & mode_mask
    v = (means - mu[tmp]) * sigma[tmp]
    v = v.reshape(-1, n)
    score = np.linalg.norm(v, axis=1)
    return score


# efficiently find a list of possible key candidates given a ciphertext structure
def bayesian_key_search1(cipher_structure, net, iter_nums, candidate_num, mean, std):
    global logfile
    n = len(cipher_structure[0])
    all_keys_candidates_list = np.zeros(candidate_num * iter_nums, dtype=np.uint16)
    all_value_list = np.zeros(candidate_num * iter_nums)
    #
    keys = np.random.choice(a=2 ** WORD_SIZE, size=candidate_num, replace=False)
    if NET_USE_VALUE_DIFF == 1:
        key_zero = keys ^ keys
        key_zeros = np.repeat(key_zero, n)
    cs_0, cs_1, cs_2, cs_3 = np.tile(cipher_structure[0], candidate_num), \
                             np.tile(cipher_structure[1], candidate_num), \
                             np.tile(cipher_structure[2], candidate_num), \
                             np.tile(cipher_structure[3], candidate_num)
    for i in range(iter_nums):
        ks = np.repeat(keys, n)
        c_01, c_02 = simon.one_round_decrypt(cipher=(cs_0, cs_1), key=ks)
        c_11, c_12 = simon.one_round_decrypt(cipher=(cs_2, cs_3), key=ks)
        if NET_USE_VALUE_DIFF == 1:
            c_01, c_02 = simon.one_round_decrypt(cipher=(c_01, c_02), key=key_zeros)
            c_11, c_12 = simon.one_round_decrypt(cipher=(c_11, c_12), key=key_zeros)
            delta_y = c_02 ^ c_12
            cipher = simon.value_diff_convert_to_bits([c_01, c_11, delta_y])
        else:
            cipher = simon.convert_to_bits([c_01, c_02, c_11, c_12])
        Z = net.predict(cipher, batch_size=10000)
        if len(Z) == 2:
            Z = Z[-1]
        Z = Z.reshape(candidate_num, -1)
        k_means = np.mean(Z, axis=1)
        Z = Z / (1.0 - Z)
        Z = np.log2(Z + Z_min)
        value = np.sum(Z, axis=1)
        all_value_list[i * candidate_num: (i + 1) * candidate_num] = value
        all_keys_candidates_list[i * candidate_num: (i + 1) * candidate_num] = np.copy(keys)
        #
        scores = weighted_euclidean_distance(candidate_keys=keys, means=k_means, mu=mean, sigma=std)
        tmp = np.argpartition(scores, candidate_num)
        keys = tmp[0: candidate_num]
        if UNGUESS > 0:
            r = np.random.randint(0,1<<UNGUESS,candidate_num,dtype=np.uint16); r = r << (16-UNGUESS); keys = keys ^ r;
    return all_keys_candidates_list, all_value_list

# efficiently find a list of possible key candidates given a ciphertext structure
def bayesian_key_search2(cipher_structure, net, iter_nums, candidate_num, mean, std):
    global logfile
    n = len(cipher_structure[0])
    all_keys_candidates_list = np.zeros(candidate_num * iter_nums, dtype=np.uint16)
    all_value_list = np.zeros(candidate_num * iter_nums)
    #
    keys = np.random.choice(a=2 ** WORD_SIZE, size=candidate_num, replace=False)
    if NET_HELP_USE_VALUE_DIFF == 1:
        key_zero = keys ^ keys
        key_zeros = np.repeat(key_zero, n)
    cs_0, cs_1, cs_2, cs_3 = np.tile(cipher_structure[0], candidate_num), \
                             np.tile(cipher_structure[1], candidate_num), \
                             np.tile(cipher_structure[2], candidate_num), \
                             np.tile(cipher_structure[3], candidate_num)
    for i in range(iter_nums):
        ks = np.repeat(keys, n)
        c_01, c_02 = simon.one_round_decrypt(cipher=(cs_0, cs_1), key=ks)
        c_11, c_12 = simon.one_round_decrypt(cipher=(cs_2, cs_3), key=ks)

        if NET_HELP_USE_VALUE_DIFF == 1:
            c_01, c_02 = simon.one_round_decrypt(cipher=(c_01, c_02), key=key_zeros)
            c_11, c_12 = simon.one_round_decrypt(cipher=(c_11, c_12), key=key_zeros)
            delta_y = c_02 ^ c_12
            cipher = simon.value_diff_convert_to_bits([c_01, c_11, delta_y])
        else:
            cipher = simon.convert_to_bits([c_01, c_02, c_11, c_12])
        Z = net.predict(cipher, batch_size=10000)
        if len(Z) == 2:
            Z = Z[-1]
        Z = Z.reshape(candidate_num, -1)
        k_means = np.mean(Z, axis=1)
        Z = Z / (1.0 - Z)
        Z = np.log2(Z + Z_min)
        value = np.sum(Z, axis=1)
        all_value_list[i * candidate_num: (i + 1) * candidate_num] = value
        all_keys_candidates_list[i * candidate_num: (i + 1) * candidate_num] = np.copy(keys)
        #
        scores = weighted_euclidean_distance(candidate_keys=keys, means=k_means, mu=mean, sigma=std)
        tmp = np.argpartition(scores, candidate_num)
        keys = tmp[0: candidate_num]
        if UNGUESS > 0:
            r = np.random.randint(0,1<<UNGUESS,candidate_num,dtype=np.uint16); r = r << (16-UNGUESS); keys = keys ^ r;
    return all_keys_candidates_list, all_value_list

low_weight = np.array(range(2 ** WORD_SIZE), dtype=np.uint16)
low_weight = low_weight[hamming_weight(low_weight) <= 2]


#  Before we return a key, we perform a small verification search weith hamming radius two round
#  the two subkey candidates that are currently best. This removes remaining bit errors in the key
#  guess. If the verification search yields an improvement, it is repeated with the new best key guess
def verifier_search(cipher_structure, best_guess, use_n, net):
    global logfile
    ck1 = best_guess[0] ^ low_weight
    ck2 = best_guess[1] ^ low_weight
    n = len(ck1)
    ck1 = np.repeat(ck1, n)
    keys_1 = np.copy(ck1)
    ck2 = np.tile(ck2, n)
    keys_2 = np.copy(ck2)
    #
    ck1 = np.repeat(ck1, use_n)
    ck2 = np.repeat(ck2, use_n)

    if NET_HELP_USE_VALUE_DIFF == 1:
        key_zeros = ck2 ^ ck2
    cs_01, cs_02, cs_11, cs_12 = np.tile(cipher_structure[0][0:use_n], n * n), \
                                 np.tile(cipher_structure[1][0:use_n], n * n), \
                                 np.tile(cipher_structure[2][0:use_n], n * n), \
                                 np.tile(cipher_structure[3][0:use_n], n * n)
    ps_01, ps_02 = simon.one_round_decrypt(cipher=(cs_01, cs_02), key=ck1)
    ps_11, ps_12 = simon.one_round_decrypt(cipher=(cs_11, cs_12), key=ck1)
    ps_01, ps_02 = simon.one_round_decrypt(cipher=(ps_01, ps_02), key=ck2)
    ps_11, ps_12 = simon.one_round_decrypt(cipher=(ps_11, ps_12), key=ck2)

    if NET_HELP_USE_VALUE_DIFF == 1:
        ps_01, ps_02 = simon.one_round_decrypt(cipher=(ps_01, ps_02), key=key_zeros)
        ps_11, ps_12 = simon.one_round_decrypt(cipher=(ps_11, ps_12), key=key_zeros)
        delta_y = ps_02 ^ ps_12
        X = simon.value_diff_convert_to_bits([ps_01, ps_11, delta_y])
    else:
        X = simon.convert_to_bits([ps_01, ps_02, ps_11, ps_12])
    Z = net.predict(X, batch_size=10000)
    if len(Z) == 2:
            Z = Z[-1]
    Z = np.log2(Z / (1.0 - Z) + Z_min)
    Z = Z.reshape(-1, use_n)
    value = np.mean(Z, axis=1) * len(cipher_structure[0])
    index = np.argmax(value)
    v = value[index]
    k_1 = keys_1[index]
    k_2 = keys_2[index]
    return k_1, k_2, v


def bayesian_optimization(cipher_structures, iteration_num, cutoff_1, cutoff_2, net, net_help,
                          mean, std, mean_help, std_help, verify_breadth=None):
    global logfile
    if verify_breadth is None:
        verify_breadth = len(cipher_structures[0][0])
    n = len(cipher_structures[0])
    #
    alpha = sqrt(n)
    best_val = [best_val_global, best_val_global]
    best_key = (0, 0)
    best_pod = 0
    #
    moment_best = np.full(n, best_val_global)
    moment_best_k1 = np.full(n, 0)

    eps = 1e-5
    num_visits = np.full(n, eps)
    for j in range(iteration_num):
        if j % 100 == 0:
            print(j, "th iteration: ", file=logfile, flush=True)
        priority = moment_best + alpha * np.sqrt(log2(j + 1) / num_visits)
        index = np.argmax(priority)
        num_visits[index] = num_visits[index] + 1
        # before return a key
        if (best_val[1] > cutoff_2):
            print("best val", best_val, "cutoff_2", cutoff_2, file=logfile, flush=True)
            improvement = (verify_breadth > 0)
            while improvement:
                tmp_cipher_structure = [cipher_structures[0][best_pod], cipher_structures[1][best_pod],
                                        cipher_structures[2][best_pod], cipher_structures[3][best_pod]]
                key_1, key_2, value = verifier_search(cipher_structure=tmp_cipher_structure,
                                                      best_guess=best_key, net=net_help, use_n=verify_breadth)
                improvement = (value > best_val[1])
                if (improvement):
                    best_key = (key_1, key_2); best_val[0] = moment_best[best_pod]; best_val[1] = value;
                    print("Improvement mid: ", iteration_num, " iterations:", file=logfile)
                    print("best_pod", best_pod, ", ", best_pod in P.true_pos, file=logfile)
                    print("best_val", best_val, file=logfile)
                    print("best key", hex(best_key[0]), hex(best_key[1]), file=logfile, flush=True)
            return best_key, j
        #
        keys_candidates_list_1, value_list_1 = bayesian_key_search1(
            cipher_structure=[cipher_structures[0][index], cipher_structures[1][index], cipher_structures[2][index],
                              cipher_structures[3][index]], net=net, iter_nums=5, candidate_num=CAND_KN, mean=mean, std=std)
        v_tmp_1 = np.max(value_list_1)
        print('index: ', '{:<5}'.format(index), " ", index in P.true_pos, ', num_visits[index]:', '{:<10}'.format(num_visits[index]), "v_tmp_1:", '{:<20}'.format(v_tmp_1), file=logfile, flush=True)
        if v_tmp_1 > P.local_best_key1_value:
            P.local_best_key1_value = v_tmp_1

        if v_tmp_1 > moment_best[index]:
            moment_best[index] = v_tmp_1
            moment_best_k1[index] = keys_candidates_list_1[np.argmax(value_list_1)]

        if (v_tmp_1 > cutoff_1):
            P.c_s_attempt = False
            diff_final = hamming_weight(keys_candidates_list_1 ^ P.keys[-1][0])
            print("min diff position", np.argmin(diff_final), ":", diff_final[np.argmin(diff_final)], file=logfile)
            print("max value position", np.argmax(value_list_1), "diff:", diff_final[np.argmax(value_list_1)], file=logfile)

            second_index = [i for i in range(len(keys_candidates_list_1)) if value_list_1[i] > cutoff_1]
            print("v_tmp_1:",  v_tmp_1, "cutoff_1:", cutoff_1, file=logfile)
            for s_index in second_index:
                c0a, c1a = simon.one_round_decrypt(cipher=(cipher_structures[0][index], cipher_structures[1][index]),
                                                   key=keys_candidates_list_1[s_index])
                c0b, c1b = simon.one_round_decrypt(cipher=(cipher_structures[2][index], cipher_structures[3][index]),
                                                   key=keys_candidates_list_1[s_index])
                keys_candidates_list_2, value_list_2 = bayesian_key_search2(cipher_structure=[c0a, c1a, c0b, c1b],
                                                                           net=net_help, iter_nums=5, candidate_num=CAND_KN,
                                                                           mean=mean_help, std=std_help)
                v_tmp_2 = np.max(value_list_2)
                k_tmp_2 = keys_candidates_list_2[np.argmax(value_list_2)]
                print("v1: ", '{:<20}'.format(value_list_1[s_index]), ", k1: ", "{0:04x}".format(keys_candidates_list_1[s_index]), ", max in value_list_2: ", '{:<20}'.format(v_tmp_2), ", k2:", "{0:04x}".format(k_tmp_2), file=logfile, flush=True)

                v_tmp_2_best = v_tmp_2
                k_tmp_1_best = keys_candidates_list_1[s_index]
                k_tmp_2_best = k_tmp_2

                if DIRECT_IMPROVE == 1:
                    improvement = (verify_breadth > 0)
                    while improvement:
                        tmp_cipher_structure_k2 = [cipher_structures[0][index], cipher_structures[1][index],
                                                cipher_structures[2][index], cipher_structures[3][index]]
                        key_1, key_2, value = verifier_search(cipher_structure=tmp_cipher_structure_k2,
                                                              best_guess=(k_tmp_1_best, k_tmp_2_best), net=net_help, use_n=verify_breadth)
                        improvement = (value > v_tmp_2_best)
                        if (improvement):
                            k_tmp_1_best = key_1
                            k_tmp_2_best = key_2
                            v_tmp_2_best = value
                            print(
                                "Improvement key2: v_tmp_2_best ", v_tmp_2_best,
                                "best key", hex(k_tmp_1_best), hex(k_tmp_2_best), file=logfile, flush=True)
                
                if v_tmp_2_best > best_val[1]:
                    best_val[1] = v_tmp_2_best; best_val[0] = v_tmp_1;
                    best_key = (k_tmp_1_best, k_tmp_2_best)
                    best_pod = index
                    print("v_tmp_2 > best_val:", v_tmp_2_best, best_val[1], file=logfile)
                    print("best_pod", best_pod, ", ", best_pod in P.true_pos, file=logfile)
                    print("best_val", best_val, file=logfile)
                    print("best_key", hex(best_key[0]), hex(best_key[1]), "true key:", hex(P.keys[-1][0]), hex(P.keys[-2][0]), file=logfile, flush=True)
                    print("HW difference:", hamming_weight(best_key[0] ^ P.keys[-1][0]), hamming_weight(best_key[1] ^ P.keys[-2][0]), file=logfile,
                          flush=True)
                if DIRECT_IMPROVE == 1:
                    if (best_val[1] > cutoff_2):
                        return best_key, j
    print("After ", iteration_num, " iterations:", file=logfile)
    print("best_pod", best_pod, ", ", best_pod in P.true_pos, file=logfile)
    print("best_val", best_val, file=logfile)
    print("best key", hex(best_key[0]), hex(best_key[1]), file=logfile, flush=True)
    improvement = (verify_breadth > 0)
    while improvement:
        tmp_cipher_structure = [cipher_structures[0][best_pod], cipher_structures[1][best_pod],
                                cipher_structures[2][best_pod], cipher_structures[3][best_pod]]
        key_1, key_2, value = verifier_search(cipher_structure=tmp_cipher_structure,
                                              best_guess=best_key, net=net_help, use_n=verify_breadth)
        improvement = (value > best_val[1])
        if (improvement):
            best_key = (key_1, key_2); best_val[0] = moment_best[best_pod]; best_val[1] = value;
            print("Improvement last: ", iteration_num, " iterations:", file=logfile)
            print("best_pod", best_pod, ", ", best_pod in P.true_pos, file=logfile)
            print("best key", hex(best_key[0]), hex(best_key[1]), file=logfile, flush=True)
    return best_key, iteration_num


def test_bayes(test_nums, c_s_num, iteration_num, cutoff_1, cutoff_2, net, net_help, mean, std, mean_help,
               std_help, rounds, verify_breadth=None):
    global logfile
    arr_1 = np.zeros(test_nums, dtype=np.uint16)
    arr_2 = np.zeros(test_nums, dtype=np.uint16)
    correct_num = 0
    for i in range(test_nums):
        print("Test: ", i, file=logfile, flush=True)

        keys = get_key(rounds=rounds)
        P.keys = keys
        P.c_s_attempt = True
        attempt = 0
        np.set_printoptions(formatter=dict(int=lambda x: "{0:{fill}4x}".format(x, fill='0')))
        print("truth key", keys[rounds - 1], keys[rounds - 2], file=logfile, flush=True)
        np.set_printoptions()
        while P.c_s_attempt and attempt < 1:
            attempt += 1
            print("Attempt:", attempt, file=logfile)
            import time
            start_time = time.time()
            cipher_structures = get_cipher_structure(nums=c_s_num, rounds=rounds, diff=in_diff, neutral_bits=neutral_bit_sets, keys=keys)
            guess_key, num_used_iteration = bayesian_optimization(cipher_structures=cipher_structures,
                                                                iteration_num=iteration_num, cutoff_1=cutoff_1,
                                                                cutoff_2=cutoff_2, net=net, net_help=net_help,
                                                                mean=mean, std=std, mean_help=mean_help,
                                                                std_help=std_help, verify_breadth=verify_breadth
                                                                )
            end_time = time.time()
            arr_1[i] = guess_key[0] ^ keys[rounds - 1]
            arr_2[i] = guess_key[1] ^ keys[rounds - 2]
            if (arr_1[i] == 0):
                correct_num = correct_num + 1
            print('during time is :', cost_time(end_time - start_time), file=logfile)
            print("used iteration number is ", num_used_iteration, file=logfile)
            np.set_printoptions(formatter=dict(int=lambda x: "{0:{fill}4x}".format(x, fill='0')))
            print("guess key: ", hex(guess_key[0]), hex(guess_key[1]), file=logfile)
            print("truth key: ", keys[rounds - 1], keys[rounds - 2], file=logfile)
            wbp1 = []
            wbp2 = []
            for bi in range(WORD_SIZE):
                if ((arr_1[i] >> bi) & 1) == 1:
                    wbp1.append(bi)
                if ((arr_2[i] >> bi) & 1) == 1:
                    wbp2.append(bi)
            print("Difference between real key and guess key:", arr_1[i], arr_2[i], file=logfile)
            print("Wrong key bit index (last subkey1, last subkey2)", wbp1, wbp2, file=logfile)
            np.set_printoptions()
            print("Hamming weight of difference between real key and guess key: ", hamming_weight(arr_1[i])+hamming_weight(arr_2[i]), file=logfile, flush=True)
            sys.stdout.flush()
    print("correct rate", correct_num / (1.0 * test_nums), file=logfile)
    print("Done", file=logfile)
    return arr_1, arr_2

def test(idx):
    global best_val_global
    global ND_R
    global PR_R
    global AP_R
    global TOTAL_R
    global KG
    global NB_N
    global NET_HELP_USE_VALUE_DIFF
    global NET_USE_VALUE_DIFF
    global DIRECT_IMPROVE
    global CAND_KN
    global UNGUESS
    global in_diff
    global neutral_bit_sets

    ND_R = 11
    PR_R = 1 + 3
    AP_R = 1
    TOTAL_R = PR_R + ND_R + AP_R
    KG = 2
    NB_N = 11
    NET_HELP_USE_VALUE_DIFF = 1
    NET_USE_VALUE_DIFF = 0
    DIRECT_IMPROVE = 0
    CAND_KN = 32
    UNGUESS = 1

    num_rounds = TOTAL_R

    if PR_R == (1 + 4):
        in_diff = (0x1000, 0x4440)
        neutral_bit_sets = [[2], [6], [12, 26], [10, 14, 28]]
        c1 = 20
        c2 = 70
        best_val_global = -2000
        cipher_s_num = 1024
        iter_num = cipher_s_num * 2
    if PR_R == (1 + 3):
        in_diff = (0x0440, 0x1000)
        neutral_bit_sets = [[2],[3],[4],[6],[8],[9],[10],[18],[22],[0, 24],[12, 26]]
        c1 = 25
        c2 = 100
        best_val_global = -2000
        cipher_s_num = 1<<7
        iter_num = cipher_s_num * 4

    now = datetime.now()
    current_time = now.strftime("%Y:%m:%d:%H:%M:%S:")
    current_time = current_time.replace(":", "_")

    global logfile
    logfile = open(current_time + '_KG' + str(KG) + "_NB" + str(len(neutral_bit_sets)) + "_CANDKN_" + str(CAND_KN) + "_NS_" + str(cipher_s_num) + "_IT_" + str(iter_num) + "_c1_" + str(c1) + "_c2_" + str(c2) + "_r" + str(num_rounds) + '_proc' + str(idx) + '.txt', 'w+')
    from keras.models import load_model
    from tensorflow.python.keras import backend as K
    import tensorflow as tf
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    K.set_session(tf.compat.v1.Session(config=config))
    from tensorflow.python.client import device_lib 
    divices = device_lib.list_local_devices()
    print(divices, file=logfile, flush=True)
    
    with tf.device("/gpu:"+str(idx)):
        print("[INFO] num_rounds:", num_rounds, file=logfile)
        print("[INFO] cutoff_1:", c1, " cutoff_2:", c2, "cipher_structure_num:", cipher_s_num, " best_val:", best_val_global,
              " iter_num: ", iter_num, file=logfile,)
        t1 = time.time()
        print("[INFO] Loading Model", file=logfile)

        if ND_R == 11:
            NET_HELP_USE_VALUE_DIFF = 1
            NET_USE_VALUE_DIFF = 0
            model_main = load_model("./ND_VV_Simon32_11R.h5")
            model_help = load_model("./ND_VD_Simon32_9R.h5")
            print("[INFO] Loading Mean and Std", file=logfile)
            mean_main = np.load("./ND_VV_Simon32_11R_mean.npy")
            std_main = np.load("./ND_VV_Simon32_11R_std.npy")
            std_main = 1.0 / std_main
            mean_help = np.load("./ND_VD_Simon32_9R_mean.npy")
            std_help = np.load("./ND_VD_Simon32_9R_std.npy")
            std_help = 1.0 / std_help


        if ND_R == 10:
            NET_USE_VALUE_DIFF = 1
            NET_HELP_USE_VALUE_DIFF = 1
            if NET_USE_VALUE_DIFF == 1:
                model_main = load_model("./ND_VD_Simon32_9R.h5")
                mean_main = np.load("./ND_VD_Simon32_9R_mean.npy")
                std_main = np.load("./ND_VD_Simon32_9R_std.npy")
                std_main = 1.0 / std_main
            else:
                model_main = load_model("./ND_VV_Simon32_10R.h5")
                mean_main = np.load("./ND_VV_Simon32_10R_mean.npy")
                std_main = np.load("./ND_VV_Simon32_10R_std.npy")
                std_main = 1.0 / std_main
            if NET_HELP_USE_VALUE_DIFF == 1:
                model_help = load_model("./ND_VD_Simon32_8R.h5")
                mean_help = np.load("./ND_VD_Simon32_8R_mean.npy")
                std_help = np.load("./ND_VD_Simon32_8R_std.npy")
                std_help = 1.0 / std_help
            else:
                model_help = load_model("./ND_VV_Simon32_9R.h5")
                mean_help = np.load("./ND_VV_Simon32_9R_mean.npy")
                std_help = np.load("./ND_VV_Simon32_9R_std.npy")
                std_help = 1.0 / std_help

        print("[INFO] Loading costs ", time.time() - t1, 's', file=logfile)
        print("\n", file=logfile, flush=True)

        print("---------Begin-------", file=logfile)
        arr_1, arr_2 = test_bayes(test_nums=20, c_s_num=cipher_s_num, iteration_num=iter_num, cutoff_1=c1, cutoff_2=c2,
                                       net=model_main,
                                       net_help=model_help,
                                       mean=mean_main, std=std_main,
                                       mean_help=mean_help, std_help=std_help,
                                       rounds=num_rounds, verify_breadth=(1<<NB_N))

        str1 = "./%s_rounds_key_recovery_1_%s_%s_%s_proc%s.npy" % (num_rounds, c1, c2, str(best_val_global), str(idx))
        str2 = "./%s_rounds_key_recovery_2_%s_%s_%s_proc%s.npy" % (num_rounds, c1, c2, str(best_val_global), str(idx))

        np.save(str1, arr_1)
        np.save(str2, arr_2)
    logfile.close()
    return str(idx) + " Done"

if __name__ == '__main__':
    pool = mp.Pool(PN)
    idx_range = range(0, PN)
    results = pool.map_async(test, idx_range).get()
    pool.close()
