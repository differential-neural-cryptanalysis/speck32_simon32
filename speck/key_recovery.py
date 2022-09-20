import speck as sp
import numpy as np

from scipy.stats import norm
from os import urandom
from math import sqrt, log, log2
from time import time
from math import log2
import sys
import sys, getopt
from datetime import datetime
import multiprocessing as mp

np.set_printoptions(threshold=np.inf)
np.set_printoptions(precision=6)
Z_min = 1e-16

PN_log2 = 3
PN = 1 << PN_log2
BS = 1<<14
test_n = 2
test_n_all = test_n * PN

LOW_WEIGHT = 2

MAIN_NOT_GUESSED_KEY1 = 2
MAIN_GUESSED_KEY1 = sp.WORD_SIZE() - MAIN_NOT_GUESSED_KEY1

HELPER_NOT_GUESS_KEY2 = 2
HELPER_GUESSED_KEY2 = sp.WORD_SIZE() - HELPER_NOT_GUESS_KEY2

KEY1_MAIN_LEFT_MASK = ((1<<MAIN_NOT_GUESSED_KEY1) - 1)
KEY1_MAIN_LEFT_OFFSET = MAIN_GUESSED_KEY1

KEY2_HELPER_GUESSED_MASK = ((1<<HELPER_GUESSED_KEY2) - 1) << MAIN_NOT_GUESSED_KEY1

HELPER_GUESSED_KEYS = MAIN_NOT_GUESSED_KEY1 + HELPER_GUESSED_KEY2

truth_key1 = 0
truth_key2 = 0

DIRECT_IMPROVE = 0
USE_MULTI_DIFF = 1
USE_MULTI_DIFF_NB = 1
PRE_RN = 4
TOTAL_R = 13
comp = '1+{0}+{1}+1'.format(PRE_RN-1, TOTAL_R-PRE_RN-1)
TOTAL_NB = 12
NB_VARIFY = 8
CAND_KN1 = 64
CAND_KN2 = 32
KG = 6
cutoff1 = 22
cutoff2 = -570
num_structures = 1<<11
it = 4 * num_structures
NS = log2(num_structures)
NIT = log2(it)
BI = 5
DC = KG + TOTAL_NB + NS + 1
EQSP = 28
logfile_fn = ''
neutral_bit_sets = []
keyschedule = 'real'


WORD_SIZE = sp.WORD_SIZE();
class P:
    c_s_attempt = True
    keys = None
    local_best_key1_value = -2000
    true_pos = None


in_diff = (0x8020, 0x4101)
out_diff = (0x0040, 0x0000)
in_diff_diff = (0x0001, 0x0000)


def cost_time(sec):
    hours = int(sec / (60*60))
    min = int(sec/60 -60*hours)
    seconds = int(sec-60*min-3600*hours)
    return "{:0>2d}:{:0>2d}:{:0>2d}".format(hours, min, seconds)

def hw(v):
  res = np.zeros(v.shape,dtype=np.uint8);
  for i in range(16):
    res = res + ((v >> i) & 1)
  return(res);

low_weight = np.array(range(2**WORD_SIZE), dtype=np.uint16);
low_weight = low_weight[hw(low_weight) <= LOW_WEIGHT];

def GET1(a, i):
    return ((a & (1 << i)) >> i)

def replace_1bit(a, b, i):
    mask = 0xffff ^ (1 << i)
    a = a & mask
    a = a | (b << i)
    return a

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
    if USE_MULTI_DIFF_NB == 1:
      p0 = np.concatenate([p0, p0 ^ 0x0020], axis = 1)
      p1 = np.concatenate([p1, p1], axis = 1)
      p0b = np.concatenate([p0b, p0b ^ 0x0060], axis = 1)
      p1b = np.concatenate([p1b, p1b], axis = 1)
    return p0, p1, p0b, p1b

#generate a Speck key, return expanded key
def gen_key(nr):
  global in_diff
  key = np.frombuffer(urandom(8),dtype=np.uint16);
  if in_diff in [(0x8020, 0x4101), (0x8021, 0x4101), (0x8060, 0x4101), (0x8061, 0x4101)]:
    ks = sp.expand_key_weakkey(key, nr, 2, 12, 1);
  else:
    ks = sp.expand_key(key, nr);
  return(ks);

def gen_plain(n):
  pt0 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  pt1 = np.frombuffer(urandom(2*n),dtype=np.uint16);
  return(pt0, pt1);

def gen_challenge(n, nr, diff, neutral_bits, keyschedule='real'):
  global logfile
  neutral_bits = neutral_bits
  key = gen_key(nr);
  if (keyschedule is 'free'): key = np.frombuffer(urandom(2*nr),dtype=np.uint16);
  #
  if (PRE_RN == 4):
    n = n >> (USE_MULTI_DIFF)

  pt0, pt1 = gen_plain(n);

  if (PRE_RN == 4) and (KG > 0):
    # Key Guess for right pairs
    set_bit_x7 = GET1(key[0], 7)
    pt0 = replace_1bit(pt0, set_bit_x7, 7)
    if (USE_MULTI_DIFF == 1) and (KG > 1):
      set_bit_x15_y8 = GET1(pt0, 15) ^ GET1(key[0], 15) ^ GET1(key[0], 8)
      pt1 = replace_1bit(pt1, set_bit_x15_y8, 8)
    if (USE_MULTI_DIFF == 0) and (KG > 2):
      set_bit_x5_y14 = GET1(pt0, 5) ^ GET1(key[0], 5) ^ GET1(key[0], 14) ^ 1
      pt1 = replace_1bit(pt1, set_bit_x5_y14, 14)
    #
    if ((USE_MULTI_DIFF == 2) and (KG > 1)) or ((USE_MULTI_DIFF == 1) and (KG > 2)) or ((USE_MULTI_DIFF == 0) and (KG > 3)):
      # Key Guess for For [5, 28]
      set_bit_x12_y5 = GET1(pt0, 12) ^ GET1(key[0], 12) ^ GET1(key[0], 5) ^ 1
      pt1 = replace_1bit(pt1, set_bit_x12_y5, 5)
    if ((USE_MULTI_DIFF == 2) and (KG > 2)) or ((USE_MULTI_DIFF == 1) and (KG > 3)) or ((USE_MULTI_DIFF == 0) and (KG > 4)):
      # Key Guess for For [15, 24]
      set_bit_y1 = GET1(key[0], 1)
      pt1 = replace_1bit(pt1, set_bit_y1, 1)
    if ((USE_MULTI_DIFF == 2) and (KG > 3)) or ((USE_MULTI_DIFF == 1) and (KG > 4)) or ((USE_MULTI_DIFF == 0) and (KG > 5)):
      # Key Guess for For [4, 27, 29]
      set_bit_x11_y4 = GET1(pt0, 11) ^ GET1(key[0], 11) ^ GET1(key[0], 4) ^ 1
      pt1 = replace_1bit(pt1, set_bit_x11_y4, 4)
    if ((USE_MULTI_DIFF == 2) and (KG > 4)) or ((USE_MULTI_DIFF == 1) and (KG > 5)) or ((USE_MULTI_DIFF == 0) and (KG > 6)):
      # Key Guess for For [6, 12, 11, 18]
      set_bit_x2_y11 = GET1(pt0, 2) ^ GET1(key[0], 2) ^ GET1(key[0], 11)
      pt1 = replace_1bit(pt1, set_bit_x2_y11, 11)
    if ((USE_MULTI_DIFF == 2) and (KG > 5)) or ((USE_MULTI_DIFF == 1) and (KG > 6)) or ((USE_MULTI_DIFF == 0) and (KG > 7)):
      # Key Guess for For [2, 25, 27, 6], [2, 4, 25]
      set_bit_x9_y2 = GET1(pt0, 9) ^ GET1(key[0], 9) ^ GET1(key[0], 2) ^ 1
      pt1 = replace_1bit(pt1, set_bit_x9_y2, 2)

  print("neutral_bits:", neutral_bits, file=logfile, flush=True)
  pt0a, pt1a, pt0b, pt1b = make_plain_structure(pt0, pt1, diff, neutral_bits);
  if (PRE_RN == 4) and (USE_MULTI_DIFF == 2):
    pt0c = np.copy(pt0b)
    pt1c = np.copy(pt1b)
    pt0c = pt0c ^ in_diff_diff[0]
    pt1c = pt1c ^ in_diff_diff[1]
  pt0a, pt1a = sp.dec_one_round((pt0a, pt1a),0);
  pt0b, pt1b = sp.dec_one_round((pt0b, pt1b),0);
  ct0a, ct1a = sp.encrypt((pt0a, pt1a), key);
  ct0b, ct1b = sp.encrypt((pt0b, pt1b), key);

  if (PRE_RN == 4) and (USE_MULTI_DIFF == 2):
    # from (8020 4101) => (8021 4101), require n//4 new queries
    pt0c, pt1c = sp.dec_one_round((pt0c, pt1c),0);
    ct0c, ct1c = sp.encrypt((pt0c, pt1c), key);
    ct0a_t = np.copy(ct0a)
    ct1a_t = np.copy(ct1a)
    ct0a = np.concatenate([ct0a, ct0a_t])
    ct1a = np.concatenate([ct1a, ct1a_t])
    ct0b = np.concatenate([ct0b, ct0c])
    ct1b = np.concatenate([ct1b, ct1c])
  
  if (PRE_RN == 4) and (USE_MULTI_DIFF >= 1):
    # For 4-r differential trail from (8020 4101) (8021 4101) => (8060 4101) (8061 4101)
    # require no new queries because of neutral_bit [22], ie, 0040 0000, [22] must be the first neutral bit in neutral_bits
    ct0a_t = np.copy(ct0a)
    ct1a_t = np.copy(ct1a)
    ct0b_t = np.copy(ct0b)
    ct1b_t = np.copy(ct1b)
    nb_n = len(ct0b[0])
    for ci in range(nb_n//2): # The first neutral bit in neutral_bits must be [22]
      ct0b_t[:,[2*ci, 2*ci+1]] = ct0b_t[:,[2*ci+1, 2*ci]]
      ct1b_t[:,[2*ci, 2*ci+1]] = ct1b_t[:,[2*ci+1, 2*ci]]
    ct0a = np.concatenate([ct0a, ct0a_t])
    ct1a = np.concatenate([ct1a, ct1a_t])
    ct0b = np.concatenate([ct0b, ct0b_t])
    ct1b = np.concatenate([ct1b, ct1b_t])

  # find correct index
  p_01, p_02 = sp.decrypt((ct0a, ct1a), key[-(nr-PRE_RN):])
  p_11, p_12 = sp.decrypt((ct0b, ct1b), key[-(nr-PRE_RN):])
  diff0 = p_01 ^ p_11
  diff1 = p_02 ^ p_12
  temp = []
  for i in range(len(diff0)):
    t0 = ((diff0[i][0] == out_diff[0]) and (diff1[i][0] == out_diff[1]))
    if t0:
      temp.append(i)
  print("true pod:", temp, file=logfile, flush=True)
  P.true_pos = temp
  truepod_n = len(temp)
  return([ct0a, ct1a, ct0b, ct1b], key, truepod_n);

def find_good(cts, key, nr=3, target_diff = (0x0040,0x0)):
  pt0a, pt1a = sp.decrypt((cts[0], cts[1]), key[nr:]);
  pt0b, pt1b = sp.decrypt((cts[2], cts[3]), key[nr:]);
  diff0 = pt0a ^ pt0b; diff1 = pt1a ^ pt1b;
  d0 = (diff0 == target_diff[0]); d1 = (diff1 == target_diff[1]);
  d = d0 * d1;
  v = np.sum(d,axis=1);
  return(v);

#having a good key candidate, exhaustively explore all keys with hamming distance less than two of this key
def verifier_search(cts, best_guess, use_n, net):
  ck1 = best_guess[0] ^ low_weight;
  ck2 = best_guess[1] ^ low_weight;
  n = len(ck1);
  ck1 = np.repeat(ck1, n); keys1 = np.copy(ck1);
  ck2 = np.tile(ck2, n); keys2 = np.copy(ck2);
  ck1 = np.repeat(ck1, use_n);
  ck2 = np.repeat(ck2, use_n);
  ct0a = np.tile(cts[0][0:use_n], n*n);
  ct1a = np.tile(cts[1][0:use_n], n*n);
  ct0b = np.tile(cts[2][0:use_n], n*n);
  ct1b = np.tile(cts[3][0:use_n], n*n);
  pt0a, pt1a = sp.dec_one_round((ct0a, ct1a), ck1);
  pt0b, pt1b = sp.dec_one_round((ct0b, ct1b), ck1);
  pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), ck2);
  pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), ck2);
  X = sp.convert_to_binary([pt0a, pt1a, pt0b, pt1b]);
  Z = net.predict(X, batch_size=BS);
  Z = Z / (1 - Z);
  Z = np.log2(Z + Z_min);
  Z = Z.reshape(-1, use_n);
  v = np.mean(Z, axis=1) * len(cts[0]);
  m = np.argmax(v); val = v[m];
  key1 = keys1[m]; key2 = keys2[m];
  return(key1, key2, val);


#test wrong-key decryption
def wrong_key_decryption(n, diff, nr, net):
  global logfile
  means = np.zeros(2**16); sig = np.zeros(2**16);
  for i in range(2**16):
    keys = np.frombuffer(urandom(8*n),dtype=np.uint16).reshape(4,-1);
    ks = sp.expand_key(keys, nr+1);
    pt0a = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt1a = np.frombuffer(urandom(2*n),dtype=np.uint16);
    pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1];
    ct0a, ct1a = sp.encrypt((pt0a, pt1a), ks);
    ct0b, ct1b = sp.encrypt((pt0b, pt1b), ks);
    rsubkeys = i ^ ks[nr];
    c0a, c1a = sp.dec_one_round((ct0a, ct1a),rsubkeys);
    c0b, c1b = sp.dec_one_round((ct0b, ct1b),rsubkeys);
    X = sp.convert_to_binary([c0a, c1a, c0b, c1b]);
    Z = net.predict(X,batch_size=BS);
    Z = Z.flatten();
    means[i] = np.mean(Z);
    sig[i] = np.std(Z);
  return(means, sig);

#here, we use some symmetries of the wrong key performance profile
#by performing the optimization step only on the 14 lowest bits and randomizing the others
#on CPU, this only gives a very minor speedup, but it is quite useful if a strong GPU is available
#In effect, this is a simple partial mitigation of the fact that we are running single-threaded numpy code here
tmp_br1 = np.arange(2**MAIN_GUESSED_KEY1, dtype=np.uint16);
tmp_br1 = np.repeat(tmp_br1, CAND_KN1).reshape(-1,CAND_KN1);

def bayesian_rank_kr1(cand, emp_mean, m, s):
  global logfile
  global tmp_br1;
  n = len(cand);
  if (tmp_br1.shape[1] != n):
      tmp_br1 = np.arange(2**MAIN_GUESSED_KEY1, dtype=np.uint16);
      tmp_br1 = np.repeat(tmp_br1, n).reshape(-1,n);
  tmp = tmp_br1 ^ cand;
  v = (emp_mean - m[tmp]) * s[tmp];
  v = v.reshape(-1, n);
  scores = np.linalg.norm(v, axis=1);
  return(scores);

tmp_br2 = np.arange(2**HELPER_GUESSED_KEYS, dtype=np.uint16);
tmp_br2 = np.repeat(tmp_br2, (1<<MAIN_NOT_GUESSED_KEY1)*CAND_KN2).reshape(-1,(1<<MAIN_NOT_GUESSED_KEY1)*CAND_KN2);

def bayesian_rank_kr2(cand, emp_mean, m, s):
  global logfile
  global tmp_br2;
  n = len(cand);
  if (tmp_br2.shape[1] != n):
      tmp_br2 = np.arange(2**HELPER_GUESSED_KEYS, dtype=np.uint16);
      tmp_br2 = np.repeat(tmp_br2, n).reshape(-1,n);
  tmp = tmp_br2 ^ cand;
  v = (emp_mean - m[tmp]) * s[tmp];
  v = v.reshape(-1, n);
  scores = np.linalg.norm(v, axis=1);
  return(scores);

def bayesian_key_recovery1(cts, net, m, s, num_cand = CAND_KN1, num_iter=5, seed = None):
  global logfile
  n = len(cts[0]);
  keys = np.random.choice(2**(WORD_SIZE-2),num_cand,replace=False); scores = 0; best = 0;
  if (not seed is None):
    keys = np.copy(seed);
  ct0a, ct1a, ct0b, ct1b = np.tile(cts[0],num_cand), np.tile(cts[1], num_cand), np.tile(cts[2], num_cand), np.tile(cts[3], num_cand);
  scores = np.zeros(2**(WORD_SIZE-2));
  used = np.zeros(2**(WORD_SIZE-2));
  all_keys = np.zeros(num_cand * (num_iter - 1),dtype=np.uint16);
  all_v = np.zeros(num_cand * (num_iter - 1));
  for i in range(num_iter):
    k = np.repeat(keys, n);
    c0a, c1a = sp.dec_one_round((ct0a, ct1a),k); c0b, c1b = sp.dec_one_round((ct0b, ct1b),k);
    X = sp.convert_to_binary([c0a, c1a, c0b, c1b]);
    Z = net.predict(X,batch_size=BS);
    Z = Z.reshape(num_cand, -1);
    #print("Z = net predict\n", Z)
    means = np.mean(Z, axis=1);
    #print("means = np.mean(Z, axis=1)\n", means)
    Z = Z/(1-Z); Z = np.log2(Z + Z_min);
    #print("Z = np.log2(Z)\n", Z)
    v =np.sum(Z, axis=1);
    #print("value = np.sum(Z, axis=1)\n", v)
    if i > 0:
      all_v[(i - 1) * num_cand:(i)*num_cand] = v;
      all_keys[(i - 1) * num_cand:(i)*num_cand] = np.copy(keys);
    scores = bayesian_rank_kr1(keys, means, m=m, s=s);
    tmp = np.argpartition(scores+used, num_cand)
    keys = tmp[0:num_cand];
    r = np.random.randint(0,1<<MAIN_NOT_GUESSED_KEY1,num_cand,dtype=np.uint16); r = r << MAIN_GUESSED_KEY1; keys = keys ^ r;
  return(all_keys, scores, all_v);

def bayesian_key_recovery2(gkey1, cts, net, m, s, num_cand = CAND_KN2, num_iter=5, seed = None):
  global logfile
  num_cand = num_cand * (1<<MAIN_NOT_GUESSED_KEY1)
  main_gkey1 = gkey1 & ((1<<MAIN_GUESSED_KEY1) - 1)
  n = len(cts[0]);
  keys = np.random.choice(2**(HELPER_GUESSED_KEYS),num_cand,replace=False); scores = 0; best = 0;
  if (not seed is None):
    keys = np.copy(seed);
  ct0a, ct1a, ct0b, ct1b = np.tile(cts[0],num_cand), np.tile(cts[1], num_cand), np.tile(cts[2], num_cand), np.tile(cts[3], num_cand);
  scores = np.zeros(2**(HELPER_GUESSED_KEYS));
  used = np.zeros(2**(HELPER_GUESSED_KEYS));
  all_keys = np.zeros(num_cand * (num_iter - 1),dtype=np.uint32);
  all_v = np.zeros(num_cand * (num_iter - 1));
  for i in range(num_iter):
    k = np.repeat(keys, n);
    k1 = np.uint16(((k & KEY1_MAIN_LEFT_MASK) << KEY1_MAIN_LEFT_OFFSET)) | main_gkey1
    k2 = np.uint16(k >> MAIN_NOT_GUESSED_KEY1)
    c0a, c1a = sp.dec_one_round((ct0a, ct1a),k1);
    c0b, c1b = sp.dec_one_round((ct0b, ct1b),k1);
    c0a, c1a = sp.dec_one_round((c0a, c1a),k2);
    c0b, c1b = sp.dec_one_round((c0b, c1b),k2);
    X = sp.convert_to_binary([c0a, c1a, c0b, c1b]);
    Z = net.predict(X,batch_size=BS);
    Z = Z.reshape(num_cand, -1);
    #print("Z = net predict\n", Z)
    means = np.mean(Z, axis=1);
    #print("means = np.mean(Z, axis=1)\n", means)
    Z = Z/(1-Z); Z = np.log2(Z + Z_min);
    #print("Z = np.log2(Z)\n", Z)
    v =np.sum(Z, axis=1);
    #print("value = np.sum(Z, axis=1)\n", v)
    if i > 0:
      all_v[(i - 1) * num_cand:(i)*num_cand] = v;
      all_keys[(i - 1) * num_cand:(i)*num_cand] = np.copy(keys);
    scores = bayesian_rank_kr2(keys, means, m=m, s=s);
    tmp = np.argpartition(scores+used, num_cand)
    keys = tmp[0:num_cand];
    r = np.random.randint(0,1<<(HELPER_NOT_GUESS_KEY2),num_cand,dtype=np.uint32); r = r << (HELPER_GUESSED_KEYS); keys = keys ^ r;
  return(all_keys, scores, all_v);


def test_bayes(cts,it, cutoff1, cutoff2, net, net_help, m_main, m_help, s_main, s_help, verify_breadth):
  global logfile
  global bayesianIter1
  global bayesianIter2
  n = len(cts[0]);
  if (verify_breadth is None): verify_breadth=len(cts[0][0]);
  alpha = sqrt(n);
  best_val = [-10000.0, -10000.0]; best_key = [0,0]; best_pod = 0; bp = 0; bv = -10000.0;
  best_val_beforeImp = [-10000.0, -10000.0];
  keys = np.random.choice(2**WORD_SIZE, 32, replace=False);
  eps = 0.0001; local_best = np.full(n,-1000); num_visits = np.full(n,eps);
  local_best_key1 = np.zeros(n, dtype=np.uint16)
  local_best_key2 = np.zeros(n, dtype=np.uint16)
  local_best2 = np.full(n,-1000)
  guess_count = np.zeros(2**16,dtype=np.uint16);
  updated = False
  for j in range(it):
      if j % 100 == 0:
            print(j, "th iteration: ", file=logfile, flush=True)
      priority = local_best + alpha * np.sqrt(log2(j+1) / num_visits); i = np.argmax(priority);
      num_visits[i] = num_visits[i] + 1;
      if (best_val[1] > cutoff2):
        best_val_beforeImp[0] = best_val[0]; best_val_beforeImp[1] = best_val[1]
        improvement = (verify_breadth > 0);
        while improvement:
          k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key,net=net_help, use_n = verify_breadth);
          improvement = (val > best_val[1]);
          if (improvement):
            best_key = [k1, k2]; best_val[0] = local_best[best_pod]; best_val[1] = val;
            print("Improvement mid: ", j, "th iteration", file=logfile)
            print("best_pod", best_pod, " ", best_pod in P.true_pos, file=logfile)
            print("best_val", best_val, file=logfile)
            print("best_key", "{0:04x}".format(best_key[0]), "{0:04x}".format(best_key[1]), file=logfile, flush=True)
            print("true_key", "{0:04x}".format(truth_key1), "{0:04x}".format(truth_key2), file=logfile)
            print("diffhw", "{0:2d}".format(hw(best_key[0]^truth_key1)), " ", "{0:2d}".format(hw(best_key[1]^truth_key2)), file=logfile, flush=True)
        return(best_key, j, best_val_beforeImp);
      keys, scores, v = bayesian_key_recovery1([cts[0][i], cts[1][i], cts[2][i], cts[3][i]], num_cand=CAND_KN1, num_iter=bayesianIter1, net=net, m=m_main, s=s_main);
      vtmp = np.max(v);
      max_vtmp_idx = np.argmax(v)
      key1_max = keys[max_vtmp_idx]
      if (vtmp > local_best[i]):
        local_best[i] = vtmp;
        local_best_key1[i] = key1_max
      if (vtmp > bv):
        bv = vtmp; bp = i;
      print('index: ', '{:<5}'.format(i), ", ", '{0:<6}'.format(i in P.true_pos), ',  num_visits[index]:', '{:<10d}'.format(int(num_visits[i])), ", key1_max: ", "{0:04x}".format(key1_max), ", truth_key1: ", "{0:04x}".format(truth_key1), ", diffhw1: ", "{0:2d}".format(hw(key1_max^truth_key1)), ", v_tmp_1:", '{:<20}'.format(vtmp), file=logfile, flush=True)
      if (vtmp > cutoff1):
        l2 = [i for i in range(len(keys)) if v[i] > cutoff1];
        print("v_tmp_1:",  vtmp, "cutoff_1:", cutoff1, file=logfile)
        for i2 in l2:
          #c0a, c1a = sp.dec_one_round((cts[0][i],cts[1][i]),keys[i2]);
          #c0b, c1b = sp.dec_one_round((cts[2][i],cts[3][i]),keys[i2]);
          keys2,scores2,v2 = bayesian_key_recovery2(keys[i2], [cts[0][i],cts[1][i], cts[2][i],cts[3][i]], num_cand=CAND_KN2, num_iter=bayesianIter2, net=net_help, m=m_help, s=s_help);
          vtmp2 = np.max(v2);
          max_vtmp2_idx = np.argmax(v2)
          k_tmp_2 = keys2[max_vtmp2_idx]
          main_gkey1 = np.uint16(keys[i2] & ((1<<MAIN_GUESSED_KEY1) - 1))
          helper_gkey1 = np.uint16(((k_tmp_2 & KEY1_MAIN_LEFT_MASK) << KEY1_MAIN_LEFT_OFFSET)) | main_gkey1
          helper_gkey2 = np.uint16(k_tmp_2 >> MAIN_NOT_GUESSED_KEY1)
          print("v1: ", '{:<20}'.format(v[i2]), ", k1: ", "{0:04x}".format(helper_gkey1), ", truth_key1: ", "{0:04x}".format(truth_key1), ", diffhw1: ", "{0:2d}".format(hw(helper_gkey1^truth_key1)), ", max v2: ", '{:<20}'.format(vtmp2), ", k2:", "{0:04x}".format(helper_gkey2), ", truth_key2: ", "{0:04x}".format(truth_key2), ", diffhw2: ", "{0:2d}".format(hw(helper_gkey2^truth_key2)), file=logfile, flush=True)
          #
          v_tmp_2_best = vtmp2
          k_tmp_1_best = helper_gkey1
          k_tmp_2_best = helper_gkey2

          if (v_tmp_2_best > local_best2[i]):
            local_best2[i] = vtmp2
            local_best_key1[i] = k_tmp_1_best
            local_best_key2[i] = k_tmp_2_best

          if DIRECT_IMPROVE == 1:
            improvement = (verify_breadth > 0)
            best_key = [k_tmp_1_best, k_tmp_2_best]; best_val[0] = v[i2]; best_val[1] = v_tmp_2_best;
            best_val_beforeImp[0] = best_val[0]; best_val_beforeImp[1] = best_val[1]
            while improvement:
                tmp_cipher_structure_k2 = [cts[0][i], cts[1][i], cts[2][i], cts[3][i]]
                key_1, key_2, value = verifier_search(tmp_cipher_structure_k2, (k_tmp_1_best, k_tmp_2_best), net=net_help, use_n=verify_breadth)
                improvement = (value > v_tmp_2_best)
                if (improvement):
                    k_tmp_1_best = key_1
                    k_tmp_2_best = key_2
                    v_tmp_2_best = value
                    best_key = [k_tmp_1_best, k_tmp_2_best]; best_val[0] = v[i2]; best_val[1] = v_tmp_2_best;
                    print(
                        "Improvement key2: v_tmp_2_best ", v_tmp_2_best,
                        "best key", "{0:04x}".format(k_tmp_1_best), "{0:04x}".format(k_tmp_2_best), file=logfile, flush=True)
          if (v_tmp_2_best > best_val[1]):
            print("v_tmp_2:",  v_tmp_2_best, "best_val:", best_val[1], file=logfile)
            best_val[1] = v_tmp_2_best; best_val[0] = v[i2]; best_key = [k_tmp_1_best, k_tmp_2_best]; best_pod=i;
            print("best_pod", best_pod, " ", best_pod in P.true_pos, file=logfile)
            print("best_val", best_val, file=logfile)
            print("best_key", "{0:04x}".format(best_key[0]), "{0:04x}".format(best_key[1]), file=logfile)
            print("true_key", "{0:04x}".format(truth_key1), "{0:04x}".format(truth_key2), file=logfile)
            print("diffhw", "{0:2d}".format(hw(best_key[0]^truth_key1)), " ", "{0:2d}".format(hw(best_key[1]^truth_key2)), file=logfile, flush=True)
            updated = True
          if DIRECT_IMPROVE == 1:
            if (best_val[1] > cutoff2):
              return (best_key, j, best_val_beforeImp)
  if updated == False:
    best_val[0] = np.max(local_best)
    best_pod = np.argmax(local_best)
    best_key[0] = local_best_key1[best_pod]
    keys2,scores2,v2 = bayesian_key_recovery2(best_key[0], [cts[0][best_pod],cts[1][best_pod], cts[2][best_pod],cts[3][best_pod]], num_cand=CAND_KN2, num_iter=bayesianIter2, net=net_help, m=m_help, s=s_help);
    best_val[1] = np.max(v2);
    k_tmp_2 = keys2[np.argmax(v2)]
    main_gkey1 = best_key[0] & ((1<<MAIN_GUESSED_KEY1) - 1)
    best_key[0] = np.uint16(((k_tmp_2 & KEY1_MAIN_LEFT_MASK) << KEY1_MAIN_LEFT_OFFSET)) | main_gkey1
    best_key[1] = np.uint16(k_tmp_2 >> MAIN_NOT_GUESSED_KEY1)

  print("After ", it, " iterations:", file=logfile)
  print("best_pod", best_pod, " ", best_pod in P.true_pos, file=logfile)
  print("best_val", best_val, file=logfile)
  print("best_key", "{0:04x}".format(best_key[0]), "{0:04x}".format(best_key[1]), file=logfile)
  print("true_key", "{0:04x}".format(truth_key1), "{0:04x}".format(truth_key2), file=logfile)
  print("diffhw", "{0:2d}".format(hw(best_key[0]^truth_key1)), " ", "{0:2d}".format(hw(best_key[1]^truth_key2)), file=logfile, flush=True)
  best_val_beforeImp[0] = best_val[0]; best_val_beforeImp[1] = best_val[1]
  improvement = (verify_breadth > 0);
  while improvement:
    k1, k2, val = verifier_search([cts[0][best_pod], cts[1][best_pod], cts[2][best_pod], cts[3][best_pod]], best_key, net=net_help, use_n = verify_breadth);
    improvement = (val > best_val[1]);
    if (improvement):
      best_key = [k1, k2]; best_val[0] = local_best[best_pod]; best_val[1] = val;
      print("Improvement last: ", it, " iterations:", file=logfile)
      print("best_pod", best_pod, " ", best_pod in P.true_pos, file=logfile)
      print("best_val", best_val, file=logfile)
      print("best_key", "{0:04x}".format(best_key[0]), "{0:04x}".format(best_key[1]), file=logfile)
      print("true_key", "{0:04x}".format(truth_key1), "{0:04x}".format(truth_key2), file=logfile)
      print("diffhw", "{0:2d}".format(hw(best_key[0]^truth_key1)), " ", "{0:2d}".format(hw(best_key[1]^truth_key2)), file=logfile, flush=True)
  return(best_key, it, best_val_beforeImp);

def set_conf():
    global logfile
    global logfile_fn
    global TOTAL_R
    global PRE_RN
    global comp
    global KG
    global TOTAL_NB
    global NB_VARIFY
    global cutoff1
    global cutoff2
    global num_structures
    global it
    global NS
    global NIT
    global CAND_KN1
    global CAND_KN2
    global BI
    global bayesianIter1
    global bayesianIter2
    global USE_MULTI_DIFF
    global USE_MULTI_DIFF_NB
    global in_diff
    global truth_key1
    global truth_key2
    global test_n
    global neutral_bit_sets
    global keyschedule
    TOTAL_R = 13
    BI = 5
    bayesianIter1 = BI
    bayesianIter2 = BI
    if TOTAL_R == 13:
      PRE_RN = 4
      KG=5
      USE_MULTI_DIFF = 1
      USE_MULTI_DIFF_NB = 1
      TOTAL_NB = 11
      NB_VARIFY = 8
      CAND_KN1 = 32
      CAND_KN2 = 32
      num_structures=1<<12 # Real data complexity: 2^(num_structures + KG) * 2^TOTAL_NB
      it=num_structures * 4
      cutoff1= 18.0
      cutoff2= -500.0
    elif TOTAL_R == 12:
      PRE_RN = 4
      if PRE_RN == 4:
        KG = 1
        USE_MULTI_DIFF = 2
        USE_MULTI_DIFF_NB = 1
        TOTAL_NB = 5
        NB_VARIFY = 6
        CAND_KN1 = 64
        CAND_KN2 = 32
        num_structures=1<<12
        it=num_structures * 2
        cutoff1= 8
        cutoff2= 10
      if PRE_RN == 3:
        KG = 0
        USE_MULTI_DIFF = 0
        USE_MULTI_DIFF_NB = 0
        TOTAL_NB = 13
        NB_VARIFY = 8
        CAND_KN1 = 32
        CAND_KN2 = 32
        num_structures= 1<<8 #500
        it=num_structures * 4 #2000
        cutoff1= 15
        cutoff2= 500
    elif TOTAL_R == 11:
      PRE_RN = 3
      KG = 0
      USE_MULTI_DIFF = 0
      USE_MULTI_DIFF_NB = 0
      TOTAL_NB = 6
      NB_VARIFY = 6
      CAND_KN1 = 32
      CAND_KN2 = 32
      num_structures=1<<7
      it=num_structures * 4
      cutoff1= 15.0
      cutoff2= 100.0
    else:
      PRE_RN = 3
      KG = 0
      USE_MULTI_DIFF = 1
      USE_MULTI_DIFF_NB = 0
      TOTAL_NB = 5
      NB_VARIFY = 5
      CAND_KN1 = 32
      CAND_KN2 = 32
      num_structures=1<<6
      it=num_structures * 4
      cutoff1= 8.0
      cutoff2= 20.0
    if PRE_RN == 3:
      in_diff = (0x0211, 0x0a04)
      neutral_bit_sets_0211_0a04 = [[21], [20], [22], [9, 16], [2, 11, 25], [14], [15], [6, 29], [23], [30], [7], [0], [8]]
      neutral_bit_sets = neutral_bit_sets_0211_0a04[:TOTAL_NB]
    if PRE_RN == 4:
      in_diff = (0x8020, 0x4101)
      if USE_MULTI_DIFF_NB == 0:
        neutral_bit_sets_8020_4101_G = [[22], [13], [20], [5, 28], [15, 24], [12, 19], [6, 29], [4, 27, 29], [6, 12, 11, 18], [14, 21], [0, 8, 31], [30], [2, 4, 25]]
      else:
        neutral_bit_sets_8020_4101_G = [[22], [13], [20], [5, 28], [15, 24], [12, 19], [6, 29], [4, 27, 29], [14, 21], [0, 8, 31], [30], [6, 12, 11, 18], [2, 4, 25]]
      neutral_bit_sets_8020_4101_NoG = [[22], [13], [20], [12, 19], [6, 29], [14, 21], [0, 8, 31], [30]]
      if TOTAL_NB <= 13 and TOTAL_NB > 8:
        neutral_bit_sets = neutral_bit_sets_8020_4101_G[:TOTAL_NB]
      if TOTAL_NB <= 8:
        neutral_bit_sets = neutral_bit_sets_8020_4101_NoG[:TOTAL_NB]
      if TOTAL_NB == 5:
        if USE_MULTI_DIFF == 2:
          neutral_bit_sets = [[22], [13], [20], [12, 19], [14, 21]]
        else:
          neutral_bit_sets = [[22], [13], [20], [12, 19], [6, 29]]
    keyschedule='real'
    now = datetime.now()
    current_time = now.strftime("%Y:%m:%d:%H:%M:%S:")
    current_time = current_time.replace(":", "_")
    logfile_fn = current_time + '_KG' + str(KG) + "_NB" + str(len(neutral_bit_sets)) + "_BI1_" + str(bayesianIter1) + "_BI2_" + str(bayesianIter2) + "_CANDKN1_" + str(CAND_KN1) + "_CANDKN2_" + str(CAND_KN2) + "_NS_" + str(num_structures) + "_IT_" + str(it) + "_c1_" + str(cutoff1) + "_c2_" + str(cutoff2) + "_r" + str(TOTAL_R)

def test(idx):
  global logfile
  global logfile_fn
  global TOTAL_R
  global PRE_RN
  global comp
  global KG
  global TOTAL_NB
  global NB_VARIFY
  global cutoff1
  global cutoff2
  global num_structures
  global it
  global NS
  global NIT
  global CAND_KN1
  global CAND_KN2
  global BI
  global bayesianIter1
  global bayesianIter2
  global USE_MULTI_DIFF
  global USE_MULTI_DIFF_NB
  global in_diff
  global truth_key1
  global truth_key2
  global test_n
  global neutral_bit_sets
  global keyschedule
  nr = TOTAL_R
  this_logfile_fn = logfile_fn + '_proc' + str(idx) + '.txt'
  logfile = open(this_logfile_fn, 'w+')
  from tensorflow.keras.models import model_from_json
  from tensorflow.keras.models import load_model
  from tensorflow.python.keras import backend as K
  import tensorflow as tf
  config = tf.compat.v1.ConfigProto()
  config.gpu_options.allow_growth = False
  K.set_session(tf.compat.v1.Session(config=config))
  from tensorflow.python.client import device_lib 
  divices = device_lib.list_local_devices()
  print(divices, file=logfile, flush=True)

  with tf.device("/gpu:"+str(idx)):
    print("Checking Speck32/64 implementation.", file=logfile);
    if (not sp.check_testvector()):
      print("Error. Aborting.", file=logfile);
      return(0);
    #load distinguishers
    json_file = open('single_block_resnet.json','r');
    json_model = json_file.read();
    net8 = model_from_json(json_model);
    net8.load_weights('net8_small.h5');
    m8 = np.load('data_wrong_key_8r_mean_1e6.npy');
    s8 = np.load('data_wrong_key_8r_std_1e6.npy'); s8 = 1.0/s8;
    net7 = model_from_json(json_model);
    net7.load_weights('net7_small.h5');
    if (TOTAL_R == 12 and PRE_RN == 4) or (TOTAL_R == 11):
      m7 = np.load('data_wrong_key_mean_7r.npy');
      s7 = np.load('data_wrong_key_std_7r.npy'); s7 = 1.0/s7;
    else:
      m7 = np.load('./DN16_lastmix/net7_small._DN16_mean_combine.npy');
      s7 = np.load('./DN16_lastmix/net7_small._DN16_std_combine.npy'); s7 = 1.0/s7;
    net6 = model_from_json(json_model);
    net6.load_weights('net6_small.h5');
    if (TOTAL_R == 12 and PRE_RN == 4) or (TOTAL_R == 11):
      m6 = np.load('./DN16_lastmix/net6_small._DN16_mean_combine.npy');
      s6 = np.load('./DN16_lastmix/net6_small._DN16_std_combine.npy'); s6 = 1.0/s6;
    else:
      m6 = np.load('data_wrong_key_mean_6r.npy');
      s6 = np.load('data_wrong_key_std_6r.npy'); s6 = 1.0/s6;
    net5 = model_from_json(json_model);
    net5.load_weights('net5_small.h5');
    m5 = np.load('data_wrong_key_mean_5r.npy');
    s5 = np.load('data_wrong_key_std_5r.npy'); s5 = 1.0/s5;

    if TOTAL_R == 13:
      net=net8
      net_help=net7
      m_main=m8
      s_main=s8
      m_help=m7
      s_help=s7
    if TOTAL_R == 12:
      if PRE_RN == 3:
        net=net8
        net_help=net7
        m_main=m8
        s_main=s8
        m_help=m7
        s_help=s7
      if PRE_RN == 4:
        net=net7
        net_help=net6
        m_main=m7
        s_main=s7
        m_help=m6
        s_help=s6
    if TOTAL_R == 11:
      if PRE_RN == 3:
        net=net7
        net_help=net6
        m_main=m7
        s_main=s7
        m_help=m6
        s_help=s6
      if PRE_RN == 4:
        net=net6
        net_help=net5
        m_main=m6
        s_main=s6
        m_help=m5
        s_help=s5

    print("[INFO] num_rounds:", nr, file=logfile)
    print("[INFO] cutoff_1:", cutoff1, " cutoff_2:", cutoff2, "cipher_structure_num:", num_structures, " iter_num: ", it, file=logfile)
    arr1 = np.zeros(test_n, dtype=np.uint16); arr2 = np.zeros(test_n, dtype=np.uint16);
    t0 = time();
    data = 0; av=0.0; good = np.zeros(test_n, dtype=np.uint8);
    zkey = np.zeros(nr,dtype=np.uint16);
    for i in range(test_n):
      print("Test:",i, file=logfile, flush=True);
      start_time = time()
      ct, key, truepod_n = gen_challenge(num_structures,nr, diff=in_diff, neutral_bits=neutral_bit_sets, keyschedule=keyschedule);
      if truepod_n == 0:
        continue
      truth_key1 = key[nr-1]
      truth_key2 = key[nr-2]
      print("truth key", "{0:04x}".format(truth_key1), "{0:04x}".format(truth_key2), file=logfile, flush=True)
      g = find_good(ct, key); g = np.max(g); good[i] = g;
      guess, num_used, best_val_beforeImp = test_bayes(ct,it=it, cutoff1=cutoff1, cutoff2=cutoff2, net=net, net_help=net_help, m_main=m_main, s_main=s_main, m_help=m_help, s_help=s_help, verify_breadth=1<<NB_VARIFY);
      end_time = time()
      num_used_d = min(num_structures, num_used); data = data + 2 * (2 ** len(neutral_bit_sets)) * num_used_d;
      arr1[i] = guess[0] ^ key[nr-1]; arr2[i] = guess[1] ^ key[nr-2];
      print('during time is :', cost_time(end_time - start_time), file=logfile)
      print("used iteration number is ", num_used, file=logfile)
      print("guess key: ", "{0:04x}".format(guess[0]), "{0:04x}".format(guess[1]), file=logfile)
      print("truth key: ", "{0:04x}".format(key[nr-1]), "{0:04x}".format(key[nr-2]), file=logfile)
      wbp1 = []
      wbp2 = []
      for bi in range(WORD_SIZE):
          if ((arr1[i] >> bi) & 1) == 1:
              wbp1.append(bi)
          if ((arr2[i] >> bi) & 1) == 1:
              wbp2.append(bi)
      print("Difference between real key and guess key", "{0:04x}".format(arr1[i]), "{0:04x}".format(arr2[i]), file=logfile)
      print("Wrong key bit index (last subkey1, last subkey2)", wbp1, wbp2, file=logfile)
      np.set_printoptions()
      dis = hw(arr1[i])+hw(arr2[i])
      print("Hamming weight of difference between real key and guess key: ", dis, file=logfile, flush=True)
      if dis <= 2:
        print("AttackSuccess_with_dis ", dis, " best_val_beforeImp ", best_val_beforeImp[0], best_val_beforeImp[1], file=logfile, flush=True)
    t1 = time();
    print("Done.", file=logfile);
    if data != 0:
        d1 = ["{0:04x}".format(x) for x in arr1]; d2 = ["{0:04x}".format(x) for x in arr2];
        print("Differences between guessed and last key:", d1, file=logfile);
        print("Differences between guessed and second-to-last key:", d2, file=logfile);
        print("Wall time per attack (average in seconds):", (t1 - t0)/test_n, file=logfile);
        print("Data blocks used (average, log2): ", log2(data) - log2(test_n), file=logfile);
        np.save(open(logfile_fn + "_run_sols1_" + str(nr)  + "r_proc" + str(idx) + ".npy", 'wb'), arr1);
        np.save(open(logfile_fn + "_run_sols2_" + str(nr)  + "r_proc" + str(idx) + ".npy", 'wb'), arr2);
        np.save(open(logfile_fn + "_run_good0_" + str(nr)  + "r_proc" + str(idx) + ".npy", 'wb'), good);
  logfile.close()
  return this_logfile_fn;

import scipy.stats as st
import matplotlib.pyplot as plt
from scipy import special, stats
import seaborn as sns
sns.set_style("white")
plt.rc('font', family='serif')
FntSize = 18
FntSize2 = 12
params = {'axes.labelsize': FntSize,
          'axes.titlesize': FntSize,
          'legend.loc': 'upper right'}
plt.rcParams.update(params)
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator, FixedLocator, MaxNLocator)

def get_best_distribution(data):
    dist_names = ["norm", "chi2", "genlogistic", "logistic", "gamma"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        #print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    #print("Best fitting distribution: "+str(best_dist))
    #print("Best p value: "+ str(best_p))
    #print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

# calculate the kl divergence
def kl_divergence(p, q):
	#return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))
    return stats.entropy(p, q)
 
# calculate the js divergence
def js_divergence(p, q):
	m = 0.5 * (p + q)
	return 0.5 * stats.entropy(p, m) + 0.5 * stats.entropy(q, m)

def draw_dist(fn):
    global TOTAL_R
    global PRE_RN
    global comp
    global KG
    global TOTAL_NB
    global cutoff1
    global cutoff2
    global num_structures
    global it
    global NS
    global NIT
    global CAND_KN1
    global CAND_KN2
    global BI
    global bayesianIter1
    global bayesianIter2
    global USE_MULTI_DIFF
    global USE_MULTI_DIFF_NB
    global DC
    global EQSP
    wdir = fn[:-3]
    comp = '1+{0}+{1}+1'.format(PRE_RN-1, TOTAL_R-PRE_RN-1)
    NB = TOTAL_NB
    NS = log2(num_structures)
    NIT = log2(it)
    DC = 0.0
    EQSP = 28
    if USE_MULTI_DIFF == 1:
        DC = KG + NB + NS
    elif USE_MULTI_DIFF == 2:
        DC = KG + NB + NS + log2(3/4)
    else:
        DC = KG + NB + NS + 1
    if USE_MULTI_DIFF_NB == 1:
        DC = DC + 1
    core = 'GPU'
    font = {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': FntSize,
    }
    TEST_DIST = 0
    test_n_all = 0
    cnt_done = 1e-32
    cnt_succ = 0
    diff_hw = []
    max_time = 0.0
    dur_time = []
    avg_time = 0.0
    all_v1 = []
    rand_v1 = []
    real_v1 = []
    cutoff_v1 = []
    all_v2 = []
    rand_v2 = []
    real_v2 = []
    succ_v1 = []
    succ_v2 = []
    used_it = []
    succ_flag = []
    hw_v1 = [[], [], [], []]
    is_real = 0
    passed = 0
    logfile = open(fn , 'r')
    finish_str = 'Hamming weight of difference between real key and guess key:'.split()
    for line in logfile:
        temp = line
        temp = temp.split()
        if len(temp) != 0:
            if temp[0] == 'Test:':
                test_n_all += 1
            if temp[0] == 'during':
                atttime = temp[-1]
                ho = int(atttime[0:2])
                mi = int(atttime[3:5])
                se = int(atttime[6:8])
                this_time = ho * 3600 + mi * 60 + se
                dur_time.append(this_time)
                if this_time > max_time:
                    max_time = this_time
            if temp[0] == 'Hamming':
                cnt_done += 1
                v = int(temp[len(finish_str)])
                diff_hw.append(v)
                if v <= 2:
                  cnt_succ += 1
                  succ_flag.append(1)
                else:
                  succ_flag.append(0)
            if temp[0] == 'true' and temp[1] == 'pod:' and temp[2] == '[]':
                passed += 1
            if temp[0] == 'index:':
                v = float(temp[18])
                all_v1.append(v)
                if temp[3] == '0': #'False':
                    is_real = 0
                    rand_v1.append(v)
                else:
                    is_real = 1
                    real_v1.append(v)
                    hw = int(temp[15])
                    if hw <= 3:
                        hw_v1[hw].append(v)
            if temp[0] == 'v1:':
                cv1 = float(temp[1])
                cutoff_v1.append(cv1)
                v2 = float(temp[14])
                all_v2.append(v2)
                if is_real == 0:
                    rand_v2.append(v2)
                else:
                    real_v2.append(v2)
            if temp[0] == 'AttackSuccess_with_dis':
                succ_v1_v = float(temp[3])
                succ_v1.append(succ_v1_v)
                succ_v2_v = float(temp[4])
                succ_v2.append(succ_v2_v)
            if temp[0] == 'used':
                used_it_v = float(temp[4])
                used_it.append(used_it_v)
    logfile.close()
    all_v1_len = len(all_v1)
    rand_v1_len = len(rand_v1)
    real_v1_len = len(real_v1)
    all_v1 = np.array(all_v1)
    rand_v1 = np.array(rand_v1)
    real_v1 = np.array(real_v1)
    cutoff_v1 = np.array(cutoff_v1)
    all_v2 = np.array(all_v2)
    rand_v2 = np.array(rand_v2)
    real_v2 = np.array(real_v2)
    #
    all_v1_mean = np.mean(all_v1)
    all_v1_std = np.std(all_v1)
    rand_v1_mean = np.mean(rand_v1)
    rand_v1_std = np.std(rand_v1)
    real_v1_mean = np.mean(real_v1)
    real_v1_std = np.std(real_v1)
    rand_v2_mean = np.mean(rand_v2)
    rand_v2_std = np.std(rand_v2)
    real_v2_mean = np.mean(real_v2)
    real_v2_std = np.std(real_v2)
    rand_v1_max = np.max(rand_v1)
    real_v1_max = np.max(real_v1)
    cutoff_v1_rate = np.sum(all_v1 > cutoff1)/(len(all_v1)+1e-32)
    cutoff_v1_rand_rate = np.sum(rand_v1 > cutoff1)/(rand_v1_len+1e-32);
    cutoff_v1_real_rate = np.sum(real_v1 > cutoff1)/(real_v1_len+1e-32);
    cutoff_v2_rate = np.sum(all_v2 > cutoff2)/(len(all_v2)+1e-32)
    cutoff_v2_rand_rate = np.sum(rand_v2 > cutoff2)/(len(rand_v2)+1e-32);
    cutoff_v2_real_rate = np.sum(real_v2 > cutoff2)/(len(real_v2)+1e-32);
    all_v2_mean = np.mean(all_v2)
    all_v2_std = np.std(all_v2)
    succ_v1 = np.array(succ_v1)
    succ_v2 = np.array(succ_v2)
    diff_hw = np.array(diff_hw)
    used_it = np.array(used_it)
    succ_flag = np.array(succ_flag)
    succ_used_it = used_it[succ_flag==1]
    fail_used_it = used_it[succ_flag==0]
    avg_time = np.mean(dur_time)
    #
    np.save(wdir + "all_v1.npy", all_v1)
    np.save(wdir + "rand_v1.npy", rand_v1)
    np.save(wdir + "real_v1.npy", real_v1)
    np.save(wdir + "cutoff_v1.npy", cutoff_v1)
    np.save(wdir + "all_v2.npy", all_v2)
    np.save(wdir + "rand_v2.npy", rand_v2)
    np.save(wdir + "real_v2.npy", real_v2)
    np.savetxt(wdir + 'all_v1.out', all_v1, delimiter=',')
    np.savetxt(wdir + 'rand_v1.out', rand_v1, delimiter=',')
    np.savetxt(wdir + 'real_v1.out', real_v1, delimiter=',')
    np.savetxt(wdir + 'cutoff_v1.out', cutoff_v1, delimiter=',')
    np.savetxt(wdir + 'all_v2.out', all_v2, delimiter=',')
    np.savetxt(wdir + 'rand_v2.out', rand_v2, delimiter=',')
    np.savetxt(wdir + 'real_v2.out', real_v2, delimiter=',')
    plt.figure(figsize=(20,10), dpi= 600)

    if TEST_DIST == 1:
        plot_1_lab = "rand #$2^{" + "{:<10.4f}".format(log2(rand_v1_len+1e-32)) + "}$ $\mu_w$: " + "{:<10.4f}".format(rand_v1_mean) + " $\sigma_w$: " + "{:<10.4f}".format(rand_v1_std) + r'${{\max}}_w:{:0.4f}$'.format(rand_v1_max)
        plot_2_lab = "real #$2^{" + "{:<10.4f}".format(log2(real_v1_len+1e-32)) + "}$ $\mu_r$: " + "{:<10.4f}".format(real_v1_mean) + " $\sigma_r$: " + "{:<10.4f}".format(real_v1_std) + r'${{\max}}_r:{:0.4f}$'.format(real_v1_max)
    else:
        plot_1_lab = "rand #$2^{" + "{:<10.4f}".format(log2(rand_v1_len+1e-32)) + "}$ $\mu_w$: " + "{:<10.4f}".format(rand_v1_mean) + " $\sigma_w$: " + "{:<10.4f}".format(rand_v1_std) + " $qct_w$: " + "{:<10.4f}".format(cutoff_v1_rand_rate)
        plot_2_lab = "real #$2^{" + "{:<10.4f}".format(log2(real_v1_len+1e-32)) + "}$ $\mu_r$: " + "{:<10.4f}".format(real_v1_mean) + " $\sigma_r$: " + "{:<10.4f}".format(real_v1_std) + " $qct_r$: " + "{:<10.4f}".format(cutoff_v1_real_rate)
    plot_1 = sns.histplot(rand_v1, color="dodgerblue", label=plot_1_lab, stat="density", kde=True, bins=200)
    plot_2 = sns.histplot(real_v1, color="orange", label=plot_2_lab, stat="density", kde=True, bins=200)
    handles,labels = [],[]
    for h,l in zip(*plot_1.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    if TEST_DIST != 1:
        if len(succ_v1) != 0:
          ax2 = plt.twinx()
          plot_3 = sns.histplot(succ_v1, color="green", label=r"succ #$2^{{{:<12.4f}}}\,\,\,{{\mu_s}}: {:<12.4f}\,\,\,\sigma_s: {:<12.4f}\,\,\,{{\min}}: {:<12.4f}$".format(log2(len(succ_v1)+1e-32), np.mean(succ_v1), np.std(succ_v1), np.min(succ_v1)), stat="density", kde=True, bins=200, ax=ax2, alpha=0.6)
          for h,l in zip(*plot_3.get_legend_handles_labels()):
              handles.append(h)
              labels.append(l)
          plot_3.tick_params(which="both", bottom=True, labelsize=FntSize)
          plot_3.set_ylabel("Densities of samples for succ.", fontsize=FntSize)
        plt.text(0.01, 0.95, r"Comp.: {0:<10}".format(comp), fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.01, 0.90, r"Conf.: $n_{{kg}}: 2^{{{0:<6}}}\,\,\,$ $n_{{b}}: 2^{{{1:<6}}}\,\,\,$ $c_1: {2:<6.1f}\,\,\,$ $c_2: {3:<6.1f}$".format(KG, str(NB)+'+'+str(USE_MULTI_DIFF_NB), cutoff1, cutoff2), fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.01, 0.85, r"Conf.: $n_{{cts}}: 2^{{{0:<6}}}\,\,\,$ $n_{{it}}: 2^{{{1:<6}}}\,\,\,$ $n_{{cand1}}: {{{2:<6}}}\,\,\,$ $n_{{cand2}}: {{{3:<6}}}\,\,\,$ $n_{{byit}}: {{{4:<6}}}$".format(NS, NIT, CAND_KN1, CAND_KN2, BI), fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.01, 0.80, r"Test Done: {0:<6} succ.: {1:<6} succ. rate: {2:<6.4f}".format(int(cnt_done + passed), cnt_succ, cnt_succ/(cnt_done + passed)), fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.01, 0.75, r"Have Real: {0:<6} succ.: {1:<6} succ. rate: {2:<6.4f}".format(int(cnt_done), cnt_succ, cnt_succ/cnt_done), fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.01, 0.70, r"Max time: {0:>8} secs = {1:>4.1f} hours".format(max_time, max_time/3600), fontdict=font, transform=plt.gca().transAxes)
        #plt.text(0.01, 0.65, r"Avg time: {0:>8.0f} secs = {1:>4.1f} hours".format(avg_time, avg_time/3600), fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.01, 0.60 + 0.05, r"Time cplx.: $2^{{{2:>2}}} {{\times}} {0:>8}$ secs = $2^{{{2:>2}}} {{\times}} {1:>4.1f}$ hours ({3:4})".format(max_time, max_time/3600, KG, core), fontdict=font, transform=plt.gca().transAxes)
        if core == 'GPU':
            plt.text(0.01, 0.55 + 0.05, r"Time cplx.: $2^{{{0:<6.2f}+r}}$ Encs ($2^{{{1:}}}$ Encs/sec)".format(log2(max_time + 1e-32)+EQSP+KG,EQSP), fontdict=font, transform=plt.gca().transAxes)
        else:
            plt.text(0.01, 0.55 + 0.05, r"Time cplx.: $2^{{{0:<6.2f}}}$ Encs ($2^{{{1:}}}$ Encs/sec)".format(log2(max_time + 1e-32)+EQSP+KG,EQSP), fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.01, 0.50 + 0.05, r"Data cplx.: $2^{{{0:<6.2f}}}$ CPs".format(DC), fontdict=font, transform=plt.gca().transAxes)
        plt.axvline(x=cutoff1, color='darkgreen', linestyle='--')
    if TEST_DIST == 1:
        plt.text(0.01, 0.96, r"{0:<10} $\,n_{{b}}: 2^{{{1:<6}}}\,\,\,$ $n_{{cand}}: {{{2:<6}}}\,\,\,$ $n_{{byit}}: {{{3:<6}}}$".format(comp, NB, CAND_KN1, BI), fontdict=font, transform=plt.gca().transAxes)
        #plt.text(0.01, 0.90, r"Conf.: $n_{{b}}: 2^{{{0:<6}}}\,\,\,$ $n_{{cand}}: {{{1:<6}}}\,\,\,$ $n_{{byit}}: {{{2:<6}}}$".format(NB, CAND_KN, BI), fontdict=font, transform=plt.gca().transAxes)
    #best_dist_rand_v1, best_p_rand_v1, params_rand_v1 = get_best_distribution(rand_v1)
    #best_dist_real_v1, best_p_real_v1, params_real_v1 = get_best_distribution(real_v1)
    params_rand_v1 = st.genlogistic.fit(rand_v1)
    params_real_v1 = st.genlogistic.fit(real_v1)
    if TEST_DIST == 1:
        rand_v1_rv = st.genlogistic(params_rand_v1[0], params_rand_v1[1], params_rand_v1[2])
        rand_v1_x = np.linspace(st.genlogistic.ppf(0.001, params_rand_v1[0], params_rand_v1[1], params_rand_v1[2]), st.genlogistic.ppf(0.999, params_rand_v1[0], params_rand_v1[1], params_rand_v1[2]), 100)
        plot_4, = plt.plot(rand_v1_x, rand_v1_rv.pdf(rand_v1_x), '--', color='purple', lw=2) #'k-', 
        handles.append(plot_4)
        labels.append(r'rand fit genlogistic ppf: $c = {0:<0.4f}, loc = {1:<0.4f}, scale = {2:<0.4f}$'.format(params_rand_v1[0], params_rand_v1[1], params_rand_v1[2]))
        real_v1_rv = st.genlogistic(params_real_v1[0], params_real_v1[1], params_real_v1[2])
        real_v1_x = np.linspace(st.genlogistic.ppf(0.001, params_real_v1[0], params_real_v1[1], params_real_v1[2]), st.genlogistic.ppf(0.999, params_real_v1[0], params_real_v1[1], params_real_v1[2]), 100)
        plot_5, = plt.plot(real_v1_x, real_v1_rv.pdf(real_v1_x), '--', color='red', lw=2) #'k-', 
        handles.append(plot_5)
        labels.append(r'real fit genlogistic ppf: $c = {0:<0.4f}, loc = {1:<0.4f}, scale = {2:<0.4f}$'.format(params_real_v1[0], params_real_v1[1], params_real_v1[2]))
        rand_v1_res = (stats.relfreq(rand_v1, numbins=500, defaultreallimits=(real_v1_mean, real_v1_max))).frequency
        real_v1_res = (stats.relfreq(real_v1, numbins=500, defaultreallimits=(real_v1_mean, real_v1_max))).frequency
        rand_v1_res[rand_v1_res == 0.0] =  1e-48
        real_v1_res[real_v1_res == 0.0] =  1e-48
        rand_real_kld_v1 = kl_divergence(rand_v1_res, real_v1_res)
        real_rand_kld_v1 = kl_divergence(real_v1_res, rand_v1_res)
        jsd_v1 = (rand_real_kld_v1 + real_rand_kld_v1)/2
        plt.text(0.50, 0.70, "Statistical distances in range [$\mu_r$:" + "{:<10.4f}".format(real_v1_mean) + r'${{\max}}_r:{:0.4f}$'.format(real_v1_max) + "]", fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.70, 0.65, r"KL(rand || real): {0:<.4f} bits".format(rand_real_kld_v1), fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.70, 0.60, r"KL(real || rand): {0:<.4f} bits".format(real_rand_kld_v1), fontdict=font, transform=plt.gca().transAxes)
        plt.text(0.70, 0.55, r"JS(real || rand): {0:<.4f} bits".format(jsd_v1), fontdict=font, transform=plt.gca().transAxes)
    #plt.text(-0.5, 0.80, r"rand_v1: Best fitting dist.: {0:<} p value: {1:<}  Params: {2:<}".format(str(best_dist_rand_v1), str(best_p_rand_v1), str(params_rand_v1)), fontdict=font, transform=plt.gca().transAxes)
    #plt.text(-0.5, 0.75, r"real_v1: Best fitting dist.: {0:<} p value: {1:<}  Params: {2:<}".format(str(best_dist_real_v1), str(best_p_real_v1), str(params_real_v1)), fontdict=font, transform=plt.gca().transAxes)
    plt.legend(handles,labels, fontsize=FntSize);
    plot_1.tick_params(which="both", bottom=True, labelsize=FntSize)
    plot_1.xaxis.set_major_locator(MultipleLocator(5.0))
    plot_1.xaxis.set_minor_locator(MultipleLocator(1.0))
    plot_1.xaxis.grid(True, which='minor', color='lightgrey', alpha=0.5)
    plot_1.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    plot_1.set_ylabel("Densities of samples for rand and real", fontsize=FntSize)
    plot_1.set_xlabel(r"${{v_1}}_{{\mathrm{{max}}}}:= {{\max}}(\{{{v_1}}_i \mid {{v_1}}_i {{\in}} L_1\})$", fontsize=FntSize)
    plt.savefig(wdir + 'rand_vs_real_v1.pdf')
    plt.clf()
    #
    if TEST_DIST != 1:
        plot_1 = sns.histplot(rand_v2, color="dodgerblue", label="rand #$2^{" + "{:8.4f}".format(log2(len(rand_v2)+1e-32)) + "}$ $\mu_w$:" + "{:8.4f}".format(rand_v2_mean) + " $\sigma_w$:" + "{:8.4f}".format(rand_v2_std) + " $qct_w$:" + "{:8.4f}".format(cutoff_v2_rand_rate), stat="density", kde=True, bins=200)
        plot_2 = sns.histplot(real_v2, color="orange", label="real #$2^{" + "{:8.4f}".format(log2(len(real_v2)+1e-32)) + "}$ $\mu_r$:" + "{:8.4f}".format(real_v2_mean) + " $\sigma_r$:" + "{:8.4f}".format(real_v2_std) + " $qct_r$:" + "{:8.4f}".format(cutoff_v2_real_rate), stat="density", kde=True, bins=200)
        handles,labels = [],[]
        for h,l in zip(*plot_1.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
        if TEST_DIST != 1:
            if len(succ_v2) != 0:
              ax2 = plt.twinx()
              plot_3 = sns.histplot(succ_v2, color="green", label=r"succ #$2^{{{:<12.4f}}}\,\,\,{{\mu_s}}: {:<12.4f}\,\,\,\sigma_s: {:<12.4f}\,\,\,{{\min}}: {:<12.4f}$".format(log2(len(succ_v2)+1e-32), np.mean(succ_v2), np.std(succ_v2), np.min(succ_v2)), stat="density", kde=True, bins=200, ax=ax2, alpha=0.6)
              for h,l in zip(*plot_3.get_legend_handles_labels()):
                  handles.append(h)
                  labels.append(l)
              plot_3.tick_params(which="both", bottom=True, labelsize=FntSize)
              plot_3.set_ylabel("Densities of samples for succ.", fontsize=FntSize)
            plt.axvline(x=cutoff2, color='darkgreen', linestyle='--')
        plt.legend(handles, labels, loc='upper right', fontsize=FntSize);
        plot_1.tick_params(which="both", bottom=True, labelsize=FntSize)
        plot_1.xaxis.set_major_locator(MaxNLocator(nbins=25))
        #plot_.xaxis.set_minor_locator(MaxNLocator(nbins=50))
        plot_1.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
        plot_1.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
        plot_1.set_ylabel("Densities of samples for rand and real", fontsize=FntSize)
        plot_1.set_xlabel(r"${{v_2}}_{{\mathrm{{max}}}}:= {{\max}}(\{{{v_2}}_i \mid {{v_2}}_i {{\in}} L_2\})$", fontsize=FntSize)
        plt.savefig(wdir + 'rand_vs_real_v2.pdf')
        plt.clf()
        #
        #bins_1 = 100
        bins_2 = 100
        #if np.mean(fail_used_it) == np.min(fail_used_it):
        #    bins_1 = 1
        if np.mean(succ_used_it) == np.min(succ_used_it):
            bins_2 = 1
        #plot_1 = sns.histplot(fail_used_it, color="red", label="fail: #$" + "{:8.4f}".format(len(fail_used_it)) + "$ $\mu_a$:" + "{:8.4f}".format(np.mean(fail_used_it)) + " $\max$:" + "{:8.4f}".format(np.max(fail_used_it)), stat="count", bins=bins_1)
        plot_1 = sns.histplot(succ_used_it, color="green", label="succ: #$" + "{:8.4f}".format(len(succ_used_it)) + "$ $\mu_a$:" + "{:8.4f}".format(np.mean(succ_used_it)) + " $\max$:" + "{:8.4f}".format(np.max(succ_used_it)), stat="count",     bins=bins_2)
        handles,labels = [],[]
        for h,l in zip(*plot_1.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
        plot_1.tick_params(which="both", bottom=True, labelsize=FntSize)
        plot_1.xaxis.set_major_locator(MaxNLocator(nbins=25))
        plot_1.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
        plot_1.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
        plt.legend(handles, labels, loc='upper right', fontsize=FntSize);
        plt.xlabel("$used$", fontsize=FntSize)
        plt.ylabel("Densities of samples", fontsize=FntSize)
        plt.savefig(wdir + 'used_it.pdf')
        plt.clf()
        plot_ = sns.histplot(diff_hw, color="green", label="Total # {:<8}".format(len(diff_hw)), stat="density", kde=True, binrange=(0, 32), discrete=True, bins=33)
        plt.legend(loc='upper right', fontsize=FntSize);
        plot_.tick_params(which="both", bottom=True, labelsize=FntSize)
        plot_.xaxis.set_major_locator(MultipleLocator(1))
        plot_.set_xlim(-1, 33)
        plot_.yaxis.set_major_locator(MultipleLocator(0.05))
        plot_.yaxis.set_minor_locator(MultipleLocator(0.01))
        plot_.yaxis.grid(True, which='minor', color='lightgrey', alpha=0.4)
        plot_.yaxis.grid(True, which='major', color='lightgrey', alpha=0.6)
        plt.xlabel(r"Hamming distance between $gk_{-1} || gk_{-2}$ and $k_{-1} || k_{-2}$", fontsize=FntSize)
        plt.ylabel("Densities of samples", fontsize=FntSize)
        plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
        plt.savefig(wdir + 'diff_hw.pdf')
        plt.clf()
        #
        plt.axvline(x=cutoff1, color='green', linestyle='--')
        plot_ = sns.histplot(all_v1, color="dodgerblue", label="#$2^{" + "{:8.4f}".format(log2(len(all_v1)+1e-32)) + "}$ $\mu_a$:" + "{:8.4f}".format(all_v1_mean) + " $\sigma_a$:" + "{:8.4f}".format(all_v1_std) + " $qct_a$:" + "{:8.4f}".format(cutoff_v1_rate), stat="density", kde=True, bins=200)
        plt.legend(loc='upper right', fontsize=FntSize);
        plot_.tick_params(which="both", bottom=True, labelsize=FntSize)
        plot_.xaxis.set_major_locator(MultipleLocator(5.0))
        plot_.xaxis.set_minor_locator(MultipleLocator(1.0))
        plot_.xaxis.grid(True, which='minor', color='lightgrey', alpha=0.5)
        plot_.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
        plt.xlabel(r"${{v_1}}_{{\mathrm{{max}}}}:= {{\max}}(\{{{v_1}}_i \mid {{v_1}}_i {{\in}} L_1\})$", fontsize=FntSize)
        plt.ylabel("Densities of samples", fontsize=FntSize)
        plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
        plt.savefig(wdir + 'all_v1.pdf')
        plt.clf()
        #
        plt.axvline(x=cutoff2, color='green', linestyle='--')
        plot_ = sns.histplot(all_v2, color="dodgerblue", label="#$2^{" + "{:8.4f}".format(log2(len(all_v2)+1e-32)) + "}$ $\mu_a$:" + "{:8.4f}".format(all_v2_mean) + " $\sigma_a$:" + "{:8.4f}".format(all_v2_std) + " $qct_a$:" + "{:8.4f}".format(cutoff_v1_rate), stat="density", kde=True, bins=200)
        plt.legend(loc='upper right', fontsize=FntSize);
        plot_.tick_params(which="both", bottom=True, labelsize=FntSize)
        plot_1.xaxis.set_major_locator(MaxNLocator(nbins=25))
        #plot_.xaxis.set_minor_locator(MaxNLocator(nbins=50))
        plot_.xaxis.grid(True, which='minor', color='lightgrey', alpha=0.5)
        plot_.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
        plt.xlabel(r"${{v_1}}_{{\mathrm{{max}}}}:= {{\max}}(\{{{v_1}}_i \mid {{v_1}}_i {{\in}} L_1\})$", fontsize=FntSize)
        plt.ylabel("Densities of samples", fontsize=FntSize)
        plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
        plt.savefig(wdir + 'all_v2.pdf')
        plt.clf()
        #
        plot_ = sns.histplot(cutoff_v1, color="dodgerblue", label="#$2^{" + "{:8.4f}".format(log2(len(cutoff_v1)+1e-32)) + "}$", stat="density", kde=True, bins=200)
        plt.legend();
        plot_.tick_params(which="both", bottom=True, labelsize=FntSize)
        plot_.xaxis.set_major_locator(MultipleLocator(5.0))
        plot_.xaxis.set_minor_locator(MultipleLocator(1.0))
        plot_.xaxis.grid(True, which='minor', color='lightgrey', alpha=0.5)
        plot_.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
        plt.xlabel(r"${{v_1}}_{{\mathrm{{max}}}}:= {{\max}}(\{{{v_1}}_i \mid {{v_1}}_i {{\in}} L_1\})$", fontsize=FntSize)
        plt.ylabel("Densities of samples", fontsize=FntSize)
        plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
        plt.savefig(wdir + 'cutoff_v1.pdf')
        plt.clf()
    #
    hw_v1_0_l = len(hw_v1[0])
    hw_v1_1_l = len(hw_v1[1])
    hw_v1_2_l = len(hw_v1[2])
    hw_v1_3_l = len(hw_v1[3])
    hw_v1_0_p = hw_v1_0_l / real_v1_len
    hw_v1_1_p = hw_v1_1_l / real_v1_len
    hw_v1_2_p = hw_v1_2_l / real_v1_len
    hw_v1_3_p = hw_v1_3_l / real_v1_len

    if hw_v1_0_l == 0:
        hw_v1_0_label = "hw 0: " + " pr. $2^{" + "{:6}".format('none') + "}$" + "  #$2^{" + "{:6}".format('none') + "}$ $\mu_0$:" + "{:6}".format('none') + " $\sigma_0$:" + "{:6}".format('none') + " min$_0$:" + "{:6}".format('none') + " median$_0$:" + "{:6}".format('none') + " max$_0$:" + "{:6}".format('none')
    else:
        hw_v1_0_label = "hw 0: " + " pr. $2^{" + "{:8.4f}".format(log2(hw_v1_0_p)) + "}$" + "  #$2^{" + "{:8.4f}".format(log2(hw_v1_0_l)) + "}$ $\mu_0$:" + "{:8.4f}".format(np.mean(hw_v1[0])) + " $\sigma_0$:" + "{:8.4f}".format(np.std(hw_v1[0])) + " min$_0$:" + "{:8.4f}".format(np.min(hw_v1[0])) + " median$_0$:" + "{:8.4f}".format(np.median(hw_v1[0])) + " max$_0$:" + "{:8.4f}".format(np.max(hw_v1[0]))
    if hw_v1_1_l == 0:
        hw_v1_1_label = "hw 1: " + " pr. $2^{" + "{:6}".format('none') + "}$" + "  #$2^{" + "{:6}".format('none') + "}$ $\mu_1$:" + "{:6}".format('none') + " $\sigma_1$:" + "{:6}".format('none') + " min$_1$:" + "{:6}".format('none') + " median$_1$:" + "{:6}".format('none') + " max$_1$:" + "{:6}".format('none')
    else:
        hw_v1_1_label = "hw 1: " + " pr. $2^{" + "{:8.4f}".format(log2(hw_v1_1_p)) + "}$" + "  #$2^{" + "{:8.4f}".format(log2(hw_v1_1_l)) + "}$ $\mu_1$:" + "{:8.4f}".format(np.mean(hw_v1[1])) + " $\sigma_1$:" + "{:8.4f}".format(np.std(hw_v1[1])) + " min$_1$:" + "{:8.4f}".format(np.min(hw_v1[1])) + " median$_1$:" + "{:8.4f}".format(np.median(hw_v1[1])) + " max$_1$:" + "{:8.4f}".format(np.max(hw_v1[1]))
    if hw_v1_2_l == 0:
        hw_v1_2_label = "hw 2: " + " pr. $2^{" + "{:6}".format('none') + "}$" + "  #$2^{" + "{:6}".format('none') + "}$ $\mu_2$:" + "{:6}".format('none') + " $\sigma_2$:" + "{:6}".format('none') + " min$_2$:" + "{:6}".format('none') + " median$_2$:" + "{:6}".format('none') + " max$_2$:" + "{:6}".format('none')
    else:
        hw_v1_2_label = "hw 2: " + " pr. $2^{" + "{:8.4f}".format(log2(hw_v1_2_p)) + "}$" + "  #$2^{" + "{:8.4f}".format(log2(hw_v1_2_l)) + "}$ $\mu_2$:" + "{:8.4f}".format(np.mean(hw_v1[2])) + " $\sigma_2$:" + "{:8.4f}".format(np.std(hw_v1[2])) + " min$_2$:" + "{:8.4f}".format(np.min(hw_v1[2])) + " median$_2$:" + "{:8.4f}".format(np.median(hw_v1[2])) + " max$_2$:" + "{:8.4f}".format(np.max(hw_v1[2]))
    if hw_v1_3_l == 0:
        hw_v1_3_label = "hw 3: " + " pr. $2^{" + "{:6}".format('none') + "}$" + "  #$2^{" + "{:6}".format('none') + "}$ $\mu_3$:" + "{:6}".format('none') + " $\sigma_3$:" + "{:6}".format('none') + " min$_3$:" + "{:6}".format('none') + " median$_3$:" + "{:6}".format('none') + " max$_3$:" + "{:6}".format('none')
    else:
        hw_v1_3_label = "hw 3: " + " pr. $2^{" + "{:8.4f}".format(log2(hw_v1_3_p)) + "}$" + "  #$2^{" + "{:8.4f}".format(log2(hw_v1_3_l)) + "}$ $\mu_3$:" + "{:8.4f}".format(np.mean(hw_v1[3])) + " $\sigma_3$:" + "{:8.4f}".format(np.std(hw_v1[3])) + " min$_3$:" + "{:8.4f}".format(np.min(hw_v1[3])) + " median$_3$:" + "{:8.4f}".format(np.median(hw_v1[3])) + " max$_3$:" + "{:8.4f}".format(np.max(hw_v1[3]))
    plot_0 = sns.histplot(hw_v1[0], color="green",  label=hw_v1_0_label, stat="density", kde=True, bins=200, alpha = 0.6, zorder = 4-0)
    plot_1 = sns.histplot(hw_v1[1], color="blue",   label=hw_v1_1_label, stat="density", kde=True, bins=200, alpha = 0.6, zorder = 4-1)
    plot_2 = sns.histplot(hw_v1[2], color="orange", label=hw_v1_2_label, stat="density", kde=True, bins=200, alpha = 0.6, zorder = 4-2)
    plot_3 = sns.histplot(hw_v1[3], color="gray",   label=hw_v1_3_label, stat="density", kde=True, bins=200, alpha = 0.6, zorder = 4-3)
    handles,labels = [],[]
    for h,l in zip(*plot_0.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    plt.legend(handles,labels,loc='upper right');
    plot_0.tick_params(which="both", bottom=True, labelsize=FntSize)
    plot_0.xaxis.set_major_locator(MaxNLocator(nbins=25))
    #plot_.xaxis.set_minor_locator(MaxNLocator(nbins=50))
    plot_0.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    plot_0.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    plot_0.set_ylabel("Densities of samples", fontsize=FntSize)
    plot_0.set_xlabel(r"${{v_1}}_{{\mathrm{{max}}}}:= {{\max}}(\{{{v_1}}_i \mid {{v_1}}_i {{\in}} L_1\})$", fontsize=FntSize)
    #plt.ylabel("Densities of samples", fontsize=FntSize)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
    plt.savefig(wdir + 'diff_k1_hw_0_3_v1.pdf')
    plt.clf()
    #
    fig, ax = plt.subplots()
    beta_step = 0.01
    beta_wrong = np.array([i for i in np.arange(0.03, 0.60, beta_step)])
    beta_real = np.array([i for i in np.arange(0.03, 0.60, beta_step)])
    ct_rand_norm = stats.genlogistic.ppf(1.0 - beta_wrong, params_rand_v1[0], params_rand_v1[1], params_rand_v1[2]) #stats.norm.ppf(1.0 - beta_wrong, rand_v1_mean, rand_v1_std) 
    ct_real_norm = stats.genlogistic.ppf(1.0 - beta_real, params_real_v1[0], params_real_v1[1], params_real_v1[2]) #stats.norm.ppf(1.0 - beta_real, real_v1_mean, real_v1_std) 
    ct_rand_exp = np.quantile(rand_v1, 1.0 - beta_wrong) 
    ct_real_exp = np.quantile(real_v1, 1.0 - beta_real) 
    ct_rand_norm_plt, = plt.plot(ct_rand_norm, beta_wrong, '--', label=r'fit genlogistic $ct_w$', color='purple')
    ct_rand_exp_plt,  = plt.plot(ct_rand_exp,  beta_wrong, label=r'exper  $ct_w$', color='dodgerblue')
    ct_real_norm_plt, = plt.plot(ct_real_norm, beta_real, '--', label=r'fit genlogistic $ct_r$', color='red')
    ct_real_exp_plt,  = plt.plot(ct_real_exp,  beta_real,  label=r'exper  $ct_r$', color='orange')
    if TEST_DIST != 1:
        p_rand_exp = len(rand_v1[rand_v1>cutoff1])/rand_v1_len
        p_real_exp = len(real_v1[real_v1>cutoff1])/real_v1_len
        plt.plot([cutoff1], [p_rand_exp], marker='*')
        x,y = cutoff1, p_rand_exp
        arrowprops={'arrowstyle': '-', 'ls':'dotted', 'color':'green'}
        plt.annotate(str(x), xy=(x,y), xytext=(x, 0), 
                     textcoords=plt.gca().get_xaxis_transform(),
                     arrowprops=arrowprops,
                     va='top', ha='center')
        plt.annotate('{:<0.2f}'.format(y), xy=(x,y), xytext=(0, y), 
                     textcoords=plt.gca().get_yaxis_transform(),
                     arrowprops=arrowprops,
                     va='center', ha='left')
        plt.plot([cutoff1], [p_real_exp], marker='*')
        x,y = cutoff1, p_real_exp
        arrowprops={'arrowstyle': '-', 'ls':'dotted', 'color':'green'}
        plt.annotate(str(x), xy=(x,y), xytext=(x, 0), 
                     textcoords=plt.gca().get_xaxis_transform(),
                     arrowprops=arrowprops,
                     va='top', ha='center')
        plt.annotate('{:<0.2f}'.format(y), xy=(x,y), xytext=(0, y), 
                     textcoords=plt.gca().get_yaxis_transform(),
                     arrowprops=arrowprops,
                     va='center', ha='left')
    handles,labels = [],[]
    handles.append(ct_rand_norm_plt)
    handles.append(ct_rand_exp_plt)
    handles.append(ct_real_norm_plt)
    handles.append(ct_real_exp_plt)
    labels.append(r'$ct_w$: fit genlogistic dist')
    labels.append(r'$ct_w$: experimental')
    labels.append(r'$ct_r$: fit genlogistic dist')
    labels.append(r'$ct_r$: experimental')
    ax.tick_params(which="both", bottom=True, labelsize=FntSize2 - 2)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))
    ax.xaxis.set_major_locator(MultipleLocator(2.00))
    ax.xaxis.set_minor_locator(MultipleLocator(0.50))
    ax.xaxis.grid(True, which='minor', color='lightgrey', alpha=0.4)
    ax.yaxis.grid(True, which='minor', color='lightgrey', alpha=0.4)
    ax.xaxis.grid(True, which='major', color='grey', alpha=0.6)
    ax.yaxis.grid(True, which='major', color='grey', alpha=0.6)
    plt.legend(handles,labels, fontsize=FntSize2);
    plt.xlabel("cutoff", fontsize=FntSize2)
    plt.ylabel("Percentage", fontsize=FntSize2)
    plt.savefig(wdir + 'rand_vs_real_v1_quantile.pdf')
    plt.clf()


if __name__ == '__main__':
    set_conf()
    pool = mp.Pool(PN)
    idx_range = range(0, PN)
    calls = pool.map_async(test, idx_range)
    results = calls.get()
    pool.close()
    pool.join()
    fn_first = results[0]
    proc_all_fn = '0_all_' + fn_first[:-5] + '.txt'
    with open(proc_all_fn, 'w') as outfile:
        for names in results:
            with open(names) as infile:
                outfile.write(infile.read())
            outfile.write("\n")
    draw_dist(proc_all_fn)
