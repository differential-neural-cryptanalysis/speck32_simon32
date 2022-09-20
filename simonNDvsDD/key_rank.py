# !/bin/bash

from os import urandom
import numpy as np
import sys
from math import *
from time import time

from tensorflow.keras.models import load_model

import numpy as np
import simon as sp

from random import randint

TR = 1 + 8 + 2

zmin = 1e-64

#load distinguishers
#json_file = open('single_block_resnet.json','r');
#json_model = json_file.read();
#net7 = model_from_json(json_model);
#net7.load_weights('net7_small.h5');

def key_rank_one_round(nr, net, n_blocks=1, diff=(0x0,0x0040)):
  pt0a = np.frombuffer(urandom(2*n_blocks),dtype=np.uint16).reshape(n_blocks,-1);
  pt1a = np.frombuffer(urandom(2*n_blocks),dtype=np.uint16).reshape(n_blocks,-1);
  pt0b, pt1b = pt0a ^ diff[0], pt1a ^ diff[1];
  pt0a, pt1a = sp.dec_one_round((pt0a, pt1a),0);
  pt0b, pt1b = sp.dec_one_round((pt0b, pt1b),0);
  key = np.frombuffer(urandom(8),dtype=np.uint16);
  ks = sp.expand_key(key, nr); k1 = ks[nr-1];
  ct0a, ct1a = sp.encrypt((pt0a, pt1a),ks);
  ct0b, ct1b = sp.encrypt((pt0b, pt1b),ks);
  trial_keys = np.arange(2**16);
  c0a, c1a = sp.dec_one_round((ct0a, ct1a),trial_keys);
  c0b, c1b = sp.dec_one_round((ct0b, ct1b),trial_keys);
  c0a, c1a = sp.dec_one_round((c0a, c1a),0);
  c0b, c1b = sp.dec_one_round((c0b, c1b),0);
  #c1a = np.tile(c1a,2**16); c1b = np.tile(c1b, 2**16);
  #the next two lines are the only bits of this function that change
  #if instead of a neural network the difference distribution table is used for inference
  #in particular, in this case, conversion to network input is replaced by calculation of trial decryption differences
  #Z is then calculated simply by looking up the relevant transition probabilities in the ddt
  #instead of a neural net, the function then expects as second input a table of size 2**32
  c0af = c0a.flatten()
  c1af = c1a.flatten()
  c0bf = c0b.flatten()
  c1bf = c1b.flatten()
  X = sp.convert_to_binary([c0af, c0bf, c1af ^ c1bf]);
  Z = net.predict(X,batch_size=10000); Z = Z/(1-Z+zmin); 
  Z = np.log2(Z+zmin); Z = Z.reshape(n_blocks,-1); Z = np.sum(Z,axis=0);
  rank0 = np.sum(Z > Z[k1]); rank1 = np.sum(Z >= Z[k1]);
  return(rank0, rank1);

def stats_key_rank(n, nr, net, n_blocks, diff=(0x0,0x0040)):
  r = np.zeros(n);
  for i in range(n):
    a,b = key_rank_one_round(nr, net, n_blocks=n_blocks, diff=diff);
    r[i] = randint(a,b);
  return(np.median(r), np.mean(r), r);

def test(net, k,n):
  for i in range(k,n):
    a,b,r = stats_key_rank(1000, TR, net,n_blocks=2**i);
    np.save('./data_11r_attack/data_'+str(i)+'.npy',r);
    print(i,a,b,flush=True);

if __name__ == '__main__':
    wdir = './SENet/'
    oldStdout = sys.stdout
    logfile = open(wdir + 'ND_VD_Simon32_8R_1+8+2_key_rank.log', 'a')
    sys.stdout = logfile
    trained_model = wdir + 'ND_VD_Simon32_8R.h5'
    net = load_model(trained_model)
    print(trained_model,flush=True)
    t0 = time()
    test(net, 1, 7);
    t1 = time()
    print('Done, time took:', t1 - t0, '\n', flush=True)
    print('\n',flush=True)
    logfile.close()

