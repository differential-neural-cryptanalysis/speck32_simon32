# !/bin/bash

import pickle
from simon import check_testvector
from simon import *
import simon as sp

from sklearn.linear_model import LinearRegression

from os import urandom
import numpy as np
import sys
from math import *
from time import time
from random import randint
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.stats as st
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

from threading import Thread
PN_log2 = 5
PN = 1 << PN_log2

zmin = 1e-32
BLOCK_SIZE = 2 * sp.WORD_SIZE()
num_rounds = 9

#from tensorflow.keras.models import load_model
from tensorflow.keras.models import load_model
from tensorflow.python.keras import backend as K

allkeys = np.arange(0, 2**sp.WORD_SIZE(), dtype=np.uint16);


def key_average_DD(ct0a, ct1a, ct0b, ct1b, keys, net):
    pt0a, pt1a = sp.dec_one_round((ct0a, ct1a), keys);
    pt0b, pt1b = sp.dec_one_round((ct0b, ct1b), keys);
    pt0a, pt1a = sp.dec_one_round((pt0a, pt1a), 0);
    pt0b, pt1b = sp.dec_one_round((pt0b, pt1b), 0);
    d0, d1 = pt0a ^ pt0b, pt1a ^ pt1b
    X = d1.astype(np.uint32) ^ (d0.astype(np.uint32) << sp.WORD_SIZE());
    Z = net[X];
    Z = Z/(1-Z); v = np.average(Z);
    v = v/(v+1);
    v = np.average(Z);
    v = v / (v + 2**(-BLOCK_SIZE))
    return(v);

def keyAverage_thread_DD(idx, TND, Z, X, net_AD):
    TN = TND // PN
    ti_start = idx * TN
    ti_end = (idx + 1) * TN
    for i in range(ti_start, ti_end):
        Z[i] = key_average_DD(X[0][i], X[1][i], X[2][i], X[3][i], allkeys, net_AD);

def GenScores():
    TDN = 1 << 25
    #
    wdir = './SENet/'
    oldStdout = sys.stdout
    logfile = open(wdir + str(num_rounds) + 'R_combiner_DD_VD_VV_AD.log', 'a')
    sys.stdout = logfile
    #
    trained_model_AD = wdir + 'ddt_40_' + str(num_rounds - 2) + 'rounds.bin'
    net_AD = np.fromfile(trained_model_AD, dtype=np.float64, count=1<<BLOCK_SIZE)
    print(trained_model_AD)
    #
    trained_model_DD = wdir + 'ddt_40_' + str(num_rounds - 1) + 'rounds.bin'
    net_DD = np.fromfile(trained_model_DD, dtype=np.float64, count=1<<BLOCK_SIZE)
    print(trained_model_DD)
    #
    trained_model_VD = wdir + 'ND_VD_Simon32_' + str(num_rounds - 1) + 'R.h5'
    net_VD = load_model(trained_model_VD)
    print(trained_model_VD)
    #
    trained_model_VV = wdir + 'ND_VV_Simon32_' + str(num_rounds) + 'R.h5'
    net_VV = load_model(trained_model_VV)
    print(trained_model_VV)
    #
    X, Y = make_train_data_noconvert(TDN, num_rounds)
    #
    Z_AD = np.zeros(TDN)
    #for i in range(TDN):
    #    Z_AD[i] = key_average_DD(X[0][i], X[1][i], X[2][i], X[3][i], allkeys, net_AD);
    Threads = [Thread(target=keyAverage_thread_DD, args=(ti, TDN, Z_AD, X, net_AD)) for ti in range(PN)]
    for ti in range(PN):
        Threads[ti].start()
    for ti in range(PN):
        Threads[ti].join()
    #
    X_VV = convert_to_binary([X[0], X[1], X[2], X[3]]);
    Z_VV = net_VV.predict(X_VV, batch_size=1<<14);
    if len(Z_VV)==2:
        Z_VV = Z_VV[-1]
    Z_VV = Z_VV.flatten()
    #
    c0a, c1a = sp.dec_one_round((X[0], X[1]),0);
    c0b, c1b = sp.dec_one_round((X[2], X[3]),0);
    #
    X_VD = convert_to_binary([c0a, c0b, c1a ^ c1b]);
    Z_VD = net_VD.predict(X_VD, batch_size=1<<14);
    if len(Z_VD)==2:
        Z_VD = Z_VD[-1]
    Z_VD = Z_VD.flatten()
    #
    d0,d1 = c0a ^ c0b, c1a ^ c1b;
    X_DD = d1.astype(np.uint32) ^ (d0.astype(np.uint32) << sp.WORD_SIZE());
    Z_DD = net_DD[X_DD]
    Z_DD = Z_DD.flatten()
    Z_DD = Z_DD / (Z_DD + 2**(-BLOCK_SIZE))
    #
    np.save(wdir + str(num_rounds) + 'R_Y.npy', Y)
    np.save(wdir + str(num_rounds) + 'R_X0.npy', X[0])
    np.save(wdir + str(num_rounds) + 'R_X1.npy', X[1])
    np.save(wdir + str(num_rounds) + 'R_X2.npy', X[2])
    np.save(wdir + str(num_rounds) + 'R_X3.npy', X[3])
    np.save(wdir + str(num_rounds) + 'R_Z_AD.npy', Z_AD)
    np.save(wdir + str(num_rounds) + 'R_Z_VV.npy', Z_VV)
    np.save(wdir + str(num_rounds) + 'R_Z_VD.npy', Z_VD)
    np.save(wdir + str(num_rounds) + 'R_Z_DD.npy', Z_DD)
    print('\n')
    #
    Zbin = (Z_AD > 0.5);
    diff = Y - Z_AD;
    mses = np.mean(diff*diff);
    n = len(Z_AD); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("AD:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    Zbin = (Z_DD > 0.5);
    diff = Y - Z_DD;
    mses = np.mean(diff*diff);
    n = len(Z_DD); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("DD:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    Zbin = (Z_VD > 0.5);
    diff = Y - Z_VD;
    mses = np.mean(diff*diff);
    n = len(Z_VD); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("VD:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    Zbin = (Z_VV > 0.5);
    diff = Y - Z_VV;
    mses = np.mean(diff*diff);
    n = len(Z_VV); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("VV:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    logfile.close()
    sys.stdout = oldStdout

def combiner_LR():
    wdir = './SENet/'
    Y = np.load(wdir + str(num_rounds) + 'R_Y.npy')
    Z_VV = np.load(wdir + str(num_rounds) + 'R_Z_VV.npy')
    Z_VD = np.load(wdir + str(num_rounds) + 'R_Z_VD.npy')
    Z_DD = np.load(wdir + str(num_rounds) + 'R_Z_DD.npy')
    Z_VV_bin = Z_VV > 0.5
    Z_VD_bin = Z_VD > 0.5
    Z_DD_bin = Z_DD > 0.5
    Z_eq_x = np.where(((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) % 3) == 0)[0]
    Z_ne_x = np.where(((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) % 3) != 0)[0]
    print("len(Z_eq_x): ", len(Z_eq_x))
    print("len(Z_ne_x): ", len(Z_ne_x))
    Y_eq = Y[Z_eq_x]
    Y_ne = Y[Z_ne_x]
    Z_VV_eq = Z_VV[Z_eq_x]
    Z_VV_ne = Z_VV[Z_ne_x]
    Z_DD_eq = Z_DD[Z_eq_x]
    Z_DD_ne = Z_DD[Z_ne_x]
    Z_VD_eq = Z_VD[Z_eq_x]
    Z_VD_ne = Z_VD[Z_ne_x]
    Z_eq = np.concatenate((Z_VV_eq.reshape(-1, 1), Z_VD_eq.reshape(-1, 1), Z_DD_eq.reshape(-1, 1)), axis=1)
    reg_eq = LinearRegression().fit(Z_eq, Y_eq)
    Z_ne = np.concatenate((Z_VV_ne.reshape(-1, 1), Z_VD_ne.reshape(-1, 1), Z_DD_ne.reshape(-1, 1)), axis=1)
    reg_ne = LinearRegression().fit(Z_ne, Y_ne)
    pickle.dump(reg_eq, open(wdir + str(num_rounds) + 'R_LR_eq-skl.pickle.sav', 'wb'))
    pickle.dump(reg_ne, open(wdir + str(num_rounds) + 'R_LR_ne-skl.pickle.sav', 'wb'))
    return reg_eq, reg_ne


def combiner_LR_4():
    wdir = './SENet/'
    Y = np.load(wdir + str(num_rounds) + 'R_Y.npy')
    Z_VV = np.load(wdir + str(num_rounds) + 'R_Z_VV.npy')
    Z_VD = np.load(wdir + str(num_rounds) + 'R_Z_VD.npy')
    Z_DD = np.load(wdir + str(num_rounds) + 'R_Z_DD.npy')
    Z_VV_bin = Z_VV > 0.5
    Z_VD_bin = Z_VD > 0.5
    Z_DD_bin = Z_DD > 0.5
    Z_0_x = np.where((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) == 0)[0]
    Z_1_x = np.where((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) == 1)[0]
    Z_2_x = np.where((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) == 2)[0]
    Z_3_x = np.where((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) == 3)[0]
    print("len(Z_0_x): ", len(Z_0_x))
    print("len(Z_1_x): ", len(Z_1_x))
    print("len(Z_2_x): ", len(Z_2_x))
    print("len(Z_3_x): ", len(Z_3_x))
    print("len(Z_0_x): ", len(Z_0_x))
    print("len(Z_1_x): ", len(Z_1_x))
    print("len(Z_2_x): ", len(Z_2_x))
    print("len(Z_3_x): ", len(Z_3_x))
    Y_0 = Y[Z_0_x]
    Y_1 = Y[Z_1_x]
    Y_2 = Y[Z_2_x]
    Y_3 = Y[Z_3_x]
    Z_VV_0 = Z_VV[Z_0_x]
    Z_VV_1 = Z_VV[Z_1_x]
    Z_VV_2 = Z_VV[Z_2_x]
    Z_VV_3 = Z_VV[Z_3_x]
    Z_VD_0 = Z_VD[Z_0_x]
    Z_VD_1 = Z_VD[Z_1_x]
    Z_VD_2 = Z_VD[Z_2_x]
    Z_VD_3 = Z_VD[Z_3_x]
    Z_DD_0 = Z_DD[Z_0_x]
    Z_DD_1 = Z_DD[Z_1_x]
    Z_DD_2 = Z_DD[Z_2_x]
    Z_DD_3 = Z_DD[Z_3_x]
    Z_0 = np.concatenate((Z_VV_0.reshape(-1, 1), Z_VD_0.reshape(-1, 1), Z_DD_0.reshape(-1, 1)), axis=1); reg_0 = LinearRegression().fit(Z_0, Y_0)
    Z_1 = np.concatenate((Z_VV_1.reshape(-1, 1), Z_VD_1.reshape(-1, 1), Z_DD_1.reshape(-1, 1)), axis=1); reg_1 = LinearRegression().fit(Z_1, Y_1)
    Z_2 = np.concatenate((Z_VV_2.reshape(-1, 1), Z_VD_2.reshape(-1, 1), Z_DD_2.reshape(-1, 1)), axis=1); reg_2 = LinearRegression().fit(Z_2, Y_2)
    Z_3 = np.concatenate((Z_VV_3.reshape(-1, 1), Z_VD_3.reshape(-1, 1), Z_DD_3.reshape(-1, 1)), axis=1); reg_3 = LinearRegression().fit(Z_3, Y_3)
    pickle.dump(reg_0, open(wdir + str(num_rounds) + 'R_LR_0-skl.pickle.sav', 'wb'))
    pickle.dump(reg_1, open(wdir + str(num_rounds) + 'R_LR_1-skl.pickle.sav', 'wb'))
    pickle.dump(reg_2, open(wdir + str(num_rounds) + 'R_LR_2-skl.pickle.sav', 'wb'))
    pickle.dump(reg_3, open(wdir + str(num_rounds) + 'R_LR_3-skl.pickle.sav', 'wb'))
    return reg_0, reg_1, reg_2, reg_3


def print_LR():
    wdir = './SENet/'
    oldStdout = sys.stdout
    logfile = open(wdir + str(num_rounds) + 'R_print_LR.txt', 'a')
    sys.stdout = logfile
    reg_eq = pickle.load(open(wdir + str(num_rounds) + 'R_LR_eq-skl.pickle.sav', 'rb'))
    reg_ne = pickle.load(open(wdir + str(num_rounds) + 'R_LR_ne-skl.pickle.sav', 'rb'))
    print('reg_eq.coef_')
    print(reg_eq.coef_)
    print('\n')
    print('reg_ne.coef_')
    print(reg_ne.coef_)
    #
    logfile.close()
    sys.stdout = oldStdout

def print_LR_4():
    wdir = './SENet/'
    oldStdout = sys.stdout
    logfile = open(wdir + str(num_rounds) + 'R_print_LR_4.txt', 'a')
    sys.stdout = logfile
    reg_0 = pickle.load(open(wdir + str(num_rounds) + 'R_LR_0-skl.pickle.sav', 'rb'))
    reg_1 = pickle.load(open(wdir + str(num_rounds) + 'R_LR_1-skl.pickle.sav', 'rb'))
    reg_2 = pickle.load(open(wdir + str(num_rounds) + 'R_LR_2-skl.pickle.sav', 'rb'))
    reg_3 = pickle.load(open(wdir + str(num_rounds) + 'R_LR_3-skl.pickle.sav', 'rb'))
    print('reg_0.coef_'); print(reg_0.coef_); print('\n')
    print('reg_1.coef_'); print(reg_1.coef_); print('\n')
    print('reg_2.coef_'); print(reg_2.coef_); print('\n')
    print('reg_3.coef_'); print(reg_3.coef_); print('\n')
    #
    logfile.close()
    sys.stdout = oldStdout

def evaluateLRCombiners(reg_eq, reg_ne):
    TDN = 1<<23
    #
    wdir = './SENet/'
    oldStdout = sys.stdout
    logfile = open(wdir + str(num_rounds) + 'R_evaluateLRCombiners.log', 'a')
    sys.stdout = logfile
    #
    trained_model_DD = wdir + 'ddt_40_' + str(num_rounds - 1) + 'rounds.bin'
    net_DD = np.fromfile(trained_model_DD, dtype=np.float64, count=1<<BLOCK_SIZE)
    print(trained_model_DD)
    #
    trained_model_VD = wdir + 'ND_VD_Simon32_' + str(num_rounds - 1) + 'R.h5'
    net_VD = load_model(trained_model_VD)
    print(trained_model_VD)
    #
    trained_model_VV = wdir + 'ND_VV_Simon32_' + str(num_rounds) + 'R.h5'
    net_VV = load_model(trained_model_VV)
    print(trained_model_VV)
    #
    X, Y = make_train_data_noconvert(TDN, num_rounds)
    #
    X_VV = convert_to_binary([X[0], X[1], X[2], X[3]]);
    Z_VV = net_VV.predict(X_VV, batch_size=1<<14);
    if len(Z_VV)==2:
        Z_VV = Z_VV[-1]
    Z_VV = Z_VV.flatten()
    #
    c0a, c1a = sp.dec_one_round((X[0], X[1]),0);
    c0b, c1b = sp.dec_one_round((X[2], X[3]),0);
    #
    X_VD = convert_to_binary([c0a, c0b, c1a ^ c1b]);
    Z_VD = net_VD.predict(X_VD, batch_size=1<<14);
    if len(Z_VD)==2:
        Z_VD = Z_VD[-1]
    Z_VD = Z_VD.flatten()
    #
    d0,d1 = c0a ^ c0b, c1a ^ c1b;
    X_DD = d1.astype(np.uint32) ^ (d0.astype(np.uint32) << sp.WORD_SIZE());
    Z_DD = net_DD[X_DD]
    Z_DD = Z_DD.flatten()
    Z_DD = Z_DD / (Z_DD + 2**(-BLOCK_SIZE))
    np.save(wdir + str(num_rounds) + 'R_evaluate_Y.npy', Y)
    np.save(wdir + str(num_rounds) + 'R_evaluate_Z_VV.npy', Z_VV)
    np.save(wdir + str(num_rounds) + 'R_evaluate_Z_VD.npy', Z_VD)
    np.save(wdir + str(num_rounds) + 'R_evaluate_Z_DD.npy', Z_DD)
    print('\n')
    #
    Zbin = (Z_DD > 0.5);
    diff = Y - Z_DD;
    mses = np.mean(diff*diff);
    n = len(Z_DD); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("DD:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    Zbin = (Z_VD > 0.5);
    diff = Y - Z_VD;
    mses = np.mean(diff*diff);
    n = len(Z_VD); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("VD:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    Zbin = (Z_VV > 0.5);
    diff = Y - Z_VV;
    mses = np.mean(diff*diff);
    n = len(Z_VV); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("VV:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    Z_VV_bin = Z_VV > 0.5
    Z_VD_bin = Z_VD > 0.5
    Z_DD_bin = Z_DD > 0.5
    Z_eq_x = np.where(((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) % 3) == 0)[0]
    Z_ne_x = np.where(((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) % 3) != 0)[0]
    print("len(Z_eq_x): ", len(Z_eq_x))
    print("len(Z_ne_x): ", len(Z_ne_x))
    Y_eq = Y[Z_eq_x]
    Y_ne = Y[Z_ne_x]
    Z_VV_eq = Z_VV[Z_eq_x]
    Z_VV_ne = Z_VV[Z_ne_x]
    Z_DD_eq = Z_DD[Z_eq_x]
    Z_DD_ne = Z_DD[Z_ne_x]
    Z_VD_eq = Z_VD[Z_eq_x]
    Z_VD_ne = Z_VD[Z_ne_x]
    Z_eq = np.concatenate((Z_VV_eq.reshape(-1, 1), Z_VD_eq.reshape(-1, 1), Z_DD_eq.reshape(-1, 1)), axis=1)
    Z_z_eq = reg_eq.predict(Z_eq)
    Z_z_eq = Z_z_eq.flatten()
    Z_ne = np.concatenate((Z_VV_ne.reshape(-1, 1), Z_VD_ne.reshape(-1, 1), Z_DD_ne.reshape(-1, 1)), axis=1)
    Z_z_ne = reg_ne.predict(Z_ne)
    Z_z_ne = Z_z_ne.flatten()
    Z_VV[Z_eq_x] = Z_z_eq
    Z_VV[Z_ne_x] = Z_z_ne
    np.save(wdir + str(num_rounds) + 'R_evaluate_Z_CB.npy', Z_VV)
    #
    Zbin = (Z_VV > 0.5);
    diff = Y - Z_VV;
    mses = np.mean(diff*diff);
    n = len(Z_VV); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("Combiner:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    logfile.close()
    sys.stdout = oldStdout

def evaluateLRCombiners_4(reg_0, reg_1, reg_2, reg_3):
    TDN = 1<<23
    #
    wdir = './SENet/'
    oldStdout = sys.stdout
    logfile = open(wdir + str(num_rounds) + 'R_evaluateLRCombiners_4.log', 'a')
    sys.stdout = logfile
    #
    trained_model_DD = wdir + 'ddt_40_' + str(num_rounds - 1) + 'rounds.bin'
    net_DD = np.fromfile(trained_model_DD, dtype=np.float64, count=1<<BLOCK_SIZE)
    print(trained_model_DD)
    #
    trained_model_VD = wdir + 'ND_VD_Simon32_' + str(num_rounds - 1) + 'R.h5'
    net_VD = load_model(trained_model_VD)
    print(trained_model_VD)
    #
    trained_model_VV = wdir + 'ND_VV_Simon32_' + str(num_rounds) + 'R.h5'
    net_VV = load_model(trained_model_VV)
    print(trained_model_VV)
    #
    X, Y = make_train_data_noconvert(TDN, num_rounds)
    #
    X_VV = convert_to_binary([X[0], X[1], X[2], X[3]]);
    Z_VV = net_VV.predict(X_VV, batch_size=1<<14);
    if len(Z_VV)==2:
        Z_VV = Z_VV[-1]
    Z_VV = Z_VV.flatten()
    #
    c0a, c1a = sp.dec_one_round((X[0], X[1]),0);
    c0b, c1b = sp.dec_one_round((X[2], X[3]),0);
    #
    X_VD = convert_to_binary([c0a, c0b, c1a ^ c1b]);
    Z_VD = net_VD.predict(X_VD, batch_size=1<<14);
    if len(Z_VD)==2:
        Z_VD = Z_VD[-1]
    Z_VD = Z_VD.flatten()
    #
    d0,d1 = c0a ^ c0b, c1a ^ c1b;
    X_DD = d1.astype(np.uint32) ^ (d0.astype(np.uint32) << sp.WORD_SIZE());
    Z_DD = net_DD[X_DD]
    Z_DD = Z_DD.flatten()
    Z_DD = Z_DD / (Z_DD + 2**(-BLOCK_SIZE))
    np.save(wdir + str(num_rounds) + 'R_evaluate_Y_4.npy', Y)
    np.save(wdir + str(num_rounds) + 'R_evaluate_Z_VV_4.npy', Z_VV)
    np.save(wdir + str(num_rounds) + 'R_evaluate_Z_VD_4.npy', Z_VD)
    np.save(wdir + str(num_rounds) + 'R_evaluate_Z_DD_4.npy', Z_DD)
    print('\n')
    #
    Zbin = (Z_DD > 0.5);
    diff = Y - Z_DD;
    mses = np.mean(diff*diff);
    n = len(Z_DD); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("DD:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    Zbin = (Z_VD > 0.5);
    diff = Y - Z_VD;
    mses = np.mean(diff*diff);
    n = len(Z_VD); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("VD:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    Zbin = (Z_VV > 0.5);
    diff = Y - Z_VV;
    mses = np.mean(diff*diff);
    n = len(Z_VV); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("VV:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    Z_VV_bin = Z_VV > 0.5
    Z_VD_bin = Z_VD > 0.5
    Z_DD_bin = Z_DD > 0.5
    Z_0_x = np.where((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) == 0)[0]
    Z_1_x = np.where((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) == 1)[0]
    Z_2_x = np.where((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) == 2)[0]
    Z_3_x = np.where((Z_VV_bin.astype(int) + Z_VD_bin.astype(int) + Z_DD_bin.astype(int)) == 3)[0]
    print("len(Z_0_x): ", len(Z_0_x))
    print("len(Z_1_x): ", len(Z_1_x))
    print("len(Z_2_x): ", len(Z_2_x))
    print("len(Z_3_x): ", len(Z_3_x))
    Z_VV_0 = Z_VV[Z_0_x]
    Z_VV_1 = Z_VV[Z_1_x]
    Z_VV_2 = Z_VV[Z_2_x]
    Z_VV_3 = Z_VV[Z_3_x]
    Z_VD_0 = Z_VD[Z_0_x]
    Z_VD_1 = Z_VD[Z_1_x]
    Z_VD_2 = Z_VD[Z_2_x]
    Z_VD_3 = Z_VD[Z_3_x]
    Z_DD_0 = Z_DD[Z_0_x]
    Z_DD_1 = Z_DD[Z_1_x]
    Z_DD_2 = Z_DD[Z_2_x]
    Z_DD_3 = Z_DD[Z_3_x]
    Z_0 = np.concatenate((Z_VV_0.reshape(-1, 1), Z_VD_0.reshape(-1, 1), Z_DD_0.reshape(-1, 1)), axis=1); z_0 = reg_0.predict(Z_0); z_0.flatten(); Z_VV[Z_0_x] = z_0
    Z_1 = np.concatenate((Z_VV_1.reshape(-1, 1), Z_VD_1.reshape(-1, 1), Z_DD_1.reshape(-1, 1)), axis=1); z_1 = reg_1.predict(Z_1); z_1.flatten(); Z_VV[Z_1_x] = z_1
    Z_2 = np.concatenate((Z_VV_2.reshape(-1, 1), Z_VD_2.reshape(-1, 1), Z_DD_2.reshape(-1, 1)), axis=1); z_2 = reg_2.predict(Z_2); z_2.flatten(); Z_VV[Z_2_x] = z_2
    Z_3 = np.concatenate((Z_VV_3.reshape(-1, 1), Z_VD_3.reshape(-1, 1), Z_DD_3.reshape(-1, 1)), axis=1); z_3 = reg_3.predict(Z_3); z_3.flatten(); Z_VV[Z_3_x] = z_3
    np.save(wdir + str(num_rounds) + 'R_evaluate_Z_CB_4.npy', Z_VV)
    #
    Zbin = (Z_VV > 0.5);
    diff = Y - Z_VV;
    mses = np.mean(diff*diff);
    n = len(Z_VV); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
    accs = np.sum(Zbin == Y) / n;
    tprs = np.sum(Zbin[Y==1]) / n1;
    tnrs = np.sum(Zbin[Y==0] == 0) / n0;
    print("Combiner:")
    print("ACC %.06f    TPR %.06f    TNR %.06f    MSE %.06f\n" % (accs, tprs, tnrs, mses));
    #
    logfile.close()
    sys.stdout = oldStdout


def drawScoreDist():
    font = {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': FntSize,
    }
    wdir = './SENet/'
    Y = np.load(wdir + str(num_rounds) + 'R_Y.npy')
    Z_VV = np.load(wdir + str(num_rounds) + 'R_Z_VV.npy')
    Z_VD = np.load(wdir + str(num_rounds) + 'R_Z_VD.npy')
    Z_DD = np.load(wdir + str(num_rounds) + 'R_Z_DD.npy')
    Z_AD = np.load(wdir + str(num_rounds) + 'R_Z_AD.npy')
    #
    Z_VV_0 = Z_VV[Y == 0]
    Z_VV_1 = Z_VV[Y == 1]
    Z_VD_0 = Z_VD[Y == 0]
    Z_VD_1 = Z_VD[Y == 1]
    Z_DD_0 = Z_DD[Y == 0]
    Z_DD_1 = Z_DD[Y == 1]
    Z_AD_0 = Z_AD[Y == 0]
    Z_AD_1 = Z_AD[Y == 1]
    kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':1})
    plt.figure(figsize=(20,10), dpi= 600)
    Z_VV_0_label = "Y = 0, Z_VV: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VV_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VV_0)) + " min:" + "{:.3f}".format(np.min(Z_VV_0)) + " median:" + "{:.3f}".format(np.median(Z_VV_0)) + " max:" + "{:.3f}".format(np.max(Z_VV_0))
    Z_VD_0_label = "Y = 0, Z_VD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VD_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VD_0)) + " min:" + "{:.3f}".format(np.min(Z_VD_0)) + " median:" + "{:.3f}".format(np.median(Z_VD_0)) + " max:" + "{:.3f}".format(np.max(Z_VD_0))
    Z_DD_0_label = "Y = 0, Z_DD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_DD_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_DD_0)) + " min:" + "{:.3f}".format(np.min(Z_DD_0)) + " median:" + "{:.3f}".format(np.median(Z_DD_0)) + " max:" + "{:.3f}".format(np.max(Z_DD_0))
    Z_AD_0_label = "Y = 0, Z_AD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_AD_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_AD_0)) + " min:" + "{:.3f}".format(np.min(Z_AD_0)) + " median:" + "{:.3f}".format(np.median(Z_AD_0)) + " max:" + "{:.3f}".format(np.max(Z_AD_0))
    plot_0 = sns.histplot(Z_VV_0, color="green",  label=Z_VV_0_label, stat="density", kde=True, bins=100, alpha = 0.3, zorder = 4-0)
    plot_1 = sns.histplot(Z_VD_0, color="blue",   label=Z_VD_0_label, stat="density", kde=True, bins=100, alpha = 0.3, zorder = 4-1)
    plot_2 = sns.histplot(Z_DD_0, color="yellow", label=Z_DD_0_label, stat="density", kde=True, bins=100, alpha = 0.3, zorder = 4-2)
    plot_3 = sns.histplot(Z_AD_0, color="red",    label=Z_AD_0_label, stat="density", kde=True, bins=100, alpha = 0.3, zorder = 4-3)
    handles,labels = [],[]
    for h,l in zip(*plot_0.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    plt.legend(handles,labels,loc='upper right');
    plot_0.tick_params(which="both", bottom=True, labelsize=FntSize)
    plot_0.xaxis.set_major_locator(MaxNLocator(nbins=25))
    #plot_.xaxis.set_minor_locator(MaxNLocator(nbins=50))
    plot_0.yaxis.set_major_locator(MaxNLocator(nbins=25))
    plot_0.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    plot_0.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    plot_0.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    plot_0.set_ylabel("Densities of samples", fontsize=FntSize)
    plot_0.set_xlabel(r"Scores", fontsize=FntSize)
    #plt.ylabel("Densities of samples", fontsize=FntSize)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y0_dist.pdf')
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y0_dist.png')
    plt.clf()
    #
    Z_VV_1_label = "Y = 1, Z_VV: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VV_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VV_1)) + " min:" + "{:.3f}".format(np.min(Z_VV_1)) + " median:" + "{:.3f}".format(np.median(Z_VV_1)) + " max:" + "{:.3f}".format(np.max(Z_VV_1))
    Z_VD_1_label = "Y = 1, Z_VD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VD_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VD_1)) + " min:" + "{:.3f}".format(np.min(Z_VD_1)) + " median:" + "{:.3f}".format(np.median(Z_VD_1)) + " max:" + "{:.3f}".format(np.max(Z_VD_1))
    Z_DD_1_label = "Y = 1, Z_DD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_DD_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_DD_1)) + " min:" + "{:.3f}".format(np.min(Z_DD_1)) + " median:" + "{:.3f}".format(np.median(Z_DD_1)) + " max:" + "{:.3f}".format(np.max(Z_DD_1))
    Z_AD_1_label = "Y = 1, Z_AD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_AD_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_AD_1)) + " min:" + "{:.3f}".format(np.min(Z_AD_1)) + " median:" + "{:.3f}".format(np.median(Z_AD_1)) + " max:" + "{:.3f}".format(np.max(Z_AD_1))
    plot_0 = sns.histplot(Z_VV_1, color="green",  label=Z_VV_1_label, stat="density", kde=True, bins=100, alpha = 0.3, zorder = 4-0)
    plot_1 = sns.histplot(Z_VD_1, color="blue",   label=Z_VD_1_label, stat="density", kde=True, bins=100, alpha = 0.3, zorder = 4-1)
    plot_2 = sns.histplot(Z_DD_1, color="yellow", label=Z_DD_1_label, stat="density", kde=True, bins=100, alpha = 0.3, zorder = 4-2)
    plot_3 = sns.histplot(Z_AD_1, color="red",    label=Z_AD_1_label, stat="density", kde=True, bins=100, alpha = 0.3, zorder = 4-3)
    handles,labels = [],[]
    for h,l in zip(*plot_0.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    plt.legend(handles,labels,loc='upper right');
    plot_0.tick_params(which="both", bottom=True, labelsize=FntSize)
    plot_0.xaxis.set_major_locator(MaxNLocator(nbins=25))
    #plot_.xaxis.set_minor_locator(MaxNLocator(nbins=50))
    plot_0.yaxis.set_major_locator(MaxNLocator(nbins=25))
    plot_0.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    plot_0.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    plot_0.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    plot_0.set_ylabel("Densities of samples", fontsize=FntSize)
    plot_0.set_xlabel(r"Scores", fontsize=FntSize)
    #plt.ylabel("Densities of samples", fontsize=FntSize)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y1_dist.pdf')
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y1_dist.png')
    plt.clf()

def addmarks(aplt, x,y,c):
    for i in range(len(x)):
        #plt.text(x[i], y[i], m[i], ha = 'center')
        aplt.plot([x[i]], [y[i]], color=c, marker='*')

def drawScoreSamples():
    font = {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': FntSize,
    }
    wdir = './SENet/'
    Y = np.load(wdir + str(num_rounds) + 'R_Y.npy')
    Z_VV = np.load(wdir + str(num_rounds) + 'R_Z_VV.npy')
    Z_VD = np.load(wdir + str(num_rounds) + 'R_Z_VD.npy')
    Z_DD = np.load(wdir + str(num_rounds) + 'R_Z_DD.npy')
    Z_AD = np.load(wdir + str(num_rounds) + 'R_Z_AD.npy')
    #
    Samples_N = 1 << 7
    #
    Z_VV_0 = Z_VV[Y == 0]; Zbin_VV_0 = Z_VV_0 > 0.5;
    Z_VV_1 = Z_VV[Y == 1]; Zbin_VV_1 = Z_VV_1 > 0.5;
    Z_VD_0 = Z_VD[Y == 0]; Zbin_VD_0 = Z_VD_0 > 0.5;
    Z_VD_1 = Z_VD[Y == 1]; Zbin_VD_1 = Z_VD_1 > 0.5;
    Z_DD_0 = Z_DD[Y == 0]; Zbin_DD_0 = Z_DD_0 > 0.5;
    Z_DD_1 = Z_DD[Y == 1]; Zbin_DD_1 = Z_DD_1 > 0.5;
    Z_AD_0 = Z_AD[Y == 0]; Zbin_AD_0 = Z_AD_0 > 0.5;
    Z_AD_1 = Z_AD[Y == 1]; Zbin_AD_1 = Z_AD_1 > 0.5;
    #
    Z_VV_0_label = "Y = 0, Z_VV: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VV_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VV_0)) + " min:" + "{:.3f}".format(np.min(Z_VV_0)) + " median:" + "{:.3f}".format(np.median(Z_VV_0)) + " max:" + "{:.3f}".format(np.max(Z_VV_0))
    Z_VD_0_label = "Y = 0, Z_VD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VD_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VD_0)) + " min:" + "{:.3f}".format(np.min(Z_VD_0)) + " median:" + "{:.3f}".format(np.median(Z_VD_0)) + " max:" + "{:.3f}".format(np.max(Z_VD_0))
    Z_DD_0_label = "Y = 0, Z_DD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_DD_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_DD_0)) + " min:" + "{:.3f}".format(np.min(Z_DD_0)) + " median:" + "{:.3f}".format(np.median(Z_DD_0)) + " max:" + "{:.3f}".format(np.max(Z_DD_0))
    Z_AD_0_label = "Y = 0, Z_AD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_AD_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_AD_0)) + " min:" + "{:.3f}".format(np.min(Z_AD_0)) + " median:" + "{:.3f}".format(np.median(Z_AD_0)) + " max:" + "{:.3f}".format(np.max(Z_AD_0))
    Z_VV_1_label = "Y = 1, Z_VV: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VV_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VV_1)) + " min:" + "{:.3f}".format(np.min(Z_VV_1)) + " median:" + "{:.3f}".format(np.median(Z_VV_1)) + " max:" + "{:.3f}".format(np.max(Z_VV_1))
    Z_VD_1_label = "Y = 1, Z_VD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VD_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VD_1)) + " min:" + "{:.3f}".format(np.min(Z_VD_1)) + " median:" + "{:.3f}".format(np.median(Z_VD_1)) + " max:" + "{:.3f}".format(np.max(Z_VD_1))
    Z_DD_1_label = "Y = 1, Z_DD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_DD_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_DD_1)) + " min:" + "{:.3f}".format(np.min(Z_DD_1)) + " median:" + "{:.3f}".format(np.median(Z_DD_1)) + " max:" + "{:.3f}".format(np.max(Z_DD_1))
    Z_AD_1_label = "Y = 1, Z_AD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_AD_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_AD_1)) + " min:" + "{:.3f}".format(np.min(Z_AD_1)) + " median:" + "{:.3f}".format(np.median(Z_AD_1)) + " max:" + "{:.3f}".format(np.max(Z_AD_1))
    #
    Z_VV_0 = Z_VV_0[:Samples_N]; Zbin_VV_0 = Zbin_VV_0[:Samples_N];
    Z_VV_1 = Z_VV_1[:Samples_N]; Zbin_VV_1 = Zbin_VV_1[:Samples_N];
    Z_VD_0 = Z_VD_0[:Samples_N]; Zbin_VD_0 = Zbin_VD_0[:Samples_N];
    Z_VD_1 = Z_VD_1[:Samples_N]; Zbin_VD_1 = Zbin_VD_1[:Samples_N];
    Z_DD_0 = Z_DD_0[:Samples_N]; Zbin_DD_0 = Zbin_DD_0[:Samples_N];
    Z_DD_1 = Z_DD_1[:Samples_N]; Zbin_DD_1 = Zbin_DD_1[:Samples_N];
    Z_AD_0 = Z_AD_0[:Samples_N]; Zbin_AD_0 = Zbin_AD_0[:Samples_N];
    Z_AD_1 = Z_AD_1[:Samples_N]; Zbin_AD_1 = Zbin_AD_1[:Samples_N];
    ##
    x0 = np.linspace(0, Samples_N, Samples_N).reshape(-1, 1)
    kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':1})
    fig = plt.figure(figsize=(20, 10))
    axs = fig.add_subplot(1, 1, 1)
    plt.axhline(y=0.5, color="red")
    plot_0 = axs.plot(x0, Z_VV_0, color="green",  label=Z_VV_0_label, alpha = 0.5, zorder = 4-0); #addmarks(axs, Z_VVw_0_x, Z_VV_0[Z_VVw_0_x], "green");
    plot_1 = axs.plot(x0, Z_VD_0, color="blue",   label=Z_VD_0_label, alpha = 0.5, zorder = 4-1); #addmarks(axs, Z_VDw_0_x, Z_VD_0[Z_VDw_0_x], "blue");
    plot_2 = axs.plot(x0, Z_DD_0, color="yellow", label=Z_DD_0_label, alpha = 0.5, zorder = 4-2); #addmarks(axs, Z_DDw_0_x, Z_DD_0[Z_DDw_0_x], "yellow");
    plot_3 = axs.plot(x0, Z_AD_0, color="red",   label=Z_AD_0_label, alpha = 0.5, zorder = 4-3); #addmarks(axs, Z_ADw_0_x, Z_AD_0[Z_ADw_0_x], "red");
    handles,labels = [],[]
    for h,l in zip(*axs.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    plt.legend(handles,labels,loc='upper right');
    axs.tick_params(which="both", bottom=True, labelsize=FntSize)
    axs.xaxis.set_major_locator(MaxNLocator(nbins=25))
    #plot_.xaxis.set_minor_locator(MaxNLocator(nbins=50))
    axs.yaxis.set_major_locator(MaxNLocator(nbins=25))
    axs.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    axs.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    axs.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    axs.set_ylabel("Scores", fontsize=FntSize)
    axs.set_xlabel("Rand Sample", fontsize=FntSize)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y0_samples%d.pdf' % (Samples_N))
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y0_samples%d.png' % (Samples_N))
    plt.clf()
    plt.close(fig)
    #
    fig = plt.figure(figsize=(20, 10))
    axs = fig.add_subplot(1, 1, 1)
    plt.axhline(y=0.5, color="red")
    plot_0 = axs.plot(x0, Z_VV_1, color="green",  label=Z_VV_1_label, alpha = 0.5, zorder = 4-0); #addmarks(axs, Z_VVw_1_x, Z_VV_1[Z_VVw_1_x], "green");
    plot_1 = axs.plot(x0, Z_VD_1, color="blue",   label=Z_VD_1_label, alpha = 0.5, zorder = 4-1); #addmarks(axs, Z_VDw_1_x, Z_VD_1[Z_VDw_1_x], "blue");
    plot_2 = axs.plot(x0, Z_DD_1, color="yellow", label=Z_DD_1_label, alpha = 0.5, zorder = 4-2); #addmarks(axs, Z_DDw_1_x, Z_DD_1[Z_DDw_1_x], "yellow");
    plot_3 = axs.plot(x0, Z_AD_1, color="red",    label=Z_AD_1_label, alpha = 0.5, zorder = 4-3); #addmarks(axs, Z_ADw_1_x, Z_AD_1[Z_ADw_1_x], "red");
    handles,labels = [],[]
    for h,l in zip(*axs.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    plt.legend(handles,labels,loc='upper right');
    axs.tick_params(which="both", bottom=True, labelsize=FntSize)
    axs.xaxis.set_major_locator(MaxNLocator(nbins=25))
    #plot_.xaxis.set_minor_locator(MaxNLocator(nbins=50))
    axs.yaxis.set_major_locator(MaxNLocator(nbins=25))
    axs.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    axs.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    axs.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    axs.set_ylabel("Scores", fontsize=FntSize)
    axs.set_xlabel("Rand Sample", fontsize=FntSize)
    #plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y1_samples%d.pdf' % (Samples_N))
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y1_samples%d.png' % (Samples_N))
    plt.clf()
    plt.close(fig)

def drawScoreSamples_marker():
    font = {'family': 'serif',
    'color':  'black',
    'weight': 'normal',
    'size': FntSize,
    }
    wdir = './SENet/'
    Y = np.load(wdir + str(num_rounds) + 'R_Y.npy')
    Z_VV = np.load(wdir + str(num_rounds) + 'R_Z_VV.npy')
    Z_VD = np.load(wdir + str(num_rounds) + 'R_Z_VD.npy')
    Z_DD = np.load(wdir + str(num_rounds) + 'R_Z_DD.npy')
    Z_AD = np.load(wdir + str(num_rounds) + 'R_Z_AD.npy')
    #
    Samples_N = 1 << 7
    #
    Z_VV_0 = Z_VV[Y == 0]; Zbin_VV_0 = Z_VV_0 > 0.5;
    Z_VV_1 = Z_VV[Y == 1]; Zbin_VV_1 = Z_VV_1 > 0.5;
    Z_VD_0 = Z_VD[Y == 0]; Zbin_VD_0 = Z_VD_0 > 0.5;
    Z_VD_1 = Z_VD[Y == 1]; Zbin_VD_1 = Z_VD_1 > 0.5;
    Z_DD_0 = Z_DD[Y == 0]; Zbin_DD_0 = Z_DD_0 > 0.5;
    Z_DD_1 = Z_DD[Y == 1]; Zbin_DD_1 = Z_DD_1 > 0.5;
    Z_AD_0 = Z_AD[Y == 0]; Zbin_AD_0 = Z_AD_0 > 0.5;
    Z_AD_1 = Z_AD[Y == 1]; Zbin_AD_1 = Z_AD_1 > 0.5;
    #
    Z_VV_0_label = "Y = 0,  Z_VV: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VV_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VV_0)) + " min:" + "{:.3f}".format(np.min(Z_VV_0)) + " median:" + "{:.3f}".format(np.median(Z_VV_0)) + " max:" + "{:.3f}".format(np.max(Z_VV_0))
    Z_VD_0_label = "Y = 0,  Z_VD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VD_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VD_0)) + " min:" + "{:.3f}".format(np.min(Z_VD_0)) + " median:" + "{:.3f}".format(np.median(Z_VD_0)) + " max:" + "{:.3f}".format(np.max(Z_VD_0))
    Z_DD_0_label = "Y = 0,  Z_DD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_DD_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_DD_0)) + " min:" + "{:.3f}".format(np.min(Z_DD_0)) + " median:" + "{:.3f}".format(np.median(Z_DD_0)) + " max:" + "{:.3f}".format(np.max(Z_DD_0))
    Z_AD_0_label = "Y = 0,  Z_AD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_AD_0)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_AD_0)) + " min:" + "{:.3f}".format(np.min(Z_AD_0)) + " median:" + "{:.3f}".format(np.median(Z_AD_0)) + " max:" + "{:.3f}".format(np.max(Z_AD_0))
    Z_VV_1_label = "Y = 1,  Z_VV: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VV_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VV_1)) + " min:" + "{:.3f}".format(np.min(Z_VV_1)) + " median:" + "{:.3f}".format(np.median(Z_VV_1)) + " max:" + "{:.3f}".format(np.max(Z_VV_1))
    Z_VD_1_label = "Y = 1,  Z_VD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_VD_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_VD_1)) + " min:" + "{:.3f}".format(np.min(Z_VD_1)) + " median:" + "{:.3f}".format(np.median(Z_VD_1)) + " max:" + "{:.3f}".format(np.max(Z_VD_1))
    Z_DD_1_label = "Y = 1,  Z_DD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_DD_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_DD_1)) + " min:" + "{:.3f}".format(np.min(Z_DD_1)) + " median:" + "{:.3f}".format(np.median(Z_DD_1)) + " max:" + "{:.3f}".format(np.max(Z_DD_1))
    Z_AD_1_label = "Y = 1,  Z_AD: " + "$\mu$:" + "{:.3f}".format(np.mean(Z_AD_1)) + " $\sigma$:" + "{:.3f}".format(np.std(Z_AD_1)) + " min:" + "{:.3f}".format(np.min(Z_AD_1)) + " median:" + "{:.3f}".format(np.median(Z_AD_1)) + " max:" + "{:.3f}".format(np.max(Z_AD_1))
    #
    Z_VV_0 = Z_VV_0[:Samples_N]; Zbin_VV_0 = Zbin_VV_0[:Samples_N];
    Z_VV_1 = Z_VV_1[:Samples_N]; Zbin_VV_1 = Zbin_VV_1[:Samples_N];
    Z_VD_0 = Z_VD_0[:Samples_N]; Zbin_VD_0 = Zbin_VD_0[:Samples_N];
    Z_VD_1 = Z_VD_1[:Samples_N]; Zbin_VD_1 = Zbin_VD_1[:Samples_N];
    Z_DD_0 = Z_DD_0[:Samples_N]; Zbin_DD_0 = Zbin_DD_0[:Samples_N];
    Z_DD_1 = Z_DD_1[:Samples_N]; Zbin_DD_1 = Zbin_DD_1[:Samples_N];
    Z_AD_0 = Z_AD_0[:Samples_N]; Zbin_AD_0 = Zbin_AD_0[:Samples_N];
    Z_AD_1 = Z_AD_1[:Samples_N]; Zbin_AD_1 = Zbin_AD_1[:Samples_N];
    # index
    Z_VVw_0 = Zbin_VV_0 & ((1 - Zbin_VD_0) | (1 - Zbin_DD_0) | (1 - Zbin_AD_0)); Z_VVw_1 = (1 - Zbin_VV_1) & (Zbin_VD_1 | Zbin_DD_1 | Zbin_AD_1)
    Z_VDw_0 = Zbin_VD_0 & ((1 - Zbin_VV_0) | (1 - Zbin_DD_0) | (1 - Zbin_AD_0)); Z_VDw_1 = (1 - Zbin_VD_1) & (Zbin_VV_1 | Zbin_DD_1 | Zbin_AD_1)
    Z_DDw_0 = Zbin_DD_0 & ((1 - Zbin_VV_0) | (1 - Zbin_VD_0) | (1 - Zbin_AD_0)); Z_DDw_1 = (1 - Zbin_DD_1) & (Zbin_VV_1 | Zbin_VD_1 | Zbin_AD_1)
    Z_ADw_0 = Zbin_AD_0 & ((1 - Zbin_VV_0) | (1 - Zbin_VD_0) | (1 - Zbin_DD_0)); Z_ADw_1 = (1 - Zbin_AD_1) & (Zbin_VV_1 | Zbin_VD_1 | Zbin_DD_1)
    # index
    Z_VVw_0_x = np.asarray(Z_VVw_0 == 1).nonzero()[0]; Z_VVw_1_x = np.asarray(Z_VVw_1 == 1).nonzero()[0]
    Z_VDw_0_x = np.asarray(Z_VDw_0 == 1).nonzero()[0]; Z_VDw_1_x = np.asarray(Z_VDw_1 == 1).nonzero()[0]
    Z_DDw_0_x = np.asarray(Z_DDw_0 == 1).nonzero()[0]; Z_DDw_1_x = np.asarray(Z_DDw_1 == 1).nonzero()[0]
    Z_ADw_0_x = np.asarray(Z_ADw_0 == 1).nonzero()[0]; Z_ADw_1_x = np.asarray(Z_ADw_1 == 1).nonzero()[0]
    #
    np.savetxt(wdir + str(num_rounds) + 'Z_VVw_0.txt', np.concatenate((Z_VVw_0_x.reshape(-1,1), Z_VV_0[Z_VVw_0_x].reshape(-1,1), Z_VD_0[Z_VVw_0_x].reshape(-1,1), Z_DD_0[Z_VVw_0_x].reshape(-1,1), Z_AD_0[Z_VVw_0_x].reshape(-1,1)), axis=1), fmt='%.4f', delimiter=',')
    np.savetxt(wdir + str(num_rounds) + 'Z_VDw_0.txt', np.concatenate((Z_VDw_0_x.reshape(-1,1), Z_VV_0[Z_VDw_0_x].reshape(-1,1), Z_VD_0[Z_VDw_0_x].reshape(-1,1), Z_DD_0[Z_VDw_0_x].reshape(-1,1), Z_AD_0[Z_VDw_0_x].reshape(-1,1)), axis=1), fmt='%.4f', delimiter=',')
    np.savetxt(wdir + str(num_rounds) + 'Z_DDw_0.txt', np.concatenate((Z_DDw_0_x.reshape(-1,1), Z_VV_0[Z_DDw_0_x].reshape(-1,1), Z_VD_0[Z_DDw_0_x].reshape(-1,1), Z_DD_0[Z_DDw_0_x].reshape(-1,1), Z_AD_0[Z_DDw_0_x].reshape(-1,1)), axis=1), fmt='%.4f', delimiter=',')
    np.savetxt(wdir + str(num_rounds) + 'Z_ADw_0.txt', np.concatenate((Z_ADw_0_x.reshape(-1,1), Z_VV_0[Z_ADw_0_x].reshape(-1,1), Z_VD_0[Z_ADw_0_x].reshape(-1,1), Z_DD_0[Z_ADw_0_x].reshape(-1,1), Z_AD_0[Z_ADw_0_x].reshape(-1,1)), axis=1), fmt='%.4f', delimiter=',')
    #
    np.savetxt(wdir + str(num_rounds) + 'Z_VVw_1.txt', np.concatenate((Z_VVw_1_x.reshape(-1,1), Z_VV_1[Z_VVw_1_x].reshape(-1,1), Z_VD_1[Z_VVw_1_x].reshape(-1,1), Z_DD_1[Z_VVw_1_x].reshape(-1,1), Z_AD_1[Z_VVw_1_x].reshape(-1,1)), axis=1), fmt='%.4f', delimiter=',')
    np.savetxt(wdir + str(num_rounds) + 'Z_VDw_1.txt', np.concatenate((Z_VDw_1_x.reshape(-1,1), Z_VV_1[Z_VDw_1_x].reshape(-1,1), Z_VD_1[Z_VDw_1_x].reshape(-1,1), Z_DD_1[Z_VDw_1_x].reshape(-1,1), Z_AD_1[Z_VDw_1_x].reshape(-1,1)), axis=1), fmt='%.4f', delimiter=',')
    np.savetxt(wdir + str(num_rounds) + 'Z_DDw_1.txt', np.concatenate((Z_DDw_1_x.reshape(-1,1), Z_VV_1[Z_DDw_1_x].reshape(-1,1), Z_VD_1[Z_DDw_1_x].reshape(-1,1), Z_DD_1[Z_DDw_1_x].reshape(-1,1), Z_AD_1[Z_DDw_1_x].reshape(-1,1)), axis=1), fmt='%.4f', delimiter=',')
    np.savetxt(wdir + str(num_rounds) + 'Z_ADw_1.txt', np.concatenate((Z_ADw_1_x.reshape(-1,1), Z_VV_1[Z_ADw_1_x].reshape(-1,1), Z_VD_1[Z_ADw_1_x].reshape(-1,1), Z_DD_1[Z_ADw_1_x].reshape(-1,1), Z_AD_1[Z_ADw_1_x].reshape(-1,1)), axis=1), fmt='%.4f', delimiter=',')
    #
    x0 = np.linspace(0, Samples_N, Samples_N).reshape(-1, 1)
    kwargs = dict(hist_kws={'alpha':.5}, kde_kws={'linewidth':1})
    fig = plt.figure(figsize=(20, 10))
    axs = fig.add_subplot(1, 1, 1)
    plt.axhline(y=0.5, color="red")
    plot_0 = axs.plot(x0, Z_VV_0, color="green",  label=Z_VV_0_label, alpha = 0.5, zorder = 4-0); addmarks(axs, Z_VVw_0_x, Z_VV_0[Z_VVw_0_x], "green");
    plot_1 = axs.plot(x0, Z_VD_0, color="blue",   label=Z_VD_0_label, alpha = 0.5, zorder = 4-1); addmarks(axs, Z_VDw_0_x, Z_VD_0[Z_VDw_0_x], "blue");
    plot_2 = axs.plot(x0, Z_DD_0, color="yellow", label=Z_DD_0_label, alpha = 0.5, zorder = 4-2); addmarks(axs, Z_DDw_0_x, Z_DD_0[Z_DDw_0_x], "yellow");
    plot_3 = axs.plot(x0, Z_AD_0, color="red",    label=Z_AD_0_label, alpha = 0.5, zorder = 4-3); addmarks(axs, Z_ADw_0_x, Z_AD_0[Z_ADw_0_x], "red");
    handles,labels = [],[]
    for h,l in zip(*axs.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    plt.legend(handles,labels,loc='upper right');
    axs.tick_params(which="both", bottom=True, labelsize=FntSize)
    axs.xaxis.set_major_locator(MaxNLocator(nbins=25))
    axs.xaxis.set_minor_locator(MaxNLocator(nbins=50))
    axs.yaxis.set_major_locator(MaxNLocator(nbins=25))
    axs.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    axs.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    axs.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    axs.set_ylabel("Scores", fontsize=FntSize)
    axs.set_xlabel("Rand Sample", fontsize=FntSize)
    plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y0_samples%d.pdf' % (Samples_N))
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y0_samples%d.png' % (Samples_N))
    plt.clf()
    plt.close(fig)
    #
    fig = plt.figure(figsize=(20, 10))
    axs = fig.add_subplot(1, 1, 1)
    plt.axhline(y=0.5, color="red")
    plot_0 = axs.plot(x0, Z_VV_1, color="green",  label=Z_VV_1_label, alpha = 0.5, zorder = 4-0); addmarks(axs, Z_VVw_1_x, Z_VV_1[Z_VVw_1_x], "green");
    plot_1 = axs.plot(x0, Z_VD_1, color="blue",   label=Z_VD_1_label, alpha = 0.5, zorder = 4-1); addmarks(axs, Z_VDw_1_x, Z_VD_1[Z_VDw_1_x], "blue");
    plot_2 = axs.plot(x0, Z_DD_1, color="yellow", label=Z_DD_1_label, alpha = 0.5, zorder = 4-2); addmarks(axs, Z_DDw_1_x, Z_DD_1[Z_DDw_1_x], "yellow");
    plot_3 = axs.plot(x0, Z_AD_1, color="red",    label=Z_AD_1_label, alpha = 0.5, zorder = 4-3); addmarks(axs, Z_ADw_1_x, Z_AD_1[Z_ADw_1_x], "red");
    handles,labels = [],[]
    for h,l in zip(*axs.get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    plt.legend(handles,labels,loc='upper right');
    axs.tick_params(which="both", bottom=True, labelsize=FntSize)
    axs.xaxis.set_major_locator(MaxNLocator(nbins=25))
    axs.xaxis.set_minor_locator(MaxNLocator(nbins=50))
    axs.yaxis.set_major_locator(MaxNLocator(nbins=25))
    axs.yaxis.set_minor_locator(MaxNLocator(nbins=50))
    axs.xaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    axs.yaxis.grid(True, which='major', color='lightgrey', alpha=0.5)
    axs.set_ylabel("Scores", fontsize=FntSize)
    axs.set_xlabel("Rand Sample", fontsize=FntSize)
    #plt.setp(plt.gca().get_legend().get_texts(), fontsize=FntSize)
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y1_samples%d.pdf' % (Samples_N))
    plt.savefig(wdir + str(num_rounds) + 'R_AD_DD_VD_VV_Y1_samples%d.png' % (Samples_N))
    plt.clf()
    plt.close(fig)


def TestNDVVtakesback():
    TDN = 1<<8
    #
    wdir = './ResNet_lr_1e-3_1e-5/'
    oldStdout = sys.stdout
    logfile = open(wdir + str(num_rounds) + 'R_TestNDVVtakesback.log', 'a')
    sys.stdout = logfile
    #
    trained_model_VV = wdir + 'best' + str(num_rounds) + 'depth10.h5'
    net_VV = load_model(trained_model_VV)
    print(trained_model_VV)
    #
    X, Y = make_train_data_noconvert(TDN, num_rounds)
    #
    Y0 = Y[Y == 0]
    Y1 = Y[Y == 1]
    x0 = np.linspace(0, len(Y0), len(Y0)).reshape(-1, 1)
    x1 = np.linspace(0, len(Y1), len(Y1)).reshape(-1, 1)
    #
    fig = plt.figure(figsize=(20, 20))
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212, sharex=ax0)
    #
    for kv in range(1<<sp.WORD_SIZE()):
        X_VV = convert_to_binary([X[0] ^ kv, X[1], X[2] ^ kv, X[3]]);
        Z_VV = net_VV.predict(X_VV, batch_size=1<<14);
        if len(Z_VV)==2:
            Z_VV = Z_VV[-1]
        Z_VV = Z_VV.flatten()
        Z_VV_0 = Z_VV[Y == 0]
        Z_VV_1 = Z_VV[Y == 1]
        #
        ax0.plot(x0, Z_VV_0,  label=str(kv), alpha = 0.3)
        ax1.plot(x1, Z_VV_1,  label=str(kv), alpha = 0.3)
    plt.savefig(wdir + str(num_rounds) + 'R_TestNDVVtakesback.pdf')
    plt.savefig(wdir + str(num_rounds) + 'R_TestNDVVtakesback.png')
    plt.clf()
    plt.close(fig)

def TestNDVVtakesback_withVD():
    TDN = 1<<6
    #
    wdir = './SENet/'
    oldStdout = sys.stdout
    logfile = open(wdir + str(num_rounds) + 'R_TestNDVVtakesback.log', 'a')
    sys.stdout = logfile
    #
    trained_model_VV = wdir + 'ND_VV_Simon32_' + str(num_rounds) + 'R.h5'
    net_VV = load_model(trained_model_VV)
    print(trained_model_VV)
    #
    trained_model_VD = wdir + 'ND_VD_Simon32_' + str(num_rounds - 1) + 'R.h5'
    net_VD = load_model(trained_model_VD)
    print(trained_model_VD)
    #
    X, Y = make_train_data_noconvert(TDN, num_rounds)
    #
    #
    c0a, c1a = sp.dec_one_round((X[0], X[1]),0);
    c0b, c1b = sp.dec_one_round((X[2], X[3]),0);
    #
    X_VD = convert_to_binary([c0a, c0b, c1a ^ c1b]);
    Z_VD = net_VD.predict(X_VD, batch_size=1<<14);
    if len(Z_VD)==2:
        Z_VD = Z_VD[-1]
    Z_VD = Z_VD.flatten()
    Z_VD_0 = Z_VD[Y == 0]
    Z_VD_1 = Z_VD[Y == 1]
    x0 = np.linspace(0, len(Z_VD_0), len(Z_VD_0)).reshape(-1, 1)
    x1 = np.linspace(0, len(Z_VD_1), len(Z_VD_1)).reshape(-1, 1)
    #
    fig = plt.figure(figsize=(20, 20))
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212, sharex=ax0)
    #
    ax0.plot(x0, Z_VD_0, marker='*', alpha = 0.8)
    ax1.plot(x1, Z_VD_1, marker='*', alpha = 0.8)
    for kv in range(1<<16): #range(1<<sp.WORD_SIZE()):
        X_VV = convert_to_binary([X[0] ^ kv, X[1], X[2] ^ kv, X[3]]);
        Z_VV = net_VV.predict(X_VV, batch_size=1<<14);
        if len(Z_VV)==2:
            Z_VV = Z_VV[-1]
        Z_VV = Z_VV.flatten()
        Z_VV_0 = Z_VV[Y == 0]
        Z_VV_1 = Z_VV[Y == 1]
        #
        ax0.plot(x0, Z_VV_0,  label=str(kv), alpha = 0.3)
        ax1.plot(x1, Z_VV_1,  label=str(kv), alpha = 0.3)
    plt.savefig(wdir + str(num_rounds) + 'R_TestNDVVtakesback.pdf')
    plt.savefig(wdir + str(num_rounds) + 'R_TestNDVVtakesback.png')
    plt.clf()
    plt.close(fig)

def TestNDVD_masked():
    TDN = 1<<6
    #
    wdir = './SENet/'
    oldStdout = sys.stdout
    logfile = open(wdir + str(num_rounds) + 'R_TestNDVD_masked.log', 'a')
    sys.stdout = logfile
    #
    trained_model_VD = wdir + 'ND_VD_Simon32_' + str(num_rounds) + 'R.h5'
    net_VD = load_model(trained_model_VD)
    print(trained_model_VD)
    #
    X, Y = make_train_data_noconvert(TDN, num_rounds)
    #
    Y0 = Y[Y == 0]
    Y1 = Y[Y == 1]
    x0 = np.linspace(0, len(Y0), len(Y0)).reshape(-1, 1)
    x1 = np.linspace(0, len(Y1), len(Y1)).reshape(-1, 1)
    #
    fig = plt.figure(figsize=(20, 20))
    ax0 = plt.subplot(211)
    ax1 = plt.subplot(212, sharex=ax0)
    #
    for kv in range(1<<sp.WORD_SIZE()):
        X_VD = convert_to_binary([X[0] ^ kv, X[2] ^ kv, X[1] ^ X[3]]);
        Z_VD = net_VD.predict(X_VD, batch_size=1<<14);
        if len(Z_VD)==2:
            Z_VD = Z_VD[-1]
        Z_VD = Z_VD.flatten()
        Z_VD_0 = Z_VD[Y == 0]
        Z_VD_1 = Z_VD[Y == 1]
        #
        ax0.plot(x0, Z_VD_0,  label=str(kv), alpha = 0.3)
        ax1.plot(x1, Z_VD_1,  label=str(kv), alpha = 0.3)
    plt.savefig(wdir + str(num_rounds) + 'R_TestNDVD_masked.pdf')
    plt.savefig(wdir + str(num_rounds) + 'R_TestNDVD_masked.png')
    plt.clf()
    plt.close(fig)

if __name__ == '__main__':
    GenScores()
    reg_eq, reg_ne = combiner_LR()
    print_LR()
    evaluateLRCombiners(reg_eq, reg_ne)
    reg_0, reg_1, reg_2, reg_3 = combiner_LR_4()
    print_LR_4()
    evaluateLRCombiners_4(reg_0, reg_1, reg_2, reg_3)
    #drawScoreDist()
    #drawScoreSamples()
    #drawScoreSamples_marker()

