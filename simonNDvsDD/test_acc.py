# !/bin/bash

from simon import check_testvector
from simon import make_train_data
from simon import make_train_data_VD

import numpy as np
import sys
from math import *
from tensorflow.keras.models import load_model

def evaluate(TDN, ITN, net, num_rounds, VD):
    accs = np.zeros(ITN)
    tprs = np.zeros(ITN)
    tnrs = np.zeros(ITN)
    mses = np.zeros(ITN)
    for iti in range(ITN):
        if VD == 0:
            X,Y = make_train_data(TDN, num_rounds);
        else:
            X,Y = make_train_data_VD(TDN, num_rounds);
        Z = net.predict(X,batch_size=1<<14);
        if len(Z)==2:
            Z = Z[-1]
        Z = Z.flatten()
        Zbin = (Z > 0.5);
        diff = Y - Z;
        mses[iti] = np.mean(diff*diff);
        n = len(Z); n0 = np.sum(Y==0); n1 = np.sum(Y==1);
        accs[iti] = np.sum(Zbin == Y) / n;
        tprs[iti] = np.sum(Zbin[Y==1]) / n1;
        tnrs[iti] = np.sum(Zbin[Y==0] == 0) / n0;
    print("Accuracy: ", np.mean(accs), "+-", np.std(accs), "TPR: ", np.mean(tprs), "+-", np.std(tprs), "TNR: ", np.mean(tnrs), "+-", np.std(tnrs), "MSE:", np.mean(mses), "+-", np.std(mses));


if __name__ == '__main__':
    check_testvector()
    TDN = 10**6
    ITN = 50
    wdir = './ResNet_lr_2e-3_1e-4/'
    oldStdout = sys.stdout
    logfile = open(wdir + 'evaluation.log', 'a')
    sys.stdout = logfile
    trained_model = wdir + 'best7depth10.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 7, 0);
    print('\n')
    trained_model = wdir + 'best8depth10.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 8, 0);
    print('\n')
    trained_model = wdir + 'best9depth10.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 9, 0);
    print('\n')
    logfile.close()
    #
    wdir = './ResNet_lr_1e-3_1e-5/'
    oldStdout = sys.stdout
    logfile = open(wdir + 'evaluation.log', 'a')
    sys.stdout = logfile
    trained_model = wdir + 'best7depth10.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 7, 0);
    print('\n')
    trained_model = wdir + 'best8depth10.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 8, 0);
    print('\n')
    trained_model = wdir + 'best9depth10.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 9, 0);
    print('\n')
    logfile.close()
    #
    wdir = './DenseNet/'
    oldStdout = sys.stdout
    logfile = open(wdir + 'evaluation.log', 'a')
    sys.stdout = logfile
    trained_model = wdir + '7_rounds_Simon_10depth.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 7, 0);
    print('\n')
    trained_model = wdir + '8_rounds_Simon_10depth.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 8, 0);
    print('\n')
    trained_model = wdir + '9_rounds_Simon_10depth.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 9, 0);
    print('\n')
    logfile.close()
    #
    wdir = './SENet/'
    oldStdout = sys.stdout
    logfile = open(wdir + 'evaluation.log', 'a')
    sys.stdout = logfile
    trained_model = wdir + 'ND_VV_Simon32_7R.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 7, 0);
    print('\n')
    trained_model = wdir + 'ND_VV_Simon32_8R.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 8, 0);
    print('\n')
    trained_model = wdir + 'ND_VD_Simon32_8R.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 8, 1);
    print('\n')
    trained_model = wdir + 'ND_VV_Simon32_9R.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 9, 0);
    print('\n')
    trained_model = wdir + 'ND_VD_Simon32_9R.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 9, 1);
    print('\n')
    trained_model = wdir + 'ND_VV_KA_Simon32_10R.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 10, 0);
    print('\n')
    trained_model = wdir + 'ND_VV_Simon32_10R.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 10, 0);
    print('\n')
    trained_model = wdir + 'ND_VV_Simon32_11R.h5'
    net = load_model(trained_model)
    print(trained_model)
    evaluate(TDN, ITN, net, 11, 0);
    print('\n')
    logfile.close()