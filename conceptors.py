# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import scipy as sp

# normalized root mean square between two time series
# output: actual output of network
# target: desired output of network)
def nrmse(output,target):
    combinedVar = 0.5 * (np.var(target, ddof=1) + np.var(output, ddof=1))
    errorSignal = output - target
    return np.sqrt(np.mean(errorSignal ** 2) / combinedVar)

# generates internal weights for the network
# nInternalUnits: how many units
# connectivity: percentage of connections
def generateInternalWeights(nInternalUnits, connectivity):
    success = False
    internalWeights = 0
    while success == False:
        try:
            internalWeights = np.random.randn(nInternalUnits,nInternalUnits) * (np.random.random((nInternalUnits,nInternalUnits)) < connectivity)
            specRad = abs(np.linalg.eig(internalWeights)[0][0])
            if (specRad > 0):
                internalWeights = internalWeights / specRad
                success = True
        except e:
            print(e)
    return internalWeights



# making waveform patterns

pSaw = lambda n: (round(n % waveLengthSamples) / waveLengthSamples * 2) - 1.0
pPulse = lambda n: (((n % waveLengthSamples) < (waveLengthSamples * 0.5)) * 2) - 1.0
pSine2 = lambda n: (sin(n) * sin((n+pi/4)/6))
pSine3 = lambda n: (sin(n) * sin((n/4)/6)/6)

pJ1 = lambda n: 1 * sin(2 * pi * n / 3.1504531)
pJ1b = lambda n: 1 * sin(n/2) ** 1

period2 = 2
rawp = np.random.randn(period2)
# rawp = np.array([1.1929,2.6856]);
maxVal = np.max(rawp)
minVal = np.min(rawp)
rp = 0.5 * (2 * (rawp - minVal) / (maxVal - minVal) - 1);
pJ2 = lambda n: rp[mod(n, period2 )]
pTri = lambda n,p: (((n % p) >= (p/2)) * ((p/2) - (n % (p/2))) + ((n % p) < (p/2)) * (n % (p/2))) * (2/p)



# make a network
# p is a set of parameters
def makeNetwork(p):
    signalPlotLength = 15
   # pattern readout learning
    patterns = np.array([1,2])

    Netconnectivity = 1
    if p['N'] > 20:
        Netconnectivity = 10.0/p['N'];
    WstarRaw = generateInternalWeights(p['N'], Netconnectivity)
    WinRaw = np.random.randn(p['N'], 1)
    WbiasRaw = np.random.randn(p['N'], 1)

    #Scale raw weights
    Wstar = p['NetSR'] * WstarRaw;
    Win = p['NetinpScaling'] * WinRaw;
    Wbias = p['BiasScaling'] * WbiasRaw;
    I = np.eye(p['N'])
    xCollector = np.zeros((p['N'], p['learnLengthWout']))
    pCollector = np.zeros((1, p['learnLengthWout']))
    x = np.zeros((p['N'],1))


    for n in arange(p['washoutLength'] + p['learnLength']):
        u = np.random.randn() * 1.5
        x = np.tanh((Wstar * x) + (Win * u + Wbias))
        if n >= p['washoutLength']:
            xCollector[:, n - p['washoutLength']] = x[:,0]
            pCollector[0, n - p['washoutLength']] = u

#     print("Mean/Max/Min Activations, random network driven by noise")
#     plot(np.mean(xCollector.T, axis=1))
#     plot(np.max(xCollector.T, axis=1))
#     plot(np.min(xCollector.T, axis=1))

    Wout = linalg.inv(xCollector.dot(xCollector.conj().T) +
                  (p['TychonovAlphaReadout'] * np.eye(p['N']))).dot(xCollector).dot(pCollector.conj().transpose()).conj().T
    print("Initial training")
    print("NRMSE: ", nrmse(Wout.dot(xCollector), pCollector))
    print("absWeight: ", mean(abs(Wout)))

    allTrainArgs = np.zeros((p['N'], p['patts'].size * p['learnLength']))
    allTrainOldArgs = np.zeros((p['N'], p['patts'].size * p['learnLength']))
    allTrainTargs = np.zeros((p['N'], p['patts'].size * p['learnLength']))
    allTrainOuts = np.zeros((1, p['patts'].size * p['learnLength']))
    xCollectors =  np.zeros((1, p['patts'].size), dtype=np.object)
    SRCollectors =  np.zeros((1, p['patts'].size), dtype=np.object)
    URCollectors =  np.zeros((1, p['patts'].size), dtype=np.object)
    patternRs =  np.zeros((1, p['patts'].size), dtype=np.object)
    train_xPL =  np.zeros((1, p['patts'].size), dtype=np.object)
    train_pPL =  np.zeros((1, p['patts'].size), dtype=np.object)
    startXs =  np.zeros((p['N'], p['patts'].size), dtype=np.object)

    for i_pattern in range(p['patts'].size):
        print('Loading pattern ', i_pattern)
        patt = p['patts'][i_pattern]
        xCollector = zeros((p['N'], p['learnLength']))
        xOldCollector = zeros((p['N'], p['learnLength']))
        pCollector = zeros((1, p['learnLength']))
        x = zeros((p['N'],1))
        for n in range(p['washoutLength'] + p['learnLength']):
            u = patt(n+1)
            xOld = x
            x = tanh((Wstar * x) + (Win * u) + Wbias)
            if n >= p['washoutLength']:
                xCollector[:, n - p['washoutLength']] = x[:,0]
                xOldCollector[:, n - p['washoutLength']] = xOld[:,0]
                pCollector[0, n - p['washoutLength']] = u

        xCollectors[0,i_pattern] = xCollector
        R = xCollector.dot(xCollector.T) / p['learnLength']
        [Ux,sx,Vx] = svd(R)
        SRCollectors[0,i_pattern] = diag(sx)
        URCollectors[0,i_pattern] = Ux
        patternRs[0,i_pattern] = R

        startXs[:,i_pattern] = x[:,0]

        #needed?
        train_xPL[0,i_pattern] = xCollector[:,:signalPlotLength]
        train_pPL[0,i_pattern] = pCollector[0,:signalPlotLength]
        ###

        allTrainArgs[:, i_pattern * p['learnLength']:(i_pattern+1) * p['learnLength']] = xCollector
        allTrainOldArgs[:, i_pattern * p['learnLength']:(i_pattern+1) * p['learnLength']] = xOldCollector
        allTrainOuts[0, i_pattern * p['learnLength']:(i_pattern+1) * p['learnLength']] = pCollector
        allTrainTargs[:, i_pattern * p['learnLength']:(i_pattern+1) * p['learnLength']] = Win.dot(pCollector)

    Wtargets = np.arctanh(allTrainArgs) - np.tile( Wbias, (1, p['patts'].size * p['learnLength']))

    W = linalg.inv(allTrainOldArgs.dot(allTrainOldArgs.conj().T) +
                      (p['TychonovAlpha'] * np.eye(p['N']))).dot(allTrainOldArgs).dot(Wtargets.conj().T).conj().T
    print("W NMRSE: ", mean(nrmse(W.dot(allTrainOldArgs), Wtargets)))
    print("absSize: ", mean(mean(abs(W), axis=0)))

    # figure(1)
    # plot(np.mean(W.dot(allTrainOldArgs).T, axis=1))

    print('Computing conceptors')

    Cs = np.zeros((4, p['patts'].size), dtype=np.object)
    for i_pattern in range(p['patts'].size):
        R = patternRs[0,i_pattern]
        [U,s,V] = svd(R)
        S = diag(s)
        Snew = (S * linalg.inv(S + pow(p['alphas'][i_pattern], -2) * np.eye(p['N'])))

        C =  U.dot(Snew).dot(U.T);
        Cs[0,i_pattern] = C
        Cs[1,i_pattern] = U
        Cs[2,i_pattern] = diag(Snew)
        Cs[3,i_pattern] = diag(S)

    x_CTestPL = np.zeros((3, p['recallTestLength'], p['patts'].size))
    p_CTestPL = np.zeros((1, p['recallTestLength'], p['patts'].size))
    for i_pattern in range(p['patts'].size):
        C = Cs[0,i_pattern]
        x = 0.5 * np.random.randn(p['N'],1)
        for n in range(p['recallTestLength'] + p['washoutLength']):
            x = np.tanh(W.dot(x) + Wbias)
            x = C.dot(x)
            if (n > p['washoutLength']):
                x_CTestPL[:,n-p['washoutLength'],i_pattern] = x[0:3].T
                p_CTestPL[:,n-p['washoutLength'],i_pattern] = Wout.dot(x)
    # for i_pattern in range(p['patts'].size):
    #     figure(2 + i_pattern)
    #     plot(p_CTestPL[:,:,i_pattern].T)
    #     plot([p['patts'][i_pattern](x) for x in arange(p['recallTestLength'])])

    return locals()

##### OSC server ####

from liblo import *
import sys


class MyServer(ServerThread):
    def __init__(self, onMorph):
        ServerThread.__init__(self, 57400)
        self.onMorph = onMorph

    @make_method('/morph', 'f')
    def morph_callback(self, path, args):
        f = args[0]
#        print("received message ",path," with arguments: ", f)
        self.onMorph(f)
        print("morph")

    @make_method(None, None)
    def fallback(self, path, args):
        print("received unknown message ", s)

def makeOSCServer(onMorph):
    try:
        server.free()
    except:
        pass

    try:
        server = MyServer(onMorph)
        print("server running")
    except err:
        print(str(err))

    server.start()




## in notebook:

# params = {'N':30, 'NetSR':1.6, 'NetinpScaling':1.6,'BiasScaling':0.3,'TychonovAlpha':0.0001,
#          'washoutLength':100, 'learnLength':500, 'TychonovAlphaReadout':0.0001,
#          'learnLengthWout':500, 'recallTestLength':100,
#          'alphas':np.array([12.0,24.0]),
#           'patts':np.array([pJ1b, pJ2])
#          }
#
# net = makeNetwork(params)
