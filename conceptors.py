# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
#from numpy import linalg
import scipy as sp

# normalized root mean square between two time series
# output: actual output of network
# target: desired output of network)
def nrmse(output,target):
    combinedVar = 0.5 * (np.var(target, ddof=1) + np.var(output, ddof=1))
    errorSignal = output - target
    return np.sqrt(np.mean(errorSignal ** 2) / combinedVar)

# generates internal weights for the network
# nInternalUnits: how many units N x N
# connectivity: percentage of connections
def generateInternalWeights(nInternalUnits, connectivity):
    success = False
    internalWeights = 0
    while success == False:
        try:
            internalWeights = np.random.randn(nInternalUnits,nInternalUnits) * (np.random.random((nInternalUnits,nInternalUnits)) < connectivity)
            specRad = abs(np.linalg.eig(internalWeights)[0][0]) ## MB: why abs? then always 0 or greater
            if (specRad > 0):                                   ## MB: or is this just checking greater than 0?
                internalWeights = internalWeights / specRad
                success = True
        except e:
            print(e)
    return internalWeights

# make a network
# p is a set of parameters, like
#params = {
    # 'N':30, # size of RNN
    # 'NetSR':1.6,
    # 'NetinpScaling':1.6,
    # 'BiasScaling':0.3,
    # 'TychonovAlpha':0.0001,
    # 'TychonovAlphaReadout':0.0001,
    # 'washoutLength':100,
    # 'learnLength':500,
    # 'learnLengthWout':500,
    # 'recallTestLength':100,
    # 'alphas':np.array([12.0,24.0]),
    # 'patts':np.array([pJ1b, pJ2])
#}
def makeNetwork(p):
    NetConnectivity = 1 # just for small networks
    if p['N'] > 20:
        NetConnectivity = 10.0/p['N'];
    WstarRaw = generateInternalWeights(p['N'], NetConnectivity)
    WinRaw = np.random.randn(p['N'], 1)
    WbiasRaw = np.random.randn(p['N'], 1)

    #Scale raw weights
    Wstar = p['NetSR'] * WstarRaw;
    Win = p['NetinpScaling'] * WinRaw;
    Wbias = p['BiasScaling'] * WbiasRaw;
    I = np.eye(p['N']) # identity matrix

    xCollector = np.zeros((p['N'], p['learnLengthWout'])) # variable to collect states of x
    pCollector = np.zeros((1, p['learnLengthWout'])) # variable to collect states of p (output?)
    x = np.zeros((p['N'],1)) # initial state

    # first training: washout is to wash out the input state 'noise'; learnLength is then the actual amount of learning samples
    for n in np.arange(p['washoutLength'] + p['learnLength']):
        u = np.random.randn() * 1.5  # random input
        x = np.tanh((Wstar * x) + (Win * u + Wbias)) # calculate next internal activation
        if n >= p['washoutLength']:
            xCollector[:, n - p['washoutLength']] = x[:,0]
            pCollector[0, n - p['washoutLength']] = u

#     print("Mean/Max/Min Activations, random network driven by noise")
#     plot(np.mean(xCollector.T, axis=1))
#     plot(np.max(xCollector.T, axis=1))
#     plot(np.min(xCollector.T, axis=1))

    # Wout
    Wout = np.linalg.inv( xCollector.dot(xCollector.conj().T) + ( p['TychonovAlphaReadout'] * np.eye(p['N']) ) ).dot(xCollector).dot(pCollector.conj().transpose()).conj().T
    print("Initial training")
    print("NRMSE: ", nrmse(Wout.dot(xCollector), pCollector))
    print("absWeight: ", np.mean(abs(Wout)))

    allTrainArgs = np.zeros((p['N'], p['patts'].size * p['learnLength']))
    allTrainOldArgs = np.zeros((p['N'], p['patts'].size * p['learnLength']))
    allTrainTargs = np.zeros((p['N'], p['patts'].size * p['learnLength']))
    allTrainOuts = np.zeros((1, p['patts'].size * p['learnLength']))
    xCollectors =  np.zeros((1, p['patts'].size), dtype=np.object)
    SRCollectors =  np.zeros((1, p['patts'].size), dtype=np.object)
    URCollectors =  np.zeros((1, p['patts'].size), dtype=np.object)
    patternRs =  np.zeros((1, p['patts'].size), dtype=np.object)
    #train_xPL =  np.zeros((1, p['patts'].size), dtype=np.object)
    #train_pPL =  np.zeros((1, p['patts'].size), dtype=np.object)
    startXs =  np.zeros((p['N'], p['patts'].size), dtype=np.object)

    for i_pattern in range(p['patts'].size):
        print('Loading pattern ', i_pattern)
        patt = p['patts'][i_pattern]
        xCollector = np.zeros((p['N'], p['learnLength']))
        xOldCollector = np.zeros((p['N'], p['learnLength']))
        pCollector = np.zeros((1, p['learnLength']))
        x = np.zeros((p['N'],1))
        for n in range(p['washoutLength'] + p['learnLength']):
            u = patt(n+1)
            xOld = x
            x = np.tanh((Wstar * x) + (Win * u) + Wbias)
            if n >= p['washoutLength']:
                xCollector[:, n - p['washoutLength']] = x[:,0]
                xOldCollector[:, n - p['washoutLength']] = xOld[:,0]
                pCollector[0, n - p['washoutLength']] = u

        xCollectors[0,i_pattern] = xCollector
        R = xCollector.dot(xCollector.T) / p['learnLength']
        [Ux,sx,Vx] = np.linalg.svd(R)
        SRCollectors[0,i_pattern] = np.diag(sx)
        URCollectors[0,i_pattern] = Ux
        patternRs[0,i_pattern] = R

        startXs[:,i_pattern] = x[:,0]

        #needed?
        #train_xPL[0,i_pattern] = xCollector[:,:signalPlotLength]
        #train_pPL[0,i_pattern] = pCollector[0,:signalPlotLength]
        ###

        allTrainArgs[:, i_pattern * p['learnLength']:(i_pattern+1) * p['learnLength']] = xCollector
        allTrainOldArgs[:, i_pattern * p['learnLength']:(i_pattern+1) * p['learnLength']] = xOldCollector
        allTrainOuts[0, i_pattern * p['learnLength']:(i_pattern+1) * p['learnLength']] = pCollector
        allTrainTargs[:, i_pattern * p['learnLength']:(i_pattern+1) * p['learnLength']] = Win.dot(pCollector)

    Wtargets = np.arctanh(allTrainArgs) - np.tile( Wbias, (1, p['patts'].size * p['learnLength']))

    W = np.linalg.inv(allTrainOldArgs.dot(allTrainOldArgs.conj().T) +
                      (p['TychonovAlpha'] * np.eye(p['N']))).dot(allTrainOldArgs).dot(Wtargets.conj().T).conj().T
    print("W NMRSE: ", np.mean(nrmse(W.dot(allTrainOldArgs), Wtargets)))
    print("absSize: ", np.mean(np.mean(abs(W), axis=0)))

    # figure(1)
    # plot(np.mean(W.dot(allTrainOldArgs).T, axis=1))

    print('Computing conceptors')

    Cs = np.zeros((4, p['patts'].size), dtype=np.object)
    for i_pattern in range(p['patts'].size):
        R = patternRs[0,i_pattern]
        [U,s,V] = np.linalg.svd(R)
        S = np.diag(s)
        Snew = (S * np.linalg.inv(S + pow(p['alphas'][i_pattern], -2) * np.eye(p['N'])))

        C =  U.dot(Snew).dot(U.T);
        Cs[0,i_pattern] = C
        Cs[1,i_pattern] = U
        Cs[2,i_pattern] = np.diag(Snew)
        Cs[3,i_pattern] = np.diag(S)

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

def conceptor_mix_step( net, x, morphvalues, oversample=1 ):
    ind = 0
    C = np.zeros( (net['p']['N'] , net['p']['N'] ) )
    for i_morph in morphvalues:
        C = C + (net['Cs'][0,ind].dot( i_morph ))
        ind = ind + 1
    Wsr = net['W'].dot(1.2)
    for i_oversample in range( oversample ):
        x = np.tanh(Wsr.dot(x) + net['Wbias'])
        x = C.dot(x)
    output = net['Wout'].dot(x)
    return output, x

##### OSC server ####

from liblo import *
import sys


class MyServer(ServerThread):
    def __init__(self, onMorph, onExit):
        ServerThread.__init__(self, 57400)
        self.onMorph = onMorph
        print("server created")

        try:
            self.target = Address(57120)
        except AddressError as err:
            print (str(err))
            sys.exit()

    def send_value( self, tag, value ):
        send( self.target, tag, value )

    def send_array( self, tag, vals ):
        # we can also build a message object first...
        msg = Message( tag )
        # ... append arguments later...
        for v in vals:
            msg.add(v)
        # ... and then send it
        send(self.target, msg)

    @make_method('/morph', 'if')
    def morph_callback(self, path, args):
        ind = args[0]
        f = args[1]
#        print("received message ",path," with arguments: ", f)
        value = self.onMorph(ind,f)
        #send(self.target, "/output", value)

    @make_method('/exit', None)
    def exit_callback(self, path, args):
#        print("received message ",path," with arguments: ", f)
        print("exit")
        value = self.onExit()
        send(self.target, "/exited" )

    @make_method(None, None)
    def fallback(self, path, args):
        print("received unknown message ", path, args)


def makeOSCServer(onMorph,onExit):
    try:
        server.free()
    except:
        pass

    try:
        server = MyServer(onMorph,onExit)
        print("server running")
    except err:
        print(str(err))

    server.start()
    return server

def freeServer():
    server.free()




## in notebook:

# params = {'N':30, 'NetSR':1.6, 'NetinpScaling':1.6,'BiasScaling':0.3,'TychonovAlpha':0.0001,
#          'washoutLength':100, 'learnLength':500, 'TychonovAlphaReadout':0.0001,
#          'learnLengthWout':500, 'recallTestLength':100,
#          'alphas':np.array([12.0,24.0]),
#           'patts':np.array([pJ1b, pJ2])
#          }
#
# net = makeNetwork(params)
