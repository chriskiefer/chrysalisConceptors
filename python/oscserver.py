# -*- coding: utf-8 -*-
#!/usr/bin/env python

##### OSC server ####

from liblo import *
import sys


class MyServer(ServerThread):
    def __init__(self, targetPort, recvPort, onExit, targetPort2 = 57110):
        ServerThread.__init__(self, recvPort)
        print("server created")
        self.onExit = onExit
        try:
            self.target = Address(targetPort)
            self.target2 = Address(targetPort2)
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

    def send_to_bus( self, busid, vals ):
        # we can also build a message object first...
        msg = Message( "/c_setn" )
        msg.add( busid )
        msg.add( len( vals ) )
        # ... append arguments later...
        for v in vals:
            msg.add(v)
        # ... and then send it
        send(self.target2, msg)

    @make_method('/osc/target', 'is')
    def osc_port_callback(self, path, args):
        port = args[0]
        hostname = args[1]
        print("received message ",path," with arguments: ", port, hostname)
        try:
            self.target = Address(hostname,port)
        except AddressError as err:
            print (str(err))
            sys.exit()
        
    @make_method('/input', 'if')
    def input_callback(self, path, args):
        ind = args[0]
        f = args[1]
        #print("received input message ",path," with arguments: ", ind, f)
        if self.onInput != None:
            value = self.onInput(ind,f)
        #send(self.target, "/output", value)

    @make_method('/accelero', None)
    def accelero_callback(self, path, args):
        #print("received input message ",path," with arguments: ", ind, f)
        if self.onAccelero != None:
            value = self.onAccelero(args)
        #send(self.target, "/output", value)

    @make_method('/train/model', None ) # sends index
    def train_model_callback(self, path, args):
        #print("received input message ",path," with arguments: ", ind, f)
        if self.onTrainModel != None:
            value = self.onTrainModel( args )
        #send(self.target, "/output", value)

    @make_method('/start/model', '') # sends index
    def start_model_callback(self, path, args):
        #print("received input message ",path," with arguments: ", ind, f)
        if self.onStartModel != None:
            value = self.onStartModel( )
        #send(self.target, "/output", value)


    @make_method('/record/on', 'i') # sends index
    def record_on_callback(self, path, args):
        #print("received input message ",path," with arguments: ", ind, f)
        if self.onRecordOn != None:
            value = self.onRecordOn( args[0] )
        #send(self.target, "/output", value)

    @make_method('/record/off', '')
    def record_off_callback(self, path, args):
        #print("received input message ",path," with arguments: ", ind, f)
        if self.onRecordOff != None:
            value = self.onRecordOff()
        #send(self.target, "/output", value)

    @make_method('/perform/on', 'i') # sends index
    def perform_on_callback(self, path, args):
        #print("received input message ",path," with arguments: ", ind, f)
        if self.onPerformOn != None:
            value = self.onPerformOn( args[0] )
        #send(self.target, "/output", value)

    @make_method('/perform/off', '')
    def perform_off_callback(self, path, args):
        #print("received input message ",path," with arguments: ", ind, f)
        if self.onPerformOff != None:
            value = self.onPerformOff()
        #send(self.target, "/output", value)


    @make_method('/conceptor/store', 'i')
    def conceptor_callback(self, path, args):
        ind = args[0]
#        print("received message ",path," with arguments: ", f)
        if self.onConceptorStore != None:
            value = self.onConceptorStore(ind)
        #send(self.target, "/output", value)

    @make_method('/spectralradius', 'if')
    def spectral_callback(self, path, args):
        ind = args[0]
        f = args[1]
#        print("received message ",path," with arguments: ", f)
        if self.onSpectral != None:
            value = self.onSpectral(ind,f)
        #send(self.target, "/output", value)

    @make_method('/leakrate', 'if')
    def leak_callback(self, path, args):
        ind = args[0]
        f = args[1]
#        print("received message ",path," with arguments: ", f)
        if self.onLeakrate != None:
            value = self.onLeakrate(ind,f)
        #send(self.target, "/output", value)

    @make_method('/morph', 'if')
    def morph_callback(self, path, args):
        ind = args[0]
        f = args[1]
#        print("received message ",path," with arguments: ", f)
        if self.onMorph != None:
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


def makeOSCServer(targetPort,recvPort,onExit):
    try:
        server.free()
    except:
        pass

    try:
        server = MyServer(targetPort,recvPort,onExit)
    except err:
        print(str(err))

    return server
