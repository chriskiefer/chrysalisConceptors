# -*- coding: utf-8 -*-
#!/usr/bin/env python

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

    @make_method('/conceptor/store', 'i')
    def input_callback(self, path, args):
        ind = args[0]
#        print("received message ",path," with arguments: ", f)
        value = self.onConceptorStore(ind)
        #send(self.target, "/output", value)
        
    @make_method('/input', 'if')
    def input_callback(self, path, args):
        ind = args[0]
        f = args[1]
#        print("received message ",path," with arguments: ", f)
        value = self.onInput(ind,f)
        #send(self.target, "/output", value)

    @make_method('/spectralradius', 'if')
    def input_callback(self, path, args):
        ind = args[0]
        f = args[1]
#        print("received message ",path," with arguments: ", f)
        value = self.onSpectral(ind,f)
        #send(self.target, "/output", value)

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

