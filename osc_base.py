# -*- coding: utf-8 -*-

### settings ###
hostport = 57120 ## where the OSC goes
###
hostip = "127.0.0.1"

myip = "127.0.0.1"
myport = 57400
###

import time
import OSC


# ========= send osc ============

def sendOSCMessage( path, args ):
  msg = OSC.OSCMessage()
  msg.setAddress( path )
  #print args
  for a in args:
    msg.append( a )
  try:
    oschost.send( msg )
    if verbose:
      print( "sending message", msg )
  except OSC.OSCClientError:
    if verbose:
      print( "error sending message", msg )

    #sendOSCMessage( "/sensenode/imu", mpusendData )

#------------ OSC handlers --------------

def handler_led( path, types, args, source ):        
    print( "Sensor input:", args, len(args) )
    #call function; args is an array with the values that are sent
      
####################### main ################

oschost = OSC.OSCClient()
send_address = ( hostip, hostport )
oschost.connect( send_address )

receive_address = ( myip, myport )
osc = OSC.OSCServer( receive_address )

# add handlers
osc.addMsgHandler( "/sensenode/address", handler_serial )  

while True:
  osc.handle_request()
  time.sleep(0.001)
