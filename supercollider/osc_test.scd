n = NetAddr.new( "127.0.0.1", 57400 );
OSCFunc.trace;

OSCFunc.trace( false );

n.sendMsg( "/ping", 2, 4.0, 0.2 );

n.sendMsg( "/iterate", 1.0, 1.0, 1.0);

~scAddr = NetAddr.new( "127.0.0.1", 57400 );
(
OSCdef(\output, {|msg, time, addr, recvPort|
	msg.postln;
}, '/output', ~scAddr);
)