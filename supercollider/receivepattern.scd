// and now we assign an action to send the data in realtime to the python script:
n = NetAddr.new( "127.0.0.1", 57400 ); // python's address

(
n.sendMsg( "/morph", 0, 0.0 );
n.sendMsg( "/morph", 1, 1.0 );
);

(
n.sendMsg( "/morph", 0, 1.0 );
n.sendMsg( "/morph", 1, 0.0 );
);

(
n.sendMsg( "/morph", 0, 0.5 );
n.sendMsg( "/morph", 1, 0.8 );
);

(
n.sendMsg( "/morph", 0, 1.0 );
n.sendMsg( "/morph", 1, 0.0 );
);

n.sendMsg( "/spectralradius", 0, 1.0 );

OSCFunc.trace(false)

~outSpec = [-2,1.0].asSpec;
~lightSpec = [0,255,\linear,1].asSpec;
~whichSpec = [0,8,\linear,1].asSpec;

OSCdef( \output, { |msg| msg[1].postln; }, "/output/1" );

OSCdef( \output, { |msg|  }, "/output/1" );


(
OSCdef( \output, { |msg|
	~colorDashes.do{ |it| it.red_( 0 ) };
	~colorDashes.wrapAt( msg[1].specMap( ~outSpec, ~whichSpec ) ).red_( msg[1].specMap( ~outSpec, ~lightSpec ) );
}, "/output/1" );
)

Tdef( \lightUpdate, { loop{	~setAllDashes.value; 0.05.wait; } } ).play; // fixed rate

// stop it!!!
n.sendMsg( "/exit" );