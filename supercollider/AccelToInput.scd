// input to patterns
/*
(
([111,112,113]+40).do{ |id,i|
	x.nodes[id].action = { |data|  };
};
);
*/

n = NetAddr.new( "127.0.0.1", 57400 );
n = NetAddr.new( "127.0.0.1", NetAddr.langPort );


// x.nodes[151].action = { |data| n.sendMsg( "/input", 3, data[0] ) };
// x.nodes[152].action = { |data| n.sendMsg( "/input", 2, data[0] ) };
x.nodes[153].action = MFunc.new;
x.nodes[153].action.add( \learn, { |data| n.sendMsg( "/input", 1, data[0] ) } );

~stretchMap = [0.74, 0.96 ].asSpec;
~spectralMap = [ 0.7, 1.8 ].asSpec;
x[\stretch3].action = MFunc.new;
x[\stretch3].action.add( \morph, { |data| n.sendMsg( "/morph", 1,~stretchMap.unmap( data[0] ) ) } );
x[\stretch3].action.add( \spectral, { |data| n.sendMsg( "/spectralradius", 1, data[0].specMap( ~stretchMap, ~spectralMap ) ) } );


//
~currentMode = \learn; // options are: \learn - \morph - \spectral

(
~changeMode = { |newmode|
	// things to disable in the old mode
	switch( ~currentMode,
		\learn, { x.nodes[153].action.disable( \learn ); },
		\morph, { x[\stretch3].action.disable( \morph ); }, // mapping stretch sensor to morphfactor
		\spectral, { x[\stretch3].action.disable( \spectral ); }, // mapping stretch sensor to spectral radius
	);
	~currentMode = newmode;
	// things to enable in new mode
	switch( ~currentMode,
		\learn, { x.nodes[153].action.enable( \learn ); },
		\morph, { x[\stretch3].action.enable( \morph ); }, // mapping stretch sensor to morphfactor
		\spectral, { x[\stretch3].action.enable( \spectral ); }, // mapping stretch sensor to spectral radius
	);
};
);

// question: in \morph should I still have input?
// question

// toggle between learn and morph
x[\duim].action = { |val| if ( val == 1 ){ if ( ~currentMode == \morph ){ ~changeMode.value( \learn ) }{ ~changeMode.value( \morph ) }; ~currentMode.postln; }; };
// toggle between learn and spectral
x[\pink].action = { |val| if ( val == 1 ){ if ( ~currentMode == \spectral ){~changeMode.value( \learn ) }{ ~changeMode.value( \spectral )}; ~currentMode.postln; }; };


x[\wijs].action = { |val| if ( val == 1 ){ n.sendMsg( "/conceptor/store", 1 ); } };
x[\middel].action = { |val| if ( val == 1 ){ n.sendMsg( "/conceptor/store", 2 ); } };
x[\ring].action = { |val| if ( val == 1 ){ n.sendMsg( "/conceptor/store", 3 ); } };

OSCFunc.trace(false);

OSCdef(\testmorph, { |msg| msg.postln; }, "/morph" );
OSCdef(\testspectral, { |msg| msg.postln; }, "/spectralradius" );
OSCdef(\testinput, { |msg| msg.postln; }, "/input" );
OSCdef(\testconceptor, { |msg| msg.postln; }, "/conceptor/store" );