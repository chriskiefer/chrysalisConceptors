MIDIIn.connectAll;
MIDIFunc.trace( false )

// S1 - S9:
// 73, 75, 72, 91, 92, 93, 94, 95, 7

(
MIDIdef.cc( \S1, { |v,ctl| [ v, ctl ].postln }, 73, 0 );
MIDIdef.cc( \S2, { |v,ctl| [ v, ctl ].postln }, 75, 0 );
MIDIdef.cc( \S3, { |v,ctl| [ v, ctl ].postln }, 72, 0 );
MIDIdef.cc( \S4, { |v,ctl| [ v, ctl ].postln }, 91, 0 );
MIDIdef.cc( \S5, { |v,ctl| [ v, ctl ].postln }, 92, 0 );
MIDIdef.cc( \S6, { |v,ctl| [ v, ctl ].postln }, 93, 0 );
MIDIdef.cc( \S7, { |v,ctl| [ v, ctl ].postln }, 94, 0 );
MIDIdef.cc( \S8, { |v,ctl| [ v, ctl ].postln }, 95, 0 );
MIDIdef.cc( \S9, { |v,ctl| [ v, ctl ].postln }, 7, 0 );
);



(
~morphMap = [0,1,\linear].asSpec;

~outRateMap = [0.01,0.3,\linear].asSpec;
);
//~morphMap.map(0)

// interactions
(
MIDIdef.cc( \S7, { |v,ctl|
	~morphFactor = ~morphMap.map( v / 127.0 );
	n.sendMsg( "/morph", 0, ~morphFactor );
	n.sendMsg( "/morph", 3, ~morphFactor );
	n.sendMsg( "/morph", 6, ~morphFactor );
}, 94, 0 );
MIDIdef.cc( \S8, { |v,ctl|
	~morphFactor = ~morphMap.map( v / 127.0 );
	n.sendMsg( "/morph", 1, ~morphFactor );
	n.sendMsg( "/morph", 4, ~morphFactor );
	n.sendMsg( "/morph", 7, ~morphFactor );
}, 95, 0 );
MIDIdef.cc( \S9, { |v,ctl|
	~morphFactor = ~morphMap.map( v / 127.0 );
	n.sendMsg( "/morph", 2, ~morphFactor );
	n.sendMsg( "/morph", 5, ~morphFactor );
	n.sendMsg( "/morph", 8, ~morphFactor );
}, 7, 0 );

~outputMap = [-2,2].asSpec;
OSCdef( \output1, { |msg|
	var val = ~outputMap.unmap( msg[1] );
	~outputValue1 = val;
	Ndef( \wavetable ).set( \rate, ~outRateMap.map( ~outputValue1 ) );
}, "/output/1" );

OSCdef( \output2, { |msg|
	var val = ~outputMap.unmap( msg[1] );
	// val.postln;
	~outputValue2 = val;
	Ndef( \wavetable2 ).set( \rate, ~outRateMap.map( ~outputValue2 ) );
}, "/output/2" );

OSCdef( \output3, { |msg|
	var val = ~outputMap.unmap( msg[1] );
	// val.postln;
	~outputValue3 = val;
	Ndef( \wavetable3 ).set( \rate, ~outRateMap.map( ~outputValue3 ) );
}, "/output/3" );

);

(
~dmxMap = [0,255,\linear,1].asSpec;
Tdef( \updateLight, {
	loop {
		~microWashes.do{ |it| it.rgb_(
			~dmxMap.map( ~outputValue1 ),
			~dmxMap.map( ~outputValue2 ),
			~dmxMap.map( ~outputValue3 ),
			0
		)};
		~setAllLights.value;
		0.05.wait;
	}
}).play;
)

// OSCdef( \x ).remove
