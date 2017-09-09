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



n = NetAddr.new( "127.0.0.1", 57400 );

~morphMap = [-1,2,\linear].asSpec;
~morphMap = [0,2,\linear].asSpec;

~outRateMap = [0.025,0.3,\linear].asSpec;

~morphMap.map(0)

// interactions
(
MIDIdef.cc( \S7, { |v,ctl|
	~morphFactor = ~morphMap.map( v / 127.0 );
	n.sendMsg( "/morph", 0, ~morphFactor );
}, 94, 0 );
MIDIdef.cc( \S8, { |v,ctl|
	~morphFactor = ~morphMap.map( v / 127.0 );
	n.sendMsg( "/morph", 1, ~morphFactor );
}, 95, 0 );
MIDIdef.cc( \S9, { |v,ctl|
	~morphFactor = ~morphMap.map( v / 127.0 );
	n.sendMsg( "/morph", 2, ~morphFactor );
}, 7, 0 );

OSCdef( \output, { |msg|
	var val = msg[1] * 0.5 + 0.5;
	// val.postln;
	~outputValue = val;
	Ndef( \wavetable ).set( \rate, ~outRateMap.map( ~outputValue ) );
	~microWashes.do{ |it| it.rgbw_( (msg[1] - 1 * 255).round(1).asInteger, 0, 0, 0 ) };
	~setAllLights.value;
}, "/output" );

);

Tdef( \updateMorph ).stop;

~microWashes.do{ |it| it.rgbw_( 0, 0, 0, 0 ) }; ~setAllLights.value;

~monitor = SWPlotterMonitor( { ~outputValue }, 500 );
~monitor.start;

(
OSCdef( \x, { |msg|
	var val = msg.copyToEnd(1);
	// val.postln;
	~xvals = val;
	b.setn( 0, ~xvals );
}, "/x" );
);

~xvals.size

~monitorx = SWPlotterMonitor( { ~xvals }, 500, 30 );
~monitorx.start;
