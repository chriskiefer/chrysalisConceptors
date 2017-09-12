s.options.numOutputBusChannels_(8);
s.boot;
/*
Ndef( \testSound, { SinOsc.ar( 300*[1,2,3.5,7], 0, 0.1/4 ).sum*EnvGen.kr(Env.perc,Impulse.kr(1)) } ).play(0)
Ndef( \testSound, { SinOsc.ar( 300, 0, 0.1 )*EnvGen.kr(Env.perc,Impulse.kr(1)) } ).play(0)

Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(1)
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(2)
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(3)
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(4)
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(5)
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(6)
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(7)
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).stop
Ndef( \testSound ).clear;
*/
// tryout

(
b = Buffer.alloc( s, 30, 1 );
~b2 = Buffer.alloc( s, 30, 1 );
~b3 = Buffer.alloc( s, 30, 1 );
);

(
Ndef( \wavetable, {
	Splay.ar(
		PlayBuf.ar( 1, \bufnum.kr(0), \rate.kr(1,0.5)*[1,5/4,4/5], loop: 1 )
	);
	// BufRd.ar( 1, \bufnum.kr(0), LFSaw.ar( \freq.kr(200) ).unipolar.poll(11) );
});
);


(
Ndef( \wavetable, {
	Splay.ar(
		PlayBuf.ar( 1, \bufnum.kr(0), \rate.kr(1,0.5)*[1,5/4,4/5], loop: 1 )
	);
	// BufRd.ar( 1, \bufnum.kr(0), LFSaw.ar( \freq.kr(200) ).unipolar.poll(11) );
});
);

(
Ndef( \wavetable2, {
	Splay.ar(
		PlayBuf.ar( 1, \bufnum.kr(1), \rate.kr(1,0.5)*[1,5/4,4/5], loop: 1 )
	);
	// BufRd.ar( 1, \bufnum.kr(0), LFSaw.ar( \freq.kr(200) ).unipolar.poll(11) );
});
);

(
Ndef( \wavetable3, {
	Splay.ar(
		PlayBuf.ar( 1, \bufnum.kr(2), \rate.kr(1,0.5)*[1,5/4,4/5], loop: 1 )
	);
	// BufRd.ar( 1, \bufnum.kr(0), LFSaw.ar( \freq.kr(200) ).unipolar.poll(11) );
});
);

(
OSCdef( \x1, { |msg|
	var val = msg.copyToEnd(1);
	// val.postln;
	~xvals = val;
	b.setn( 0, ~xvals );
}, "/x/1" );

OSCdef( \x2, { |msg|
	var val = msg.copyToEnd(1);
	// val.postln;
	~xvals2 = val;
	~b2.setn( 0, ~xvals2 );
}, "/x/2" );

OSCdef( \x3, { |msg|
	var val = msg.copyToEnd(1);
	// val.postln;
	~xvals3 = val;
	~b3.setn( 0, ~xvals3 );
}, "/x/3" );
);

Ndef( \wavetable ).play;
Ndef( \wavetable2 ).play;
Ndef( \wavetable3 ).play;

NdefMixer.new(s);

// wavetable players:
OSCdef( \x1wt ).free
Signal.with( *~xvals );
~b3.free;
~b3.plot;
~b3 = Buffer.alloc( s, 32*2, 1 );
(
OSCdef( \x1wt, { |msg|
	var val = msg.copyToEnd(1);
	var signal = Signal.with( *val );
	// signal.size.postln;
	~b3.setn( 0, signal.asWavetable );
}, "/x/1" );
);
(
Ndef( \wavetableOSC, {
	Splay.ar(
		Osc.ar( \bufnum.kr(0), \freq.kr(1,0.5) )
	);
	// BufRd.ar( 1, \bufnum.kr(0), LFSaw.ar( \freq.kr(200) ).unipolar.poll(11) );
});
);

Ndef( \wavetableOSC ).set( \bufnum, ~b3.bufnum );

Ndef( \wavetableOSC ).set( \freq, 50 );

(
~freqMap = [20, 1000, \exponential ].asSpec;
OSCdef( \output1wt, { |msg|
	var val = ~outputMap.unmap( msg[1] );
	// val.postln;
	Ndef( \wavetableOSC ).set( \freq, ~freqMap.map( ~outputValue1 ) );
}, "/output/1" );
);