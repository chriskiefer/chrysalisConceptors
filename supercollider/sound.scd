s.options.numOutputBusChannels_(8);
s.boot;
/*
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(0)
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