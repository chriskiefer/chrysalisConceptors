s.boot;

Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(0)
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).play(1)
Ndef( \testSound, { WhiteNoise.ar( 0.2 ) } ).stop
Ndef( \testSound ).clear;

// tryout

b = Buffer.alloc( s, 30, 1 );
b.setn( 0, ~xvals );

Ndef( \wavetable ).clear;
b.plot
(
Ndef( \wavetable, {
	Splay.ar(
		PlayBuf.ar( 1, \bufnum.kr(0), \rate.kr(1,0.5)*[1,5/4,4/5], loop: 1 )
	);
	// BufRd.ar( 1, \bufnum.kr(0), LFSaw.ar( \freq.kr(200) ).unipolar.poll(11) );
});
);

Ndef( \wavetable ).play;
NdefMixer.new(s);