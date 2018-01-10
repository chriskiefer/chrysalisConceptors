(
Ndef( \sinOsc, {
	// SplayAz.ar( 8,
	SinOsc.ar(
		\freq.kr(100) * Array.geom( 100, 1, 1.05 ),
		0,
		// In.kr( 100, 500 ) * 2 * pi,
		HPF.kr( In.kr( 100, 100 ), \hpf.kr(0.2) )
		// 1/500
	).sum
// )
* \amp.kr(0.001);
	// SinOsc.ar( 100, 0, In.kr( 100 ) * \amp.kr(0.1) )
}).play
);

Ndef( \sinOsc ).clear;

Ndef( \sinOsc ).set( \amp, 0.01, \hpf, 0.5 );



(
Ndef( \resonz, {
	// SplayAz.ar( 8,
	BPF.ar(
		HPF.ar( K2A.ar( In.kr( 100, 100 ) ), \hpf.kr(0.2) ),
		\freq.kr(100) * Array.geom( 100, 1, 1.05 ),
		\rq.kr(0.1),
	).sum
// )
* \amp.kr(0.001);
	// SinOsc.ar( 100, 0, In.kr( 100 ) * \amp.kr(0.1) )
}).play
);

Ndef( \resonz ).set( \amp, 0.5, \hpf, 0.1, \rq, 0.01 )

Ndef( \resonz ).clear;

~buffer = Buffer.alloc( s, 64, 500 );

~buffer2 = Buffer.alloc( s, 64, 1 );

(
Ndef( \writeBufs ).clear;
Ndef( \writeBufs ).set( \bufnum, ~buffer2 );
Ndef( \writeBufs, {
	var intrig = InTrig.kr( 100, 1 );
	var stepper = Stepper.kr( intrig, min: 0, max: 64 );
	// Poll.kr( intrig, stepper );
	BufWr.kr( In.kr(100, 1 ), \bufnum.kr(0), stepper );
	// BufWr.kr( In.kr(
});
);

(
Ndef( \playBufs ).set( \bufnum, ~buffer2 );
Ndef( \playBufs, {
	BufRd.ar( 1, \bufnum.kr(0), Phasor.ar( 0, \rate.kr(1), 0, BufFrames.kr(0) ) )
	* \amp.kr(0.01)
}).play
);

Ndef( \playBufs ).clear;

Ndef( \playBufs ).set( \amp, 0.2, \rate, 0.1 );

NdefMixer.new(s)


