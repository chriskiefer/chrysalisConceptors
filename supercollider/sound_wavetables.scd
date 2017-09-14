(
~xsizes = [128, 32, 64, 128 ];
~xbufferCL1 = Buffer.alloc( s, ~xsizes[0]*2, 1 );
~xbufferCL2 = Buffer.alloc( s, ~xsizes[1]*2, 1 );
~xbufferCL3 = Buffer.alloc( s, ~xsizes[2]*2, 1 );
~xbufferPG = Buffer.alloc( s, ~xsizes[3]*2, 1 );

~xvalCL1 = Array.fill( ~xsizes[0], 0 );
~xvalCL2 = Array.fill( ~xsizes[1], 0 );
~xvalCL3 = Array.fill( ~xsizes[2], 0 );
~xvalPG = Array.fill( ~xsizes[3], 0 );
);
(
OSCdef( \xCL1, { |msg|
	var val = msg.copyToEnd(1);
	var signal = Signal.with( *val );
	// signal = signal.zeroPad(~xsizes[0]);
	~xvalCL1 = val;
	~xbufferCL1.setn( 0, signal.asWavetable );
}, "/x/CL1" );
);
(
OSCdef( \xCL2, { |msg|
	var val = msg.copyToEnd(1);
	var signal = Signal.with( *val );
	// signal = signal.zeroPad(~xsize);
	~xvalCL2 = val;
	~xbufferCL2.setn( 0, signal.asWavetable );
}, "/x/CL2" );
);
(
OSCdef( \xCL3, { |msg|
	var val = msg.copyToEnd(1);
	var signal = Signal.with( *val );
	// signal = signal.zeroPad(~xsize);
	~xvalCL3 = val;
	~xbufferCL3.setn( 0, signal.asWavetable );
}, "/x/CL3" );
);
(
OSCdef( \xPG, { |msg|
	var val = msg.copyToEnd(1);
	var signal = Signal.with( *val );
	signal = signal.zeroPad(~xsizes[3]);
	~xvalPG = val;
	~xbufferPG.setn( 0, signal.asWavetable );
	~waveSynthPG.set( \amp, 5 / ( ~xvalPG.maxItem - ~xvalPG.minItem ) );
}, "/x/PG" );
);
(
SynthDef( \wavetableOSC, {
	var osc = Osc.ar( \bufnum.kr(0), \freq.kr(1,0.5) );
	var signal = RLPF.ar( osc, \ffreq.kr(1000, 0.5) * \farray.kr( [1,5/4,4/5] ), \rq.kr(0.2) ).sum / 3;
	var spat = PanAz.ar( 8, signal, \pos.kr(0,0.2) + \posOffset.kr(0) );
	Out.ar( 0, spat * 0.05 * \amp.kr(1) );
}).add;
);

s.sync;

(
~freqs = 20 * [1,5/4,4/5,2/3];
~ffreqs = 600 * [1,5/4,4/5,2/3];
~waveSynthCL1 = Synth.new( \wavetableOSC, [ \bufnum, ~xbufferCL1.bufnum, \freq, ~freqs[0], \ffreq, ~ffreqs[0], \posOffset, 0, \amp, 1 ] );
~waveSynthCL2 = Synth.new( \wavetableOSC, [ \bufnum, ~xbufferCL2.bufnum, \freq, ~freqs[1], \ffreq, ~ffreqs[1], \posOffset, 0.5, \amp, 1 ] );
~waveSynthCL3 = Synth.new( \wavetableOSC, [ \bufnum, ~xbufferCL3.bufnum, \freq, ~freqs[2], \ffreq, ~ffreqs[2], \posOffset, 1, \amp, 1 ] );
~waveSynthPG  = Synth.new( \wavetableOSC, [ \bufnum, ~xbufferPG.bufnum, \posOffset, 1.5, \amp, 0.001,  \rq, 0.01, \ffreq, 300, \freq, 1, \farray, [ 1, 3/2, 4/5 ] ] );

~waveSynthCL1.set( \amp, 1, \freq, ~freqs[0], \ffreq, ~ffreqs[0], \rq, 0.05 );
~waveSynthCL2.set( \amp, 1, \freq, ~freqs[1], \ffreq, ~ffreqs[1], \rq, 0.05 );
~waveSynthCL3.set( \amp, 1, \freq, ~freqs[2], \ffreq, ~ffreqs[2], \rq, 0.05 );
);

//
// ~waveSynthPG.set( \freq, 100 );
// ~waveSynthPG.set( \ffreq, 500, \rq, 0.01 );
// ~waveSynthPG.set( \farray, [ 1, 3/2, 4/5 ] );

/*

~waveSynthCL1.set( \amp, 1, \freq, ~freqs[0], \ffreq, ~ffreqs[0], \rq, 0.05 );
~waveSynthCL2.set( \amp, 1, \freq, ~freqs[1], \ffreq, ~ffreqs[1], \rq, 0.05 );
~waveSynthCL3.set( \amp, 1, \freq, ~freqs[2], \ffreq, ~ffreqs[2], \rq, 0.05 );

(
~waveSynthCL1.free;
~waveSynthCL2.free;
~waveSynthCL3.free;
);

~waveSynthPG.free;

~xbufferPG.free;
*/