(
SynthDef( \countDown, { Out.ar( 0, SinOsc.ar( \freq.kr(300)*[1,2,3.5,7], 0, 0.1/4 ).sum*EnvGen.kr(Env.perc,timeScale: \dur.kr(0.2),doneAction:2) ) }).add;
// ~gestures.shape;
Tdef( \recordGesture, {
	~gestures = Array.fill( 9, { Array.fill( 120, { Array.fill( 3, nil ) } ) } );
	"count down to make gesture!".postln;
	3.reverseDo{ |it,i|
		("..."++it).post;
		Synth.new( \countDown, [\freq, [300,300,450].at( i ), \dur, 0.2 ]  );
		1.0.wait;
	};
	Synth.new( \countDown, [\freq, 600, \dur, 1 ]  );
	"...go!!".postln;
	120.do{|i|
		[111, 112, 113,  131, 132, 133,  141, 142, 143 ].do{ |id,j|
			~gestures.at( j ).put( i, x.nodes[id].data );
		};
		(1/30).wait;
	};
	("...recorded gesture" + ~counter).postln;
	Synth.new( \countDown, [\freq, 200, \dur, 1 ]  );
	~writeGestureToSCFile.value( ~gestures );
});
);


(
~basepath = "".resolveRelative;
~counter = 0;
// function to write to file in a format that can be read into python:
~writeGestureToSCFile = { |gesture|
	var file = File.new( (~basepath +/+ "gesturedata_" ++ ~counter ++ ".scd"), "w" );
	var string = gesture.asCompileString;
	file.write( string );
	// file.write( string.replace( "[ ", "" ).replace( " ]", "" ).replace( ", ", " " ) );
	file.close;
	~counter = ~counter + 1;
};
);

x[\duim].action = { |val| if ( val == 1 ){ Tdef( \recordGesture ).reset.play; } };


// now get ready to make the gesture!
Tdef( \recordGesture ).play;

// plot
~gestures.plot;





(
~accDevSumMap = [0.001,0.5, \exponential].asSpec;
~dmxMap = [1,255,\linear,1].asSpec;

Tdef( \updateLightRecording, {
	loop {
		var vals = [ x[\acc1devSum].data[0], x[\acc2devSum].data[0], x[\acc3devSum].data[0] ];
		vals = ~accDevSumMap.unmap( vals );
		vals = ~dmxMap.map( vals );
		// vals = vals.specMap( ~accdevSumMap, ~dmxMap );
		~microWashes.do{ |it| it.rgb_( *vals )	};
		~setAllLights.value;
		0.05.wait;
	}
});
);


~gest1 = File.readAllString( (~basepath +/+ "gesturedata_1.scd") ).interpret;
(
// ~gest1[0].flop.plot;
// ~gest1[3].flop.plot;
// ~gest1[6].flop.plot;
~gest1[6].flop.sum.plot("gesture1");
);


~gest2 = File.readAllString( (~basepath +/+ "gesturedata_2.scd") ).interpret;
(
// ~gest2[0].flop.plot;
// ~gest2[3].flop.plot;
// ~gest2[6].flop.plot;
~gest2[6].flop.sum.plot("gesture2");
);

~gest3 = File.readAllString( (~basepath +/+ "gesturedata_3.scd") ).interpret;
(
// ~gest2[0].flop.plot;
// ~gest2[3].flop.plot;
// ~gest2[6].flop.plot;
~gest3[6].flop.sum.plot("gesture3");
);

(
~gest4 = File.readAllString( (~basepath +/+ "gesturedata_4.scd") ).interpret;
// ~gest2[0].flop.plot;
// ~gest2[3].flop.plot;
// ~gest2[6].flop.plot;
~gest4[6].flop.sum.plot("gesture4");
);

(
~gest5 = File.readAllString( (~basepath +/+ "gesturedata_5.scd") ).interpret;
// ~gest2[0].flop.plot;
// ~gest2[3].flop.plot;
// ~gest2[6].flop.plot;
~gest5[6].flop.sum.plot("gesture5");
);

(
~gest6 = File.readAllString( (~basepath +/+ "gesturedata_6.scd") ).interpret;
// ~gest2[0].flop.plot;
// ~gest2[3].flop.plot;
// ~gest2[6].flop.plot;
~gest6[6].flop.sum.plot("gesture6");
)

(
~gest7 = File.readAllString( (~basepath +/+ "gesturedata_7.scd") ).interpret;
// ~gest2[0].flop.plot;
// ~gest2[3].flop.plot;
// ~gest2[6].flop.plot;
~gest7[6].flop.sum.plot("gesture7");
)

(
// function to write to file in a format that can be read into python:
~writeGestureToPythonFile = { |gesture,id1,id2|
	var file = File.new( ("gesturedata_" ++ id1 ++ "_" ++ id2 ++ ".txt").resolveRelative, "w" );
	var string = gesture.asCompileString;
	file.write( string.replace( "[ ", "" ).replace( " ]", "" ).replace( ", ", " " ) );
	file.close;
}
);

~gesture1a = ~gest1[6].flop.sum.copyFromStart(49);
~gesture1b = ~gest1[7].flop.sum.copyFromStart(49);
~gesture1c = ~gest1[8].flop.sum.copyFromStart(49);
// write to file
~writeGestureToPythonFile.value( ~gesture1a, 1, 1 );
~writeGestureToPythonFile.value( ~gesture1b, 1, 2 );
~writeGestureToPythonFile.value( ~gesture1c, 1, 3 );


~gesture5a = ~gest5[6].flop.sum.copyRange(50,99);
~gesture5b = ~gest5[7].flop.sum.copyRange(50,99);
~gesture5c = ~gest5[8].flop.sum.copyRange(50,99);
// write to file
~writeGestureToPythonFile.value( ~gesture5a, 5, 1 );
~writeGestureToPythonFile.value( ~gesture5b, 5, 2 );
~writeGestureToPythonFile.value( ~gesture5c, 5, 3 );
