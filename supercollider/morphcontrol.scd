(
~morphMap = [-0.5,1.5,\linear].asSpec;
// ~stretchMap = [0.74, 0.96 ].asSpec;
~stretchMap = [0.96, 0.74 ].asSpec;
);

(
[101,102,103].do{ |id,i|
	x.nodes[id].action = MFunc.new;
	x.nodes[id].action.add( \morph, { |data|
		~jupyter.sendMsg( "/morph", i, data[0].specMap( ~stretchMap, ~morphMap ); );
	});
};
);


(
~outputMap = [-2,2].asSpec;
~positionMap = [0,2].asSpec;
~freqPGMap = [1, 10, \exponential].asSpec;
~ffreqPGMap = [350, 600, \exponential].asSpec;
OSCdef( \output1, { |msg|
	var val = msg[1];
	~outputValue = val;
	val = ~outputMap.unmap( val );
	~waveSynthPG.set(
		\pos, ~positionMap.map( val ),
		\freq, ~freqPGMap.map( val ),
		\ffreq, ~ffreqPGMap.map( val ),
	);
}, "/output" );
);

(
~outputMapAbs = [0,3].asSpec;
~dmxMap = [0,255,\linear,1].asSpec;
~intensitySpec = [1/255,1].asSpec;
~hueSpec = [0,0.999].asSpec;

Tdef( \updateLight, {
	loop {
		var intensity = ~outputValue.abs.specMap( ~outputMapAbs, ~intensitySpec );
		var hue = ~outputValue.specMap( ~outputMap, ~hueSpec );
		var color = Color.hsv( hue, 1, intensity, 0 );
		~microWashes.do{ |it| it.rgbw_( *~dmxMap.map( color.asArray ) )	};
		~setAllLights.value;
		0.05.wait;
	}
});
);

// ( ~xvalPG.maxItem - ~xvalPG.minItem )/2;
// ~waveSynthPG.set( \pos, 0);