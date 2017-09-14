~basePath = "".resolveRelative;

(
(~basePath +/+ "dmxSetup.scd").load;
(~basePath +/+ "ColorDash.scd").load;

);

~jupyter = NetAddr.new( "127.0.0.1", 57400 );
~jupyter2 = NetAddr.new( "127.0.0.1", 57401 );

(
s.options.numOutputBusChannels_(10);
s.waitForBoot({
	(~basePath +/+ "sound_wavetables.scd").load;
	(~basePath +/+ "plotting.scd").load;

	(~basePath +/+ "MiniBeeData.scd").load;
	(~basePath +/+ "morphcontrol.scd").load;

});
);

(
Tdef( \updatePlot ).play;
~monitor.start;
Tdef( \updateLight ).play;
);

// optional
x.makeNodeGui;


~jupyter.sendMsg( "/exit" );
~jupyter2.sendMsg( "/exit" );
