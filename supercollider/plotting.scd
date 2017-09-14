// ~monitor.stop;

~evidences = Array.fill( 3, 0 );
~outputValue = 0;
~monitor = SWPlotterMonitor( { ~evidences ++ [ ~outputValue ] }, 500, 4 );
// ~monitor.start;
~monitor.plotter.setProperties( \backgroundColor, Color.black );

~scopeplotter = Plotter.new( "Scope Monitor", Rect(600, 30, 800, 250) );
~scopeplotter.parent.alpha_(0.5);
~scopeplotter.plotMode_( \steps );
~scopeplotter.setProperties( \backgroundColor, Color.black );

(
~scopeplotter.setProperties(
	\plotColor, [
			Color.yellow, Color.blue, Color.green, Color.magenta, Color.cyan, Color.yellow,
			Color.red(0.9), Color.blue(0.9), Color.green(0.9), Color.magenta(0.9), Color.cyan(0.9), Color.yellow(0.9),
		] );
);

// Tdef( \updatePlot ).stop;
(
Tdef( \updatePlot, {
	loop {
		defer{ ~scopeplotter.value_( [ ~xvalCL1, ~xvalCL2, ~xvalCL3, ~xvalPG ] ); };
		0.05.wait;
	}
});
);