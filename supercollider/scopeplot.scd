~monitor.stop;

~monitor = SWPlotterMonitor( { [~outputValue1, ~outputValue2, ~outputValue3 ] }, 500, 3 );
~monitor.start;
~monitor.plotter.setProperties( \backgroundColor, Color.black );


~monitorx = SWPlotterMonitor( { ~xvals }, 500, 30 );
~monitorx.start;

s.scope


~scopeplotter = Plotter.new( "Scope Monitor", Rect(600, 30, 800, 250) );
~scopeplotter.parent.alpha_(0.5);
~scopeplotter.plotMode_( \steps )
(
~scopeplotter.setProperties( \backgroundColor, Color.black,
	\plotColor, [
			Color.yellow, Color.blue, Color.green, Color.magenta, Color.cyan, Color.yellow,
			Color.red(0.9), Color.blue(0.9), Color.green(0.9), Color.magenta(0.9), Color.cyan(0.9), Color.yellow(0.9),
		] );
)
Tdef( \updatePlot ).stop;
(
Tdef( \updatePlot, {
	loop {
		defer{ ~scopeplotter.value_( [ ~xvals, ~xvals2, ~xvals3 ] ); };
		0.05.wait;
	}
}).play;
);

// [ ~xvals.first, ~xvals.last ]
