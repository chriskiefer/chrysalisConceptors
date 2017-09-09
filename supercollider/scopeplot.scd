~scopeplotter = Plotter.new( "Scope Monitor", Rect(600, 30, 800, 250) );

(
Tdef( \updatePlot, {
	loop {
		defer{ ~scopeplotter.value_( ~xvals.dup(10).flatten ); };
		0.05.wait;
	}
}).play;
);

[ ~xvals.first, ~xvals.last ]

