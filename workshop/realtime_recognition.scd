// realtime recognition

// first load the network in python and set it up running there

// and now we assign an action to send the data in realtime to the python script:
n = NetAddr.new( "127.0.0.1", 57400 ); // python's address

// this is the maxvalue that we calculated in python
~maxValue = 0.0516839488978;

x.nodes[31].action = { |data| n.sendMsg( '/input', 0, (data[0]/~maxValue * 2 - 1); ); }; // second argument 0 is just an index; the code in python is made so that it can support multiple networks

// get our evidence
OSCdef( \evidence, { |data,time,src|  data.postln; }, '/evidence' );

// and x-vector
OSCdef( \x1, { |data,time,src|  data.postln; }, '/x/1' );

OSCdef( \x1 ).free;

// x.removeNode(41);
x.addExpected( 41, \evidence, 4 );

OSCdef( \evidence, { |data,time,src| x.setData( 41, data.copyToEnd(1) ) }, '/evidence' );

// start plotting with [Mon] button

// turn off autoscaling
x.nodes[41].datamonitor.plotter.findSpecs_(false);
x.nodes[41].datamonitor.plotter.specs_( [-50,60].asSpec );

// let's look at the difference!

// x.removeNode( 51 );
x.addExpected( 51, \eviDifference, 4 );
(
x.nodes[41].action = { |data|
	x.setData( 51, [
		data[1] - data[0], // difference between gesture2 and gesture1
		data[0] + data[3], // gesture1 and not gesture2
		data[1] + data[2], // gesture2 and not gesture1
		(data[0] + data[3]) - // gesture1 and not gesture2
		(data[1] + data[2]), // gesture2 and not gesture1
		// data[2] - data[3], // difference between not gesture1 and not gesture2
	] )
};
);
// start plotting with [Mon] button


// turn off autoscaling
x.nodes[51].datamonitor.plotter.findSpecs_(false);
x.nodes[51].datamonitor.plotter.specs_( [-40,10].asSpec );

// stop it!!!
n.sendMsg( "/exit" );