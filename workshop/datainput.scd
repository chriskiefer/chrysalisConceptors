// SC code: simple interactive coding: just press [shift]+[enter] while on a line of code to execute it

// To install:
Quarks.fetchDirectory;
Quarks.install( "DataNetwork" );

// Let Marije know which IP-address you have:
NetAddr.myIP;

// receiving data via osc (forwarded from Marije's computer)
OSCdef( \minibeedata, { |data,time,src|  data.postln; }, '/minibee/data' );

// a SWDataNetwork is a hub for data with lots of useful extra functions like realtime plotting, simple data analysis, etc.
x = SWDataNetwork.new;
x.addExpected( 1, \acceleration, 3 ); // create a node with ID=1, name=\acceleration and 3 data streams

OSCdef( \minibeedata, { |data,time,src| x.setData(1, data.copyToEnd(2) ); }, '/minibee/data' ); // redirect the incoming OSC data to the node we made.

x.makeNodeGui; // makes a GUI of the datanetwork.
// click on the button [ > ] to see the individual datastreams - click on [ Mon ] to have a realtime plot

// now we will create the mean of the accelerometer data, we use a UGen on the server for this so we need to boot the server:

s.boot;

x.nodes[1].createBus(s); // create a bus on the server and redirect the data from node 1 (our accelerometer data) to it

~meanAcceleroNode = MeanNode.new( 11, x, x.nodes[1].bus, s ).set( \length, 200 ); // create a node that calculates the mean over 200 samples of kr-rate data; in the background this creates a synth on the server, and sends the data back to the datanetwork to node 11.
~meanAcceleroNode.start; // start the synth

// in the GUI we now see a second node 11


(
// now we want to look at the deviation from the mean of the accelerometer data
x.addExpected( 21, \accDev, 3 );
x.nodes[11].action = { |data| x.setData( 21, data - x.nodes[1].data ); };
);

(
// to combine the deviation data of all three axes, we take the power of two of each axis, sum the three axes and take the root of that
x.addExpected( 31, \accDevSum, 1 );
x.nodes[21].action = { |data| x.setData( 31, (data*data).sum.sqrt.bubble ); };
);

// now we want to record the data and write it to file

// task to record a 1-second gesture at 30 Hz, with a 3-second count down
(
SynthDef( \countDown, { Out.ar( 0, SinOsc.ar( \freq.kr(300)*[1,2,3.5,7], 0, 0.1/4 ).sum*EnvGen.kr(Env.perc,timeScale: \dur.kr(0.2),doneAction:2) ) }).add;
~gesture = Array.new;
Tdef( \recordGesture, {
	"count down to make gesture!".postln;
	3.reverseDo{ |it,i|
		("..."++it).post;
		Synth.new( \countDown, [\freq, [300,300,450].at( i ), \dur, 0.2 ]  );
		1.0.wait;
	};
	Synth.new( \countDown, [\freq, 600, \dur, 1 ]  );
	"...go!!".postln;
	30.do{|i|
		~gesture = ~gesture.add( x.nodes[31].data[0] );
		(1/30).wait;
	};
	("...recorded gesture" + ~counter).postln;
});
);

(
~counter = 0;
// function to write to file in a format that can be read into python:
~writeGestureToFile = { |gesture|
	var file = File.new( ("gesturedata_" ++ ~counter ++ ".txt").resolveRelative, "w" );
	var string = gesture.asCompileString;
	file.write( string.replace( "[ ", "" ).replace( " ]", "" ).replace( ", ", " " ) );
	file.close;
	~counter = ~counter + 1;
}
);

// now get ready to make the gesture!
~gesture = Array.new; Tdef( \recordGesture ).play;

// plot
~gesture.plot;
// write to file
~writeGestureToFile.value( ~gesture );

// make another one:
~gesture = Array.new; Tdef( \recordGesture ).play;

// plot
~gesture.plot;
// write to file
~writeGestureToFile.value( ~gesture );


// NOW we go to the jupyter notebook to look at the data there and see whether our algorithm can distinguish the two gestures!

