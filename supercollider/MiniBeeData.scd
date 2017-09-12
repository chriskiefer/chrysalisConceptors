~configFile = "/home/nescivi/git/projects/chrysalis/minibee2osc/configuration_stretch.xml".resolveRelative;
~mbeePort = "/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DA01MBBJ-if00-port0";
// ~mbeePort = "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_A800f7Ue-if00-port0 ";
("minibee2osc -c" + ~configFile + "-s" + ~mbeePort + "-p" + NetAddr.langPort ).runInTerminal;

(
OSCdef( \minibeedata, { |data,time,src|  data.postln; }, '/minibee/data' );
OSCdef( \minibeeinfo, { |data,time,src| [data,time,src].postln; }, '/minibee/info' );
OSCdef( \minibeestatus, { |data,time,src| [data,time,src].postln; }, '/minibee/status' );
OSCdef( \minibeeprivate, { |data,time,src| [data,time,src].postln; }, '/minibee/private' );
);

~pydon = NetAddr.new( "127.0.0.1", 57600 );
// ~pydon.sendMsg( "/minibee/output", 4, 255, 255 );
// ~pydon.sendMsg( "/minibee/output", 4, 0, 0 );
// ~pydon.sendMsg( "/minibee/custom", 1, 0, 10, 0, 90, 2 );
// ~pydon.sendMsg( "/minibee/reset", 2 );
// ~pydon.sendMsg( "/minibee/configuration", 5, 4 );
// ~pydon.sendMsg( "/minihive/configuration/save", "config_chrysalis.xml" );

x = x ? SWDataNetwork.new;

(
x.addExpected( 101, \stretch1, 2 );
x.addExpected( 111, \acceleration1, 3 );
x.addExpected( 102, \stretch2, 2 );
x.addExpected( 112, \acceleration2, 3 );
x.addExpected( 103, \stretch3, 2 );
x.addExpected( 113, \acceleration3, 3 );
);
(
OSCdef( \minibeedata, { |data,time,src|
	// data.copyRange(2,6).postln;
	// src.postln;
	if ( data[1] == 1 ){
		x.setData( \stretch1, data.copyRange( 2, 3 ) );
		x.setData( \acceleration1, data.copyToEnd( 4 ) );
	};
	if ( data[1] == 2 ){
		x.setData( \stretch2, data.copyRange( 2, 3 ) );
		x.setData( \acceleration2, data.copyToEnd( 4 ) );
	};
	if ( data[1] == 3 ){
		x.setData( \stretch3, data.copyRange( 2, 3 ) );
		x.setData( \acceleration3, data.copyToEnd( 4 ) );
	};
}, '/minibee/data', NetAddr.new( "127.0.0.1", 57600 ) );
);

(
x.addExpected( 300, \buttons, 5 );
x.add( \duim, [300,3] );
x.add( \wijs, [300,4] );
x.add( \middel, [300,2] );
x.add( \ring, [300,1] );
x.add( \pink, [300,0] );
OSCdef( \minibeedata5, { |data,time,src|
	// data.postln;
	x.setData( \buttons, 1- data.copyToEnd( 2 ) );
}, '/minibee/data', NetAddr.new( "127.0.0.1", 57600 ), argTemplate: [5] );
);

(
x.addExpected( 310, \vibro, 6 );
OSCdef( \minibeedata4, { |data,time,src|
	// data.postln;
	x.setData( \vibro, data.copyToEnd( 2 ) );
}, '/minibee/data', NetAddr.new( "127.0.0.1", 57600 ), argTemplate: [4] );
);

// data range: 0.6 - 0.95
// ~stretchSpec = [ 0.95, 0.6 ].asSpec;

//~stretchSpec.unmap( 0.95 )

(
~stdStretchNodes = [101,102,103].collect{ |id|
	x.nodes[id].createBus(s);
	StdDevNode.new( id + 20, x, x.nodes[id].bus, s ).set( \length, 200 );
};
fork{
	1.0.wait;
	~stdStretchNodes.do{ |it| it.start };
};
[101,102,103].do{ |id,i|
	x.add( ("stretch"++(i+1)++"std").asSymbol, id+20 );
};
);

(
~meanAcceleroNodes = [111,112,113].collect{ |id|
	x.nodes[id].createBus(s);
	MeanNode.new( id + 20, x, x.nodes[id].bus, s ).set( \length, 200 );
};
fork{
	1.0.wait;
	~meanAcceleroNodes.do{ |it| it.start };
};
[111,112,113].do{ |id,i|
	x.add( ("acc"++(i+1)++"mean").asSymbol, id+20 );
};
);

(
([111,112,113]+20).do{ |id,i|
	x.addExpected( id+10, ("acc"++(i+1)++"dev").asSymbol, 3 );
	x.nodes[id].action = { |data| x.setData( id+10, data - x.nodes[id-20].data ); };
};
);

(
([111,112,113]+30).do{ |id,i|
	x.addExpected( id+10, ("acc"++(i+1)++"devSum").asSymbol, 1 );
	x.nodes[id].action = { |data| x.setData( id+10, (data*data).sum.sqrt.bubble ); };
};
);

// x.makeNodeGui