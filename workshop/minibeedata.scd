// this is the file that distributes the minibee data to different computers (this will run on Marije's computer)
(
~configFile = "configuration_minibee.xml".resolveRelative;
~mbeePort = "/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DA01MBBJ-if00-port0";
("minibee2osc -c" + ~configFile + "-s" + ~mbeePort + "-p" + NetAddr.langPort ).runInTerminal;

OSCdef( \minibeedata, { |data,time,src|  data.postln; }, '/minibee/data' );
OSCdef( \minibeeinfo, { |data,time,src| [data,time,src].postln; }, '/minibee/info' );
OSCdef( \minibeestatus, { |data,time,src| [data,time,src].postln; }, '/minibee/status' );
OSCdef( \minibeeprivate, { |data,time,src| [data,time,src].postln; }, '/minibee/private' );
);

// netadresses to use:
(
~netAddresses = IdentityDictionary[
	\marije -> NetAddr.new( "127.0.0.1", 57120 ),
	\chris -> NetAddr.new( "127.0.0.1", 57120 ),
	\alice -> NetAddr.new( "127.0.0.1", 57120 ),
	\rutz -> NetAddr.new( "127.0.0.1", 57120 ),
];
);

// forward the data:
OSCdef( \minibee1, { |data,time,src| ~netAddresses[ \marije ].sendMsg( *data ); }, '/minibee/data', argTemplate: [ 1 ] );
OSCdef( \minibee2, { |data,time,src| ~netAddresses[ \marije ].sendMsg( *data ); }, '/minibee/data', argTemplate: [ 2 ] );
OSCdef( \minibee3, { |data,time,src| ~netAddresses[ \marije ].sendMsg( *data ); }, '/minibee/data', argTemplate: [ 3 ] );
OSCdef( \minibee4, { |data,time,src| ~netAddresses[ \marije ].sendMsg( *data ); }, '/minibee/data', argTemplate: [ 4 ] );
OSCdef( \minibee5, { |data,time,src| ~netAddresses[ \marije ].sendMsg( *data ); }, '/minibee/data', argTemplate: [ 5 ] );
