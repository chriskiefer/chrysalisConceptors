~jupyterAcc = NetAddr.new( "127.0.0.1", 57402 );
~jupyterAcc2 = NetAddr.new( "127.0.0.1", 57403 );

/*
x[\buttonAcc].action = MFunc.new;

x[\buttonAcc].action.add( \forward2network, { |data| ~jupyterAcc.sendMsg( *(['/accelero'] ++ (data - 0.5 * 2 / 0.21) ) ) } );

x[\buttonAcc].action.add( \forward2network2, { |data| ~jupyterAcc2.sendMsg( *(['/accelero'] ++ (data - 0.5 * 2 / 0.21) ) ) } );
*/

x[\butAccDev].action = MFunc.new;

/*
x[\butAccDev].action.add( \forward2network, { |data| ~jupyterAcc.sendMsg( *(['/accelero'] ++ (data * 5) ) ) } );

x[\butAccDev].action.add( \forward2network2, { |data| ~jupyterAcc2.sendMsg( *(['/accelero'] ++ (data *5) ) ) } );

x[\butAccDev].action.remove( \forward2network );
x[\butAccDev].action.remove( \forward2network2 );
*/

(
Tdef( \sendData ).set( \deltaTime, 0.05 );
Tdef( \sendData, { |ev|
	loop{
		// var data = x[\buttonAcc].data - 0.5 * 5;
		// ~jupyterAcc.sendMsg( *(['/accelero'] ++ data ) );
		// ~jupyterAcc2.sendMsg( *(['/accelero'] ++ data ) );
		var data = x[\butAccDev].data * 10;
		var data2 = x[\butAccMean].data - 0.5 * 20;
		~jupyterAcc.sendMsg( *(['/accelero'] ++ data ++ data2 ) );
		~jupyterAcc2.sendMsg( *(['/accelero'] ++ data ++ data2 ) );
		x.setData( 600, data ++ data2 );
		// ~jupyterAcc.sendMsg( *(['/accelero'] ++ data  ) );
		// ~jupyterAcc2.sendMsg( *(['/accelero'] ++ data  ) );
		ev.deltaTime.wait;
	}
});
);

x.addExpected( 600 )

Tdef( \sendData ).play;
x.addExpected( 500 )


OSCdef( \inputValues, { |data,time,src| data.postln; }, '/inputvalues' );

OSCdef( \outputValues, { |data,time,src| x.setData( 500, data.copyToEnd(1) ) }, '/output' );
(
x.addExpected( 510, \deltaTime );
~prevOutTime = 0;
OSCdef( \outputValues, { |data,time,src|
	x.setData( 510, [ time - ~prevOutTime ] );
	~prevOutTime = time;
	x.setData( 501, data.copyToEnd(1) );
	// x.setData( 502, data.copyRange(1,3) );
}, '/output' );
)

x.addExpected( 501, 'netOut' );
x.addExpected( 502, 'netOut2' );
x.addExpected( 503, 'diffs' );

x.removeNode( 500 );

x.nodes[501].action = MFunc.new;
(
x.nodes[501].action.add( \diffResults, { |data|
	x.setData( 502, [
		data[0] - data[3],
		data[1] - data[3],
		data[2] - data[3],
	])
})
);

(
x.nodes[501].action.add( \diff3, { |data|
	x.setData( 511, [
		data[3] + (data[0] + data[1] + data[2] * 2 / 3)
	])
})
);

x.addExpected( 511, \diff3 );

x.nodes[502].action = MFunc.new;

x.nodes[502].action.remove(\diffResults);
(
x.nodes[501].action.add( \diffResults, { |data|
	x.setData( 503, [
		data[0]*2 - data[1] - data[2],
		data[1]*2 - data[0] - data[2],
		data[2]*2 - data[1] - data[0],
	])
})
);

(
x.nodes[501].action.add( \meMinMean, { |data|
	// data.size.postln;
	x.setData( 505, [
		data[0] - (data.copyFromStart(2).sum / 3),
		data[1] - (data.copyFromStart(2).sum / 3),
		data[2] - (data.copyFromStart(2).sum / 3),
	])
})
)

x.addExpected( 505 )

x.nodes[505].createBus( s );
~meanNodeConc = MeanNode.new( 506, x, x.nodes[505].bus, s );
~meanNodeConc.set( \length,  200 );
~meanNodeConc.start;

x.nodes[506].action = MFunc.new;
x.nodes[506].action.add( \devConc, { |data| x.setData( 507, x.nodes[505].data - data ) } );
x.addExpected( 507 )

x.nodes[506].action.add( \maxIndex, { |data| x.setData( 504, [ data.maxIndex] ); } );
x.addExpected( 504 );

b = Bus.new( \control, 100, 500, s );

b.getn( 500, { |d| defer{ d.plot } } );

~busVals.plot;



x.nodes[503].action = MFunc.new;
x.nodes[503].action.remove( \maxIndex );
x.nodes[503].action.add( \maxIndex, { |data| x.setData( 504, [ data.maxIndex] ); } );

x.addExpected( 504, \maxindex );

x.makeNodeGui

~recIndex = 0;
~recOn = false;
x[\wijs].action = MFunc.new;

(
x[\wijs].action.add( \recOnOff, { |data| if ( data == 1 ){
	if ( ~recOn ){
		~jupyterAcc.sendMsg( "/record/off" ); ~recIndex = (~recIndex + 1).mod(3);
		~recOn = false;
	}{
		~recOn = true;
		~jupyterAcc.sendMsg( "/record/on", ~recIndex ); }
} } );
);

x[\wijs].action.add( \recOnOff, { |data| if ( data == 1 ){ ~jupyterAcc.sendMsg( "/record/on", ~recIndex ); }{} } );


~jupyterAcc.sendMsg( "/record/on", 0 );
~jupyterAcc.sendMsg( "/record/off" );


OSCdef( \evidence, { |data,time,src| ~evidences[0] = data[2] - data[1]; }, '/evidence' );


~jupyterAcc.sendMsg( "/exit" );
~jupyterAcc2.sendMsg( "/exit" );
