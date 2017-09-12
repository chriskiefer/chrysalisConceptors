// generating patterns

// delta time of 0.05 feels nice on the eye
// when I play it slower, I need to interpolate

// [ time, targetlight*4 ]

(
~recordPattern = List.new;
Tdef( \lightPattern ).set( \delta, 0.05 );
Tdef( \lightPattern ).set( \pattern, [
	[0.5, [255,0,0,0] ],
	[0.5, [0,0,255,0] ],
	[0.5, [255,0,0,0] ],
	[0.5, [0,0,255,0] ],
	[1, [0,255,0,0] ],
	[1, [0,0,255,0] ],
	[0.5, [0,0,0,0] ],
]);
/*
Tdef( \lightPattern ).set( \pattern, [
	[0.5, [127,127,  0,0] ],
	[0.5, [127,  0,127,0] ],
	[0.5, [  0,127,127,0] ],
	[0.5, [  0,  0,127,0] ],
	[1,   [127,127,  0,0] ],
	[1,   [  0,127,  0,0] ],
	[0.5, [  0,  0,  0,0] ],
]);
*/
Tdef( \lightPattern, { |ev|
	var lights = [ 0,0,0,0];
	ev.pattern.do{ |it|
		var time = it[0];
		var target = it[1];
		var steps = (time / ev.delta).max(1);
		var stepsize = (target - lights) / (steps);
		[ steps, stepsize ].postln;
		steps.do{
			lights = (lights + stepsize).round(1).min(255).max(0);
			lights.postln;
			~recordPattern.add( lights );
			~microWashes.do{ |jt|
				jt.rgbw_(*lights);
			};
			~setAllLights.value;
			ev.delta.wait;
		};
		lights = target;
		~microWashes.do{ |jt|
			jt.rgbw_(*lights);
		};
		~recordPattern.add( lights );
		~setAllLights.value;
		ev.delta.wait;
	};
}).play;
);

~recordPattern.flop.asArray.plot;


(
Tdef( \playPattern ).set( \delta, 0.05 );
Tdef( \playPattern, { |ev|
	loop{
		~recordPattern.do{ |it|
			~microWashes.do{ |jt| jt.rgbw_(*it); };
			~setAllLights.value;
			ev.delta.wait;
		};
	};
}).play;
);

Tdef( \playPattern ).stop;

~recordPattern1 = ~recordPattern;
~recordPattern2 = ~recordPattern;