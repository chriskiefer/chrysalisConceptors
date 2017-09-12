~colorDashes = [50,60,70,80, 90,100,110,120].collect{ |id| DMXAccentQuad.new( id ); };
~setAllDashes = { ~colorDashes.do{ |it| ~lights.merge( it.subCue ); }; if ( ~dmx.device.notNil ){ ~dmx.setCue; }; };
~colorDashes.do{ |it| it.master_(255); it.program_(0); };
~colorDashes.do{ |it| it.red_(255) };
~colorDashes.do{ |it| it.rgba_(255,255,0,0) };

~colorDashes.do{ |it| it.rgba_(0,0,0,0) };
~colorDashes[0].rgba_(255,255,255,255);
~colorDashes[1].rgba_(255,255,255,255);
~colorDashes[2].rgba_(255,255,255,255);
~colorDashes[3].rgba_(255,255,255,255);
~colorDashes[4].rgba_(255,255,255,255);
~colorDashes[5].rgba_(255,255,255,255);
~colorDashes[6].rgba_(255,0,255,255);
~colorDashes[7].rgba_(255,255,255,255);
~setAllDashes.value;

(
Tdef( \loopLight, {
	var lights = [ 255,0,0,0];
	loop{
		~colorDashes.do{ |jt|
			~colorDashes.do{ |it| it.rgba_(0,0,0,0) };
			// jt.rgba_(255,255,255,255);
			// jt.rgba_(0,0,0,255);
			// jt.rgba_(0,0,255,0);
			// jt.rgba_(0,255,0,0);
			jt.rgba_(*lights);
			~setAllDashes.value;
			0.05.wait;
		};
		~colorDashes.do{ |it| it.rgba_(0,0,0,0) };
		~setAllDashes.value;
		~microWashes.do{ |jt|
			~microWashes.do{ |it| it.rgbw_(0,0,0,0) };
			// jt.rgba_(255,255,255,255);
			// jt.rgba_(0,0,0,255);
			// jt.rgba_(0,0,255,0);
			// jt.rgba_(0,255,0,0);
			jt.rgbw_(*lights);
			~setAllLights.value;
			0.05.wait;
		};
		~microWashes.do{ |it| it.rgbw_(0,0,0,0) };
		~setAllLights.value;
		lights = lights.rotate(1);
	}
}).play;
);

Tdef( \loopLight ).stop;

a.master_(255);
a.program_(0);


a.red_(255);
a.green_(255);
a.blue_(255);
a.amber_(255);

a.rgba_(0,0,0,0);

a.strobe_(0);


~lights.merge( a.subCue ); if ( ~dmx.device.notNil ){ ~dmx.setCue; };
