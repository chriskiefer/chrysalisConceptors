~lightSpec = [1,255,\linear,1].asSpec;
~freqsSpec = [3, 15, \exponential ].asSpec;
~ffreqsSpec = [500, 800, \exponential ].asSpec;

~radiusSpec = [0.7,1.3].asSpec;
~evidenceSpec = [-3,3].asSpec;
~diffSpec1a = [0, -3 ].asSpec;
~diffSpec1b = [0, 3 ].asSpec;
~diffSpec2a = [0, -3 ].asSpec;
~diffSpec2b = [0, 3 ].asSpec;
~diffSpec3a = [0, -3 ].asSpec;
~diffSpec3b = [0, 3 ].asSpec;

// Tdef( \lightUpdate, { loop{	~setAllDashes.value; 0.025.wait; } } ).play; // fixed rate

(
x.nodes[400].slots[0].action = { |v|
	v = v + 1;
	// v.postln;
	~colorDashes[0].do{ |it| it.red_( v.specMap( ~diffSpec1a, ~lightSpec ) ) };
	~colorDashes[1].do{ |it| it.red_( v.specMap( ~diffSpec1a, ~lightSpec ) ) };

	~colorDashes[0].do{ |it| it.green_( v.specMap( ~diffSpec1b, ~lightSpec ) ) };
	~colorDashes[1].do{ |it| it.green_( v.specMap( ~diffSpec1b, ~lightSpec ) ) };

	~waveSynthCL1.set(
		\freq, v.specMap( ~evidenceSpec, ~freqsSpec ),
		\ffreq, v.specMap( ~evidenceSpec, ~ffreqsSpec )
	);

	// ~jupyter.sendMsg( "/spectralradius", 0, v.specMap( ~evidenceSpec, ~radiusSpec ); );

};
);

(
x.nodes[400].slots[1].action = { |v|
	v = v - 0.75;
	// v.postln;
	~colorDashes[2].do{ |it| it.red_( v.specMap( ~diffSpec2a, ~lightSpec ) ) };
	~colorDashes[3].do{ |it| it.red_( v.specMap( ~diffSpec2a, ~lightSpec ) ) };

	~colorDashes[2].do{ |it| it.green_( v.specMap( ~diffSpec2b, ~lightSpec ) ) };
	~colorDashes[3].do{ |it| it.green_( v.specMap( ~diffSpec2b, ~lightSpec ) ) };

	~waveSynthCL2.set(
		\freq, v.specMap( ~evidenceSpec, ~freqsSpec ) * 4/5,
		\ffreq, v.specMap( ~evidenceSpec, ~ffreqsSpec ) * 4/5
	);

};
);

(
x.nodes[400].slots[2].action = { |v|
	// v = v + 1;
	// v.postln;
	~colorDashes[4].do{ |it| it.red_( v.specMap( ~diffSpec3a, ~lightSpec ) ) };
	~colorDashes[5].do{ |it| it.red_( v.specMap( ~diffSpec3a, ~lightSpec ) ) };

	~colorDashes[4].do{ |it| it.green_( v.specMap( ~diffSpec3b, ~lightSpec ) ) };
	~colorDashes[5].do{ |it| it.green_( v.specMap( ~diffSpec3b, ~lightSpec ) ) };

	~waveSynthCL3.set(
		\freq, v.specMap( ~evidenceSpec, ~freqsSpec )*5/4,
		\ffreq, v.specMap( ~evidenceSpec, ~ffreqsSpec )*5/4
	);

};
);



/*

x.nodes[51].slots[0].action = { |v| }
x.nodes[51].slots[3].action = { |v| }


x.nodes[41].slots[0].action = { |v| }
x.nodes[41].slots[1].action = { |v| }

~evSpec1 = [0, 20 ].asSpec;

(
x.nodes[41].slots[0].action = { |v|
	~colorDashes.do{ |it| it.red_( v.specMap( ~evSpec1, ~lightSpec ) ) };
};
x.nodes[41].slots[1].action = { |v|
	~colorDashes.do{ |it| it.blue_( v.specMap( ~evSpec1, ~lightSpec ) ) };
};
)

*/