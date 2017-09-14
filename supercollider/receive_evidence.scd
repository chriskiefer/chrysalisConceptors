x.addExpected( 400, \evidence, 3 );

// get our evidence
OSCdef( \evidence1, { |data,time,src| ~evidences[0] = data[2] - data[1]; }, '/evidence/CL1' );
OSCdef( \evidence2, { |data,time,src| ~evidences[1] = data[2] - data[1]; }, '/evidence/CL2' );
OSCdef( \evidence3, { |data,time,src| ~evidences[2] = data[2] - data[1]; x.setData( 400, ~evidences ); }, '/evidence/CL3' );

x.nodes[151].action = { |data| ~jupyter2.sendMsg( "/input", 0, data[0]*250 ) };
x.nodes[152].action = { |data| ~jupyter2.sendMsg( "/input", 1, data[0]*250 ) };
x.nodes[153].action = { |data| ~jupyter2.sendMsg( "/input", 2, data[0]*250 ) };


Tdef( \lightUpdateDash, { loop{	~setAllDashes.value; 0.05.wait; } } ).play; // fixed rate
