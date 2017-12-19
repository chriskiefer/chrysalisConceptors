(
w = Window.new("gesture", Rect(0,0, 500, 500 ) );
~gestBox = StaticText.new( w, Rect( 0, 0, 500, 500 ) );
~gestBox.background_( Color.black );
~gestBox.align_( \center );
w.front;
w.onClose_( { Tdef( \indicateGesture ).stop; } );
);

// tune these two: (first value is the baseline difference; 2nd value is the maximum/minimum)
~diffSpec1 = [0, -1.5 ].asSpec;
~diffSpec2 = [0, 1 ].asSpec;

(
Tdef( \indicateGesture, {
	loop{
		var gesture1, gesture2, evidence;
		evidence = x.nodes[51].data[0]; // difference between the two gestures
		gesture1 = ~diffSpec1.unmap( evidence );
		gesture2 = ~diffSpec2.unmap( evidence );
		defer{
			~gestBox.background_( Color.new( gesture1, gesture2, 0 ) );
			if ( gesture1 > 0.2 ){
				~gestBox.string_( "gesture 1" );
			}{
				if ( gesture2 > 0.2 ){
					~gestBox.string_( "gesture 2" );
				}{
					~gestBox.string_( "no gesture" );
				};
			};
		};
		0.05.wait;
	};
}).play;
);
