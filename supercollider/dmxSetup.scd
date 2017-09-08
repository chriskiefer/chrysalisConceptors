(
// This creates an instance of the DMX class (see the DMX quark):
~dmx = DMX.new;
// create a DMX device (the argument is the serial path at which it can be reached; see SerialPort.devices)
// ~dmxSerialDevice = "/dev/tty.usbserial-ENSMO7IT"; /// ADAPT TO SERIAL DEVICE!
~dmxSerialDevice = "/dev/serial/by-id/usb-ENTTEC_DMX_USB_PRO_EN083820-if00-port0";
// ~dmxSerialDevice = "/dev/serial/by-id/usb-ENTTEC_DMX_USB_PRO_Mk2_EN183075-if00-port0"; // Jaimes

SerialPort.devicePattern = "/dev/serial/by-id/usb-*"; // linux usb serial
// SerialPort.devices;

Tdef( \openSerial,{
	var mydev;
	while ( { mydev.isNil },{
		mydev = SerialPort.devices.detect{ |it| ~dmxSerialDevice == it }; // this needs an adjustment
        "Found serial devices for DMX %\n".postf( mydev );
		if ( mydev.notNil ){
			if ( ~dmxDevice.notNil ){
				if ( ~dmxDevice.isOpen.not ){
					~dmxDevice = EntTecDMXUSBPro.new( ~dmxSerialDevice );
					~dmx.device = ~dmxDevice;
				}
			}{ // q is nil
				~dmxDevice = EntTecDMXUSBPro.new( ~dmxSerialDevice );
				~dmx.device = ~dmxDevice;
			};
			~dmxDevice.doneAction = { Tdef( \openSerial ).play; };
			~dmx.device = ~dmxDevice;
		}{
			// no devices found
			// probe every three seconds;
			3.0.wait;
		};
	} );
});
Tdef( \openSerial ).play;
// set the device for DMX
//~dmx.device = ~dmxDevice;
ShutDown.add( {	Tdef( \openSerial ).stop; if ( ~dmxDevice.notNil ){ ~dmxDevice.close; } });
);

(
~lights = DMXCue.new;
~lights.mode = \integer;
//~lights.mode = \float;

~dmx.currentCue = ~lights;
if ( ~dmx.device.notNil ){
	~dmx.setCue;
};
);

(
~microWashes = [40,10,20,30].collect{ |it| DMXMicroWash.new( it ); };

~microWashes.do{ |mw|
	mw.action = { |subcue| ~lights.merge( subcue ); if ( ~dmx.device.notNil ){ ~dmx.setCue; }; };
};
~microWashes.do{ |mw|
	mw.master_( 255 );
	mw.mode_( 0 );
	mw.strobe_( 0 );
	mw.rgbw_( 0,0,0,0 );
};

~microWashes.do{ |mw|
	mw.action = nil;
};

~setAllLights = { ~microWashes.do{ |it| ~lights.merge( it.subCue ); }; if ( ~dmx.device.notNil ){ ~dmx.setCue; }; };
);



/*
/// TESTING

~microWashes[0]

~microWashes[0].rgbw_(255,0,0,0); ~setAllLights.value;
~microWashes[0].rgbw_(255,0,0,0); ~setAllLights.value;

~microWashes[0].macros_(0); ~setAllLights.value;
~microWashes[0].strobe_(1); ~setAllLights.value;
~microWashes[0].rgbw_(255,0,0,0); ~setAllLights.value;
~microWashes[0].rgbw_(0,255,0,0); ~setAllLights.value;
~microWashes[0].rgbw_(0,0,255,0); ~setAllLights.value;
~microWashes[0].rgbw_(0,0,0,255); ~setAllLights.value;
~microWashes[0].rgbw_(255,255,255,255); ~setAllLights.value;

~microWashes[0].rgbw_(0,0,0,0); ~setAllLights.value;
~microWashes[1].rgbw_(0,0,0,0); ~setAllLights.value;
~microWashes[2].rgbw_(0,0,0,0); ~setAllLights.value;
~microWashes[3].rgbw_(0,0,0,0); ~setAllLights.value;

~microWashes[0].rgbw_(255,0,0,0); ~setAllLights.value;
~microWashes[1].rgbw_(255,0,0,0); ~setAllLights.value;
~microWashes[2].rgbw_(255,0,0,0); ~setAllLights.value;
~microWashes[3].rgbw_(255,0,0,0); ~setAllLights.value;
~microWashes[3].rgbw_(255,255,255,255); ~setAllLights.value;

~microWashes.do{ |mw| mw.rgbw_(0,0,0,0); }; ~setAllLights.value;

~microWash1.rgbw_( 0, 0, 0, 0 );
~microWash2.rgbw_( 0, 0, 0, 0 );
~microWash3.rgbw_( 0, 0, 0, 0 );
~microWash4.rgbw_( 0, 0, 0, 0 );

~microWash1.rgbw_( 255, 0, 0, 0 );
~microWash2.rgbw_( 255, 0, 0, 0 );
~microWash3.rgbw_( 255, 0, 0, 0 );
~microWash4.rgbw_( 255, 0, 0, 0 );

~microWash1.rgbw_( 0, 255, 0, 0 );
~microWash2.rgbw_( 0, 255, 0, 0 );
~microWash3.rgbw_( 0, 255, 0, 0 );
~microWash4.rgbw_( 0, 255, 0, 0 );

~microWash1.rgbw_( 0, 0, 255, 0 );
~microWash2.rgbw_( 0, 0, 255, 0 );
~microWash3.rgbw_( 0, 0, 255, 0 );
~microWash4.rgbw_( 0, 0, 255, 0 );

~microWash1.rgbw_( 0, 0, 0, 255 );
~microWash2.rgbw_( 0, 0, 0, 255 );
~microWash3.rgbw_( 0, 0, 0, 255 );
~microWash4.rgbw_( 0, 0, 0, 255 );
*/