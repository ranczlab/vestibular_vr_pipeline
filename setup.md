## [BONSAI-RX](https://bonsai-rx.org/)
This is the software environment used for hardware and experimental control as well as recording.  
:information_source: [Discord channel](https://discord.com/channels/722881880122327101/727928858757562430) for support and general discussions.

### vestibular-VR SETUP
The repo is [here](https://github.com/neurogears/vestibular-vr), managed by NeuroGears.  
This contains all experimental Bonsai workflows and development. Experimental protocol schemas are .yaml files, living in the [src](https://github.com/neurogears/vestibular-vr/tree/main/src) folder. 

### [BonVision](https://bonvision.github.io/)  
A Bonsai package for generating VR visual scenes. Can create simple objects and textures. Can take blender or CAD (OBJ) scenes for more complex scenarios.  
  
## HARP
This is the [hardware platform](https://harp-tech.org/articles/about.html) used for control and most recordings.  
  
### [Clock synchroniser](https://github.com/harp-tech/device.clocksynchronizer)
Used in generator mode to syncronise accross HARP devices.  
  
### [H1 board](https://github.com/harp-tech/device.vestibularH1)
HARP board located on the rotating platform with multiple input and output registers. Used for reading optical flow sensor and lick detection data, triggering eye-camera and reward. 
[list of registers - incomplete?](https://github.com/harp-tech/device.vestibularH1/blob/main/Firmware/VestibularH1/registers.xls). Registers can be read, write or event, with different payloads (e.g. unsigned 8, float 64,...)  
  
### [H2 board](https://github.com/harp-tech/device.vestibularH2)  
HARP board located off the rotating platform with multiple input and output registers, chiefly responsible for motor control. Used for motor driver control, motor encoder and Hall (homing) sensor reading. 
[list of registers - incomplete?](https://github.com/harp-tech/device.vestibularH2/blob/main/Firmware/VestibularVrH2/registers.xls). Registers can be read, write or event, with different payloads (e.g. unsigned 8, float 64,...)    

## [ONIX](https://open-ephys.org/onix)
This is the hardware platform used for neuropixels recording and other analogue and digital I/O.    
:information_source: [Discord channel](https://discord.com/channels/932637696503971850/946378096288882718) for support and general discussions.
[Bonsai package](https://github.com/open-ephys/bonsai-onix1)






