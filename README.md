# pyenf_extraction
Python based, ENF extraction from video/audio recordings. 

# Environments

Python Version Tested = 3.6.13
Environments used are mentioned in requirements.txt 

## Currently Testing 

Python = 3.9 is currently being tested with the libraries updated to their current version. 

- In the previous version 3.6.13, the scipy library usage was restricted to <= 1.2.0 since the function 'imresize' was used to resize array dimensions.
- The dependency is removed by replacing the function with Pillow library based resize function. 
- The updated pyenf module works for audio recordings. The video recordings are still being tested. 

# Usage

To test if the required python libraries are installed properly for audio recording based ENF estimation, run the following code with the input recording. A sample power recording is available in Rec_demo folder.
```
python pyenf.py --file INPUT_FILENAME --nominal NOMINAL_FREQUENCY
```

The defaults are already set to Rec_demo/power_recording.wav file and 60 Hz nominal frequency value. So, by simply running the program without any input should be enough to verify code functionality

```
python pyenf.py
```