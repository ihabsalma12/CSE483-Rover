# CSE483-Sample-and-Return-Rover
## ASU ENG Fall 2022

Team members: Salma Hamed [19P8794] Madonna Magdy [19P2671] Noorhan Hatem [19P5821] Youssef Mahmoud Massoud [18P8814] repo link: https://github.com/ihabsalma12/CSE483-Rover/

### Prerequisites to Install
-python
-python-engineio version 3.13.2
-python-socketio version 4.6.1
-other python libraries as needed

## Run Instructions
There are three main modes of operation: normal autonomous, debugging, and test_dataset pipeline. Download the code as a ZIP file, then extract.

### Normal/ Debugging Autonomous Mode
Follow these steps closely. Please restart the steps since if one is missed, code might not work

#### Steps
0) Make sure perception.py debugger flag variable is set/ cleared. Close perception.py
1) Open terminal in folder 'code'. Then, open simulator, but don't select graphics yet.
2) Enter the following commands into the terminal: export PYTHONUNBUFFERED=true export QT_QPA_PLATFORM=offscreen This ensures python buffer is cleared and the output pipeline images are generated.
3) Enter the following command into the terminal to drive the simulator: python -u drive_rover.py
4) Select the graphics and input options, then choose "Autonomous mode" from the sim menu
5) When done, close the terminal and simulator.
If an older simulation model appears, close and restart simulator until the problem is fixed. If any other problems occur with terminal, run the file on jupyter notebook using the magic command %run driver_rover.py

Restart the steps to test the debugging modes.

### test_dataset Pipeline Output Images
#### Run generator.py (folder: 'code') on the terminal to generate the pipeline images of test_dataset/IMG
The pipeline images are saved to IMG2

To generate pipeline of recording of training mode, just change the path to output IMG instead of test_dataset/IMG. But remember, first manually create a folder called 'IMG2' where IMG and csv file is.

(We can further expand the pipeline images to include the rover world_map and positions, by reading the values from the csv file generated.)

Note: Empty the pipeline_realtime folder for every debugging run. Note: Empty the IMG2 folder for every run of generator.py for the same reason.
