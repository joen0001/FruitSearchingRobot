#prelab
- clear workspace
- have emails open ready to download true map
- have moodle open ready to submit 'slam.txt' and 'targets.txt'
- have moodle open ready to download our submitted FinalDemo folder

SIMULATION

#STEP 1: load environment and map
#source ~/catkin_ws/devel/setup.bash
#roslaunch penguinpi_gazebo ECE4078.launch
#
#source ~/catkin_ws/devel/setup.bash
#rosrun penguinpi_gazebo scene_manager.py -l TrueMap.txt


#STEP 2: run operate
#LIVEDEMO
#cd G107_Intelligent_Robots
#cd Milestone05
#
#python3 operate.py

#STEP 3: create map
- teleoperate to find markers
- save map ('S')
- take images of fruits (no two fruits of the same type in the same image)
- save images ('P' ... 'N')
- exit slam


#STEP 4: generate targets.txt
#python3 TargetPoseEst.py

#STEP 5: search for fruits
- reset robot pose to 0
#python3 navigation.py
#use navigation_dynamics.py if not working
- enter waypoints (x,y)
- find 3 fruits

#STEP 6: submit files
- upload slam.txt and targets.txt to moodle

#FINISH SIM HERE ----------------
#START PHYSICAL HERE ------------

#STEP 1: change calibration codes


#STEP 2: load environment and map
#source ~/catkin_ws/devel/setup.bash
#roslaunch penguinpi_gazebo ECE4078.launch
#
#source ~/catkin_ws/devel/setup.bash
#rosrun penguinpi_gazebo scene_manager.py -l TrueMap.txt



#STEP 3: run operate
#cd G107_Intelligent_Robots
#cd Milestone05
#
#python3 operate.py

#STEP 4: create map
- teleoperate to find markers
- save map ('S')
- take images of fruits (no two fruits of the same type in the same image)
- save images ('P' ... 'N')
- exit slam


#STEP 5: generate targets.txt
#python3 TargetPoseEst.py

#STEP 6: search for fruits
- reset robot pose to 0
#python3 navigation.py
#use navigation_dynamics.py if not working
- enter waypoints (x,y)
- find 3 fruits


#step 7: submit files
- upload slam.txt and targets.txt to moodle


