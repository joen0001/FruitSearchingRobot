# G107_Intelligent_Robots
oi don't copy us x

# Launch brick world environment
source ~/catkin_ws/devel/setup.bash

roslaunch penguinpi_gazebo ECE4078_brick.launch

# Launch simulator environment
source ~/catkin_ws/devel/setup.bash

roslaunch penguinpi_gazebo ECE4078.launch

# Spawn objects
source ~/catkin_ws/devel/setup.bash

rosrun penguinpi_gazebo scene_manager.py -l map1.txt


# Reset git repository
rm -rf G107_Intelligent_Robots
git clone https://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@github.com/nikkihobman/G107_Intelligent_Robots


# eval file
python3 SLAM_eval.py TrueMap.txt lab_output/slam.txt

# connecting to robot
password: egb439123

python3 operate.py --ip 192.168.50.1 --port 8080
