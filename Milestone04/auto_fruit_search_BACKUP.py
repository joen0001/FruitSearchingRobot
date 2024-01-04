# M4 - Autonomous fruit searching

# basic python packages
import sys, os
import cv2
import numpy as np
import json
import argparse
import time

# import SLAM components
#Might need to fix this later 
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco

# import utility functions
sys.path.insert(0, "util")
from pibot import PenguinPi
import measure as measure
from operate import Operate


def read_true_map(fname):
    """Read the ground truth map and output the pose of the ArUco markers and 3 types of target fruit to search

    @param fname: filename of the map
    @return:
        1) list of target fruits, e.g. ['apple', 'pear', 'lemon']
        2) locations of the target fruits, [[x1, y1], ..... [xn, yn]]
        3) locations of ArUco markers in order, i.e. pos[9, :] = position of the aruco10_0 marker
    """
    with open(fname, 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_true_pos = []
        aruco_true_pos = np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)
            y = np.round(gt_dict[key]['y'], 1)

            if key.startswith('aruco'):
                if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos


def read_search_list():
    """Read the search order of the target fruits

    @return: search order of the target fruits
    """
    search_list = []
    with open('search_list.txt', 'r') as fd:
        fruits = fd.readlines()

        for fruit in fruits:
            search_list.append(fruit.strip())

    return search_list


def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order

    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    print("Search order:")
    n_fruit = 1
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        n_fruit += 1


# Waypoint navigation
# the robot automatically drives to a given [x,y] coordinate
# additional improvements:
# you may use different motion model parameters for robot driving on its own or driving while pushing a fruit
# try changing to a fully automatic delivery approach: develop a path-finding algorithm that produces the waypoints
def drive_to_point(waypoint, robot_pose):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    wheel_vel = 30 # tick
    x_des = waypoint[0]
    y_des = waypoint[1]
    
    x_robot = robot_pose[0]
    y_robot = robot_pose[1]
    theta_robot = robot_pose[2]
    
    x_dif = x_des - x_robot
    y_dif = y_des - y_robot
  
    # turn towards the waypoint
    alpha = np.arctan2(y_dif, x_dif) #bound between [-pi,pi]
    
    # if angle difference negative, convert [0,360]
    #if alpha < 0:
    #    alpha = 2*np.pi + alpha
    
    # if robot angle greater than 360, convert
    #if theta_robot > 2*np.pi:
    #    while theta_robot > 2*np.pi:
    #        theta_robot = theta_robot - 2*np.pi
            
    # if robot angle less than 0, convert 
    #if theta_robot < 0:
    #    while theta_robot < 0:
    #        theta_robot = 2*np.pi + theta_robot
    
    #if alpha < theta_robot:
    #    add_term = 2*np.pi - theta_robot
    #    theta_des = alpha + add_term
    #else:
    #    theta_des = alpha - theta_robot
    
    #NEW PART
    #first bound to [0,360]
    if theta_robot > 2*np.pi:
        while theta_robot > 2*np.pi:
            theta_robot = theta_robot - 2*np.pi
    if theta_robot < 0:
        while theta_robot < 0:
            theta_robot = 2*np.pi + theta_robot
    
    #covert to [-pi,pi]
    if theta_robot > np.pi:
        theta_robot = theta_robot - 2*np.pi
    
    #alpha and theta_robot now in [-pi,pi]
    
    #CALC
    if (alpha >= 0) and (theta_robot >=0):
        theta_des = alpha - theta_robot
    elif (alpha < 0) and (theta_robot < 0):
        if np.abs(theta_robot) > np.abs(alpha):
            theta_des = np.abs(theta_robot) - np.abs(alpha)
        else:
            theta_des = -(np.abs(theta_robot) - np.abs(alpha))
    else:
        theta_des = alpha - theta_robot
        if theta_des > np.pi:
            theta_des = -(2*np.pi - theta_des)
        elif theta_des < -np.pi:
            theta_des = 2*np.pi + theta_des
    
        
    theta_des = theta_des * 180 / np.pi
    print("D_theta")
    print(theta_des)
    print("theta_robot")
    print( theta_robot* 180 / np.pi)
    print("alpha")
    print(alpha)
    #turn_speed = wheel_vel / baseline
    #turn_time = theta_des / turn_speed # replace with your calculation
    
    print("turn_time")
    print(2 * theta_des * baseline / wheel_vel)

    if theta_des >= 180:
        theta_des = theta_des - 20 #good
    elif theta_des >= 90:
        theta_des = theta_des - 11 #good
    elif theta_des >= 75:
        theta_des = theta_des - 12
    elif theta_des >= 60:
        theta_des = theta_des
    elif theta_des >= 45:
        theta_des = theta_des - 8
    elif theta_des >= 30:
        theta_des = theta_des - 5.92

    if theta_des <= -135:
        theta_des = theta_des + 6 #good
    elif theta_des <= -90:
        theta_des = theta_des + 7 #good
    elif theta_des <= -75:
        theta_des = theta_des + 12
    elif theta_des <= -60:
        theta_des = theta_des
    elif theta_des <= -45:
        theta_des = theta_des + 6 #good
    elif theta_des <= -30:
        theta_des = theta_des + 5.92


    
    abs_theta_des = np.abs(theta_des)
    
    turn_time = 2 * abs_theta_des * baseline / wheel_vel
    #print("Turning for {:.2f} seconds".format(turn_time))
    

    

    #NEW PART
    if theta_des >= 0:
        lv, rv = operate.pibot.set_velocity([0, 1], turning_tick=wheel_vel, time=turn_time)
    else:
        lv, rv = operate.pibot.set_velocity([0, -1], turning_tick=wheel_vel, time=turn_time)
        
    start = time.time()
    time.sleep(turn_time)
    print("print me after finish turning")
    end = time.time()
    operate.pibot.set_velocity([0, 0])
    drive_meas = measure.Drive(lv, rv, end - start)
    operate.take_pic()
    operate.update_slam(drive_meas)
    operate.record_data()
    # after turning, drive straight to the waypoint
    dist_des = np.sqrt(x_dif**2 + y_dif**2)
    drive_speed = wheel_vel * scale

    drive_time = dist_des / drive_speed # replace with your calculation
    #print("Driving for {:.2f} seconds".format(drive_time))
    lv, rv = operate.pibot.set_velocity([1, 0], tick=wheel_vel, time=drive_time)
    start = time.time()
    ###################################################
    time.sleep(drive_time)
    end = time.time()
    operate.pibot.set_velocity([0, 0])
    drive_meas = measure.Drive(lv, rv, end - start)
    operate.take_pic()
    operate.update_slam(drive_meas)
    operate.record_data()
    
    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))


def get_robot_pose():
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    
    # update the robot pose [x,y,theta]
    print("robot_pose")
    pose =  operate.ekf.robot.state # replace with your calculation
    robot_pose = [pose[0][0],pose[1][0], pose[2][0]]
    ####################################################
    print(robot_pose)
    return robot_pose



# main loop
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map_3fruits.txt') #change this line depending on what map to read
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()
    
    operate = Operate(args)
    

    # read in the true map
    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    
    wheel_vel = 30 # tick
    #initialize pibot a zero velocity and look for markers
    lv, rv = operate.pibot.set_velocity([0, 0], tick=wheel_vel, time=2)
    start = time.time()
    ###################################################
    time.sleep(2)
    end = time.time()
    drive_meas = measure.Drive(lv, rv, end - start)
    operate.take_pic()
    operate.update_slam(drive_meas)
    operate.record_data()
    
    
    # The following code is only a skeleton code the semi-auto fruit searching task
    while True:
        # enter the waypoints
        # instead of manually enter waypoints, you can get coordinates by clicking on a map, see camera_calibration.py
        x,y = 0.0,0.0
        x = input("X coordinate of the waypoint: ")
        try:
            x = float(x)
        except ValueError:
            print("Please enter a number.")
            continue
        y = input("Y coordinate of the waypoint: ")
        try:
            y = float(y)
        except ValueError:
            print("Please enter a number.")
            continue

        # estimate the robot's pose
        
        robot_pose = get_robot_pose()

        # robot drives to the waypoint
        waypoint = [x,y]
        drive_to_point(waypoint,robot_pose)
        robot_pose = get_robot_pose()
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        operate.pibot.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
