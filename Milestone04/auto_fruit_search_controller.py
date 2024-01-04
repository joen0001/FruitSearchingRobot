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
def drive_to_point(v,w,dt):
    # imports camera / wheel calibration parameters 
    fileS = "calibration/param/scale.txt"
    scale = np.loadtxt(fileS, delimiter=',')
    fileB = "calibration/param/baseline.txt"
    baseline = np.loadtxt(fileB, delimiter=',')
    
    ####################################################
    # TODO: replace with your codes to make the robot drive to the waypoint
    # One simple strategy is to first turn on the spot facing the waypoint,
    # then drive straight to the way point

    lv, rv = ppi.set_velocity([v, w], 20, 5, dt)
    start = time.time()
    ###################################################
    time.sleep(dt)
    end = time.time()
    operate.pibot.set_velocity([0, 0])
    drive_meas = measure.Drive(lv, rv, dt)
    operate.take_pic()
    operate.update_slam(drive_meas)
    operate.record_data()

def get_robot_pose(v,w,dt,current_pose):
    ####################################################
    # TODO: replace with your codes to estimate the pose of the robot
    # We STRONGLY RECOMMEND you to use your SLAM code from M2 here
    
    # update the robot pose [x,y,theta]
    # pose =  operate.ekf.robot.state # replace with your calculation
    # robot_pose = [pose[0][0],pose[1][0], pose[2][0]]
    x = current_pose[0]
    y = current_pose[1]
    th = current_pose[2]
    if w == 0:
        x_new = x + dt*v*np.cos(th)
        y_new = y + dt*v*np.sin(th)
        th_new = (th + w*dt)
    else:
        R = v/w
        th_new = (th + w*dt)
        x_new = x + R*(-np.sin(th)+np.sin(th_new))
        y_new = y + R*(np.cos(th)-np.cos(th_new))
    robot_pose = [x_new,y_new,th_new]
    ####################################################
    return robot_pose

class TentaclePlanner:
    
    def __init__(self,dt=0.1,steps=2,alpha=1,obstacles=[]):
        
        self.dt = dt
        self.steps = steps
        # Tentacles are possible trajectories to follow
        self.tentacles = [(1,0),(-1,0),(0,0)] # (velocity v, angular velocity w)
        self.obstacles = obstacles
        self.alpha = alpha

    # Play a trajectory and evaluate where you'd end up
    def roll_out(self,v,w,goal_x,goal_y,x,y,th):
        
        for j in range(self.steps):
            if w == 0:
                x_new = x + self.dt*v*np.cos(th)
                y_new = y + self.dt*v*np.sin(th)
                th_new = (th + w*self.dt)
            else:
                R = v/w
                th_new = (th + w*self.dt)
                x_new = x + R*(-np.sin(th)+np.sin(th_new))
                y_new = y + R*(np.cos(th)-np.cos(th_new))
                
            
            if (self.check_collision(x_new,y_new)):
                return np.inf
        
        cost = self.alpha*((goal_x-x_new)**2 + (goal_y-y_new)**2)
        
        return cost
    
    def check_collision(self,x,y):
        
        min_dist = np.min(np.sqrt((x-self.obstacles[:,0])**2+(y-self.obstacles[:,1])**2))
        
        if (min_dist < 0.1):
            return True
        return False
    
    # Choose trajectory that will get you closest to the goal
    def plan(self,goal_x,goal_y,x,y,th):
        
        costs =[]
        for v,w in self.tentacles:
            costs.append(self.roll_out(v,w,goal_x,goal_y,x,y,th))
        
        best_idx = np.argmin(costs)
        
        return self.tentacles[best_idx]

class RobotController:
    
    def __init__(self,Kp=0.1,Ki=0.01):
        self.fileS = "{}scale.txt".format("calibration/param/")
        self.scale = np.loadtxt(self.fileS, delimiter=',')
        self.fileB = "{}baseline.txt".format("calibration/param/")  
        self.baseline = np.loadtxt(self.fileB, delimiter=',')
        
        self.Kp = Kp
        self.Ki = Ki
        self.e_sum_l = 0
        self.e_sum_r = 0

    def p_control(self,w_desired,w_measured,e_sum):
        
        duty_cycle = min(max(-1,self.Kp*(w_desired-w_measured) + self.Ki*e_sum),1)
        
        e_sum = e_sum + (w_desired-w_measured)
        
        return duty_cycle, e_sum
        
        
    def drive(self,v_desired,w_desired,wl,wr):
        
        wl_desired = (2.0*v_desired-w_desired*self.baseline)/(2.0*self.scale)
        wr_desired = (w_desired*self.baseline+2.0*v_desired)/(2.0*self.scale)
        
        duty_cycle_l,self.e_sum_l = self.p_control(wl_desired,wl,self.e_sum_l)
        duty_cycle_r,self.e_sum_r = self.p_control(wr_desired,wr,self.e_sum_r)
        
        return duty_cycle_l, duty_cycle_r

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
    aruco_true_pos = aruco_true_pos[1:]
    print("obstacles =",aruco_true_pos)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)

    waypoint = [0.0,0.0]
    robot_pose = [0.0,0.0,0.0]
    
    scale = 1
    dt = 1/scale
    planner = TentaclePlanner(dt=dt,steps=10*scale,alpha=1,obstacles=aruco_true_pos)
    controller = RobotController(Kp=1,Ki=0.25)
    ppi = PenguinPi(args.ip,args.port)

    wheel_vel = 20 # tick
    #initialize pibot a zero velocity and look for markers
    v = 0
    w = 0
    lv, rv = operate.pibot.set_velocity([v, w], tick=wheel_vel, time=1)
    start = time.time()
    ###################################################
    time.sleep(1)
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
        
        # robot_pose = get_robot_pose(v,w,dt,robot_pose)
        # robot drives to the waypoint
        waypoint = [x,y]

        for i in range(50):
            v,w = planner.plan(waypoint[0],waypoint[1],robot_pose[0],robot_pose[1],robot_pose[2])
            print("v=",v)
            print("w=",w)
            drive_to_point(v,w,dt)
            # time.sleep(dt)
            # drive_to_point(v,w,dt)
            robot_pose = get_robot_pose(v,w,dt,robot_pose)
        print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

        # exit
        operate.pibot.set_velocity([0, 0])
        uInput = input("Add a new waypoint? [Y/N]")
        if uInput == 'N':
            break
