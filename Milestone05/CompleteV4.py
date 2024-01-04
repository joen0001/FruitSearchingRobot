import math
import heapq
import matplotlib.pyplot as plt

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

class Env:
    def __init__(self, aruco_list, fruit_list):
        self.x_range = 31  ### check the size of the map and add one more to the width and height### also the map has negative values. Not a major problem though
        self.y_range = 31
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.aruco_list = aruco_list
        self.fruit_list = fruit_list
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        
        x = self.x_range
        y = self.y_range
        aruco_list = self.aruco_list
        fruit_list = self.fruit_list
        obs = set()
    
        # Generating Border Walls
        dimension = round((x-1)/2)
        for i in range(-int(dimension),int(dimension)):
            obs.add((i, -dimension))
        for i in range(-int(dimension),int(dimension)):
            obs.add((i, dimension))
        for i in range(-int(dimension),int(dimension)+1):
            obs.add((-dimension, i))
        for i in range(-int(dimension),int(dimension)+1):
            obs.add((dimension, i))
        
        # you need the dimensions of the aruco markers
        #the positions are in decimals, this causes problems because the code is based on intergers
        #also since some the positions are negative they don't work with the map created here. Easily solved my modifing the map
        #modify the range of the for loop to accept the 10 aruco markers
        #the decimal issue seems to be fixed with np.arange(); however the path doesnt consider it and goes through it 

        # Generating Aruco Markers
        for x in range(0,len(aruco_list)):
            Aruco_x = aruco_list[x][0] 
            Aruco_y = aruco_list[x][1] 

            for i in np.arange(Aruco_x-1, Aruco_x+2):
                obs.add((i, Aruco_y-1))
                obs.add((i, Aruco_y))
                obs.add((i, Aruco_y+1))

        # Generating Obstacle Fruit
        for x in range(0,len(fruit_list)):
            fruit_x = fruit_list[x][0] 
            fruit_y = fruit_list[x][1] 

            for i in np.arange(fruit_x, fruit_x+1):
                obs.add((i, fruit_y))

        return obs

class AStar:
    
    def __init__(self, s_start, s_goal, heuristic_type, aruco_list, fruit_list):
        self.s_start = s_start
        self.s_goal = s_goal
        self.heuristic_type = heuristic_type
        #self.A1 = A1

        self.Env = Env(aruco_list,fruit_list)  

        self.u_set = self.Env.motions  
        self.obs = self.Env.obs  

        self.OPEN = []  
        self.CLOSED = []  
        self.PARENT = dict()  
        self.g = dict()  

    def searching(self):

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (self.f_value(self.s_start), self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:  
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s
                    heapq.heappush(self.OPEN, (self.f_value(s_n), s_n))

        return self.extract_path(self.PARENT), self.CLOSED

    def searching_repeated_astar(self, e):
        

        path, visited = [], []

        while e >= 1:
            p_k, v_k = self.repeated_searching(self.s_start, self.s_goal, e)
            path.append(p_k)
            visited.append(v_k)
            e -= 0.5

        return path, visited

    def repeated_searching(self, s_start, s_goal, e):
        

        g = {s_start: 0, s_goal: float("inf")}
        PARENT = {s_start: s_start}
        OPEN = []
        CLOSED = []
        heapq.heappush(OPEN,
                       (g[s_start] + e * self.heuristic(s_start), s_start))

        while OPEN:
            _, s = heapq.heappop(OPEN)
            CLOSED.append(s)

            if s == s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = g[s] + self.cost(s, s_n)

                if s_n not in g:
                    g[s_n] = math.inf

                if new_cost < g[s_n]:  
                    g[s_n] = new_cost
                    PARENT[s_n] = s
                    heapq.heappush(OPEN, (g[s_n] + e * self.heuristic(s_n), s_n))

        return self.extract_path(PARENT), CLOSED

    def get_neighbor(self, s):
        

        return [(s[0] + u[0], s[1] + u[1]) for u in self.u_set]

    def cost(self, s_start, s_goal):
        

        if self.is_collision(s_start, s_goal):
            return math.inf

        return math.hypot(s_goal[0] - s_start[0], s_goal[1] - s_start[1])

    def is_collision(self, s_start, s_end):
        

        if s_start in self.obs or s_end in self.obs:
            return True

        if s_start[0] != s_end[0] and s_start[1] != s_end[1]:
            if s_end[0] - s_start[0] == s_start[1] - s_end[1]:
                s1 = (min(s_start[0], s_end[0]), min(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
            else:
                s1 = (min(s_start[0], s_end[0]), max(s_start[1], s_end[1]))
                s2 = (max(s_start[0], s_end[0]), min(s_start[1], s_end[1]))

            if s1 in self.obs or s2 in self.obs:
                return True

        return False

    def f_value(self, s):
        

        return self.g[s] + self.heuristic(s)

    def extract_path(self, PARENT):
       

        path = [self.s_goal]
        s = self.s_goal

        while True:
            s = PARENT[s]
            path.append(s)

            if s == self.s_start:
                break

        return list(path)

    def heuristic(self, s):
        

        heuristic_type = self.heuristic_type  
        goal = self.s_goal  

        if heuristic_type == "manhattan":
            return abs(goal[0] - s[0]) + abs(goal[1] - s[1])
        else:
            return math.hypot(goal[0] - s[0], goal[1] - s[1])

class Dijkstra(AStar):
    
    def searching(self):
        

        self.PARENT[self.s_start] = self.s_start
        self.g[self.s_start] = 0
        self.g[self.s_goal] = math.inf
        heapq.heappush(self.OPEN,
                       (0, self.s_start))

        while self.OPEN:
            _, s = heapq.heappop(self.OPEN)
            self.CLOSED.append(s)

            if s == self.s_goal:
                break

            for s_n in self.get_neighbor(s):
                new_cost = self.g[s] + self.cost(s, s_n)

                if s_n not in self.g:
                    self.g[s_n] = math.inf

                if new_cost < self.g[s_n]:  
                    self.g[s_n] = new_cost
                    self.PARENT[s_n] = s

                    
                    heapq.heappush(self.OPEN, (new_cost, s_n))

        return self.extract_path(self.PARENT), self.CLOSED

class Plotting:
    def __init__(self, xI, xG,aruco_list,fruit_list):
        self.xI, self.xG = xI, xG
        self.env = Env(aruco_list,fruit_list)
        self.obs = self.env.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def animation(self, path, visited, name):
        self.plot_grid(name)
        self.plot_visited(visited)
        self.plot_path(path)
        plt.show()

    def animation_lrta(self, path, visited, name):
        self.plot_grid(name)
        cl = self.color_list_2()
        path_combine = []

        for k in range(len(path)):
            self.plot_visited(visited[k], cl[k])
            plt.pause(0.2)
            self.plot_path(path[k])
            path_combine += path[k]
            plt.pause(0.2)
        if self.xI in path_combine:
            path_combine.remove(self.xI)
        self.plot_path(path_combine)
        plt.show()

    def animation_ara_star(self, path, visited, name):
        self.plot_grid(name)
        cl_v, cl_p = self.color_list()

        for k in range(len(path)):
            self.plot_visited(visited[k], cl_v[k])
            self.plot_path(path[k], cl_p[k], True)
            plt.pause(0.5)

        plt.show()

    def animation_bi_astar(self, path, v_fore, v_back, name):
        self.plot_grid(name)
        self.plot_visited_bi(v_fore, v_back)
        self.plot_path(path)
        plt.show()

    def plot_grid(self, name):
        obs_x = [x[0] for x in self.obs]
        obs_y = [x[1] for x in self.obs]

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")
        plt.plot(obs_x, obs_y, "sk")
        plt.title(name)
        plt.axis("equal")

    def plot_visited(self, visited, cl='gray'):
        if self.xI in visited:
            visited.remove(self.xI)

        if self.xG in visited:
            visited.remove(self.xG)

        count = 0

        for x in visited:
            count += 1
            plt.plot(x[0], x[1], color=cl, marker='o')
            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if count < len(visited) / 3:
                length = 20
            elif count < len(visited) * 2 / 3:
                length = 30
            else:
                length = 40
            #
            # length = 15

            if count % length == 0:
                plt.pause(0.001)
        plt.pause(0.01)

    def plot_path(self, path, cl='r', flag=False):
        path_x = [path[i][0] for i in range(len(path))]
        path_y = [path[i][1] for i in range(len(path))]
        #print(path_x)
        #print(path_y)
        if not flag:
            plt.plot(path_x, path_y, linewidth='3', color='r')
        else:
            plt.plot(path_x, path_y, linewidth='3', color=cl)

        plt.plot(self.xI[0], self.xI[1], "bs")
        plt.plot(self.xG[0], self.xG[1], "gs")

        plt.pause(0.01)

    def plot_visited_bi(self, v_fore, v_back):
        if self.xI in v_fore:
            v_fore.remove(self.xI)

        if self.xG in v_back:
            v_back.remove(self.xG)

        len_fore, len_back = len(v_fore), len(v_back)

        for k in range(max(len_fore, len_back)):
            if k < len_fore:
                plt.plot(v_fore[k][0], v_fore[k][1], linewidth='3', color='gray', marker='o')
            if k < len_back:
                plt.plot(v_back[k][0], v_back[k][1], linewidth='3', color='cornflowerblue', marker='o')

            plt.gcf().canvas.mpl_connect('key_release_event',
                                         lambda event: [exit(0) if event.key == 'escape' else None])

            if k % 10 == 0:
                plt.pause(0.001)
        plt.pause(0.01)

    @staticmethod
    def color_list():
        cl_v = ['silver',
                'wheat',
                'lightskyblue',
                'royalblue',
                'slategray']
        cl_p = ['gray',
                'orange',
                'deepskyblue',
                'red',
                'm']
        return cl_v, cl_p

    @staticmethod
    def color_list_2():
        cl = ['silver',
              'steelblue',
              'dimgray',
              'cornflowerblue',
              'dodgerblue',
              'royalblue',
              'plum',
              'mediumslateblue',
              'mediumpurple',
              'blueviolet',
              ]
        return cl

def print_target_fruits_pos(search_list, fruit_list, fruit_true_pos):
    """Print out the target fruits' pos in the search order
    @param search_list: search order of the fruits
    @param fruit_list: list of target fruits
    @param fruit_true_pos: positions of the target fruits
    """

    #print("Search order:")
    n_fruit = 1
    fruits_in_order = []
    for fruit in search_list:
        for i in range(3):
            if fruit == fruit_list[i]:
                #print('{}) {} at [{}, {}]'.format(n_fruit, fruit, np.round(fruit_true_pos[i][0], 1), np.round(fruit_true_pos[i][1], 1)))
                fruits_in_order.append([fruit_true_pos[i][0],fruit_true_pos[i][1]])
                

        
        n_fruit += 1
    #print(fruits_in_order)
    return fruits_in_order

##function to reap the map text file 
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
        aruco_true_pos = []#np.empty([10, 2])

        # remove unique id of targets of the same type
        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)*10
            y = np.round(gt_dict[key]['y'], 1)*10

            if key.startswith('aruco'):
                if len(aruco_true_pos) == 0:
                    aruco_true_pos = np.array([[x, y]])
                else:
                    aruco_true_pos = np.append(aruco_true_pos, [[x, y]], axis=0)
                #why was this block needed? 
                """if key.startswith('aruco10'):
                    aruco_true_pos[9][0] = x
                    aruco_true_pos[9][1] = y
                else:
                    marker_id = int(key[5])
                    aruco_true_pos[marker_id][0] = x
                    aruco_true_pos[marker_id][1] = y"""
            else:
                fruit_list.append(key[:-2])
                if len(fruit_true_pos) == 0:
                    fruit_true_pos = np.array([[x, y]])
                else:
                    fruit_true_pos = np.append(fruit_true_pos, [[x, y]], axis=0)

        return fruit_list, fruit_true_pos, aruco_true_pos

def generate_path(current_pose, current_fruit, aruco_true_pos, fruit_curr2_pos):

    s_start = current_pose
    s_goal = current_fruit 
    #print("path_goal"+ str(s_goal))

    dijkstra = Dijkstra(s_start, s_goal, 'None', aruco_true_pos,fruit_curr2_pos)
    plot = Plotting(s_start, s_goal, aruco_true_pos,fruit_curr2_pos)

    path, visited = dijkstra.searching()
    #print(path)# path store all of the waypoints generated by the algorithm 
    plot.animation(path, visited, "Dijkstra's")  # animation generate
    return path

def corner_points(path):
    corners = []
    curr_grad = 0
    for i in range(len(path)-1):
        x_dif = path[i+1][0] - path[i][0]
        y_dif = path[i+1][1] - path[i][1]
        if x_dif == 0:
            grad = 0
        elif y_dif == 0:
            grad = 20
        else:
            grad = y_dif/x_dif
        if grad != curr_grad:
            corners.append((path[i][0],path[i][1]))
            curr_grad = grad
    corners.append(path[-4])
    return corners

def Activate_Autonomous_Naviagtion(fruit_list, fruit_true_pos, aruco_true_pos, search_list):
    
    for current_goal in range(len(search_list)):
       
        #robot pose will come from a seperate function 
        # make sure to convert it to a tuple 
        robot_pose = get_robot_pose()
        robot_pose2 = (robot_pose[0], robot_pose[1])

       
        fruit_curr_pos = print_target_fruits_pos(search_list, fruit_list, fruit_true_pos)
        #print("curr: "+str(fruit_curr_pos))
        
        fruit_curr2_pos = fruit_curr_pos.copy()
        fruit_curr2_pos.remove(fruit_curr_pos[current_goal])
        #print("obstacle fruit "+str(fruit_curr2_pos))
        
        #print("curr_pos: "+str(fruit_curr_pos[current_goal]))
        #divide the waypoints by 10 because of the scale
        path = generate_path(robot_pose2, tuple(fruit_curr_pos[current_goal]), aruco_true_pos, fruit_curr2_pos)
        path.reverse()
        corners = corner_points(path)
        for waypoint in corners:
            waypoint_p = list(waypoint) 
            #for p in range(2):
            waypoint_p[0] /= 10
            waypoint_p[1] /= 10
            print(waypoint_p)
            robot_pose = get_robot_pose()
                #robot_pose2 = (robot_pose[0], robot_pose[1])
            drive_to_point(waypoint_p, robot_pose)


    return 0

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
    #turn_speed = wheel_vel / baseline
    #turn_time = theta_des / turn_speed # replace with your calculation

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
    print("Turning Finished")
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
    pose  =  operate.ekf.robot.state # replace with your calculation
    robot_pose = [pose[0][0],pose[1][0], pose[2][0]]
    ####################################################
    return robot_pose


def main(map):
    
    

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

    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map(map)

    #this will come from a seperate function 
    search_list = read_search_list()
    #print(search_list[0])
    #call on the function to iterate on each fruit goal and generate a path for each one
    Activate_Autonomous_Naviagtion(fruit_list, fruit_true_pos, aruco_true_pos, search_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map_3fruits.txt') #change this line depending on what map to read
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()
    
    operate = Operate(args)
    main(args.map)
    
