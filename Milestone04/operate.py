# basic python packages
import numpy as np
import cv2 
import os, sys
import time

# import utility functions
sys.path.insert(0, "{}/utility".format(os.getcwd()))
from util.pibot import PenguinPi # access the robot
import util.DatasetHandler as dh # save/load functions
import util.measure as measure # measurements
import pygame # python package for GUI
import shutil # python package for file operations

# import SLAM components
sys.path.insert(0, "{}/slam".format(os.getcwd()))
from slam.ekf import EKF
from slam.robot import Robot
import slam.aruco_detector as aruco



class Operate:
    def __init__(self, args):
        self.folder = 'pibot_dataset/'
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            shutil.rmtree(self.folder)
            os.makedirs(self.folder)
        
        # initialise data parameters
        if args.play_data:
            self.pibot = dh.DatasetPlayer("record")
        else:
            self.pibot = PenguinPi(args.ip, args.port)

        # initialise SLAM parameters
        self.ekf = self.init_ekf(args.calib_dir, args.ip)
        self.aruco_det = aruco.aruco_detector(
            self.ekf.robot, marker_length = 0.07) # size of the ARUCO markers

        if args.save_data:
            self.data = dh.DatasetWriter('record')
        else:
            self.data = None
        self.output = dh.OutputWriter('lab_output')
        self.command = {'motion':[0, 0], 
                        'inference': False,
                        'output': False,
                        'save_inference': False,
                        'save_image': False}
        self.quit = False
        self.pred_fname = ''
        self.request_recover_robot = False
        self.file_output = None
        self.ekf_on = False
        self.double_reset_comfirm = 0
        self.image_id = 0
        self.notification = 'Press ENTER to start SLAM'
        # a 5min timer
        self.count_down = 300
        self.start_time = time.time()
        self.control_clock = time.time()
        # initialise images
        self.img = np.zeros([240,320,3], dtype=np.uint8)
        self.aruco_img = np.zeros([240,320,3], dtype=np.uint8)
        self.bg = pygame.image.load('pics/gui_mask.jpg')

    # wheel control
    def control(self):       
        if args.play_data:
            lv, rv = self.pibot.set_velocity()            
        else:
            lv, rv = self.pibot.set_velocity(
                self.command['motion'], tick = 30)
        if not self.data is None:
            self.data.write_keyboard(lv, rv)
        dt = time.time() - self.control_clock
        drive_meas = measure.Drive(lv, rv, dt)
        self.control_clock = time.time()
        return drive_meas
    # camera control
    def take_pic(self):
        self.img = self.pibot.get_image()
        if not self.data is None:
            self.data.write_image(self.img)

    # SLAM with ARUCO markers       
    def update_slam(self, drive_meas):
        lms, self.aruco_img = self.aruco_det.detect_marker_positions(self.img)
        print(lms)
        if self.request_recover_robot:
            is_success = self.ekf.recover_from_pause(lms)
            if is_success:
                self.notification = 'Robot pose is successfuly recovered'
                self.ekf_on = True
            else:
                self.notification = 'Recover failed, need >2 landmarks!'
                self.ekf_on = False
            self.request_recover_robot = False
        elif self.ekf_on: # and not self.debug_flag:
            self.ekf.predict(drive_meas)
            self.ekf.add_landmarks(lms)
            self.ekf.update(lms)

    # save images taken by the camera
    def save_image(self):
        f_ = os.path.join(self.folder, f'img_{self.image_id}.png')
        if self.command['save_image']:
            image = self.pibot.get_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f_, image)
            self.image_id += 1
            self.command['save_image'] = False
            self.notification = f'{f_} is saved'

    # wheel and camera calibration for SLAM
    def init_ekf(self, datadir, ip):
        fileK = "{}intrinsic.txt".format(datadir)
        camera_matrix = np.loadtxt(fileK, delimiter=',')
        fileD = "{}distCoeffs.txt".format(datadir)
        dist_coeffs = np.loadtxt(fileD, delimiter=',')
        fileS = "{}scale.txt".format(datadir)
        scale = np.loadtxt(fileS, delimiter=',')
        if ip == 'localhost':
           scale /= 2
        fileB = "{}baseline.txt".format(datadir)  
        baseline = np.loadtxt(fileB, delimiter=',')
        robot = Robot(baseline/2, scale, camera_matrix, dist_coeffs)
        return EKF(robot)

    # save SLAM map
    def record_data(self):
        if self.command['output']:
            self.output.write_map(self.ekf)
            self.notification = 'Map is saved'
            self.command['output'] = False

    # paint the GUI            
    def draw(self, canvas):
        canvas.blit(self.bg, (0, 0))
        text_colour = (220, 220, 220)
        v_pad = 40
        h_pad = 20

        # paint SLAM outputs
        ekf_view = self.ekf.draw_slam_state(res=(320, 480+v_pad),
            not_pause = self.ekf_on)
        canvas.blit(ekf_view, (2*h_pad+320, v_pad))
        robot_view = cv2.resize(self.aruco_img, (320, 240))
        self.draw_pygame_window(canvas, robot_view, 
                                position=(h_pad, v_pad)
                                )

        # canvas.blit(self.gui_mask, (0, 0))
        self.put_caption(canvas, caption='SLAM', position=(2*h_pad+320, v_pad)) # M2
        self.put_caption(canvas, caption='Detector (M3)',
                         position=(h_pad, 240+2*v_pad)) # M3
        self.put_caption(canvas, caption='PiBot Cam', position=(h_pad, v_pad))

        notifiation = TEXT_FONT.render(self.notification,
                                          False, text_colour)
        canvas.blit(notifiation, (h_pad+10, 596))

        time_remain = self.count_down - time.time() + self.start_time
        if time_remain > 0:
            time_remain = f'Count Down: {time_remain:03.0f}s'
        elif int(time_remain)%2 == 0:
            time_remain = "Time Is Up !!!"
        else:
            time_remain = ""
        count_down_surface = TEXT_FONT.render(time_remain, False, (50, 50, 50))
        canvas.blit(count_down_surface, (2*h_pad+320+5, 530))
        return canvas

    @staticmethod
    def draw_pygame_window(canvas, cv2_img, position):
        cv2_img = np.rot90(cv2_img)
        view = pygame.surfarray.make_surface(cv2_img)
        view = pygame.transform.flip(view, True, False)
        canvas.blit(view, position)
    
    @staticmethod
    def put_caption(canvas, caption, position, text_colour=(200, 200, 200)):
        caption_surface = TITLE_FONT.render(caption,
                                          False, text_colour)
        canvas.blit(caption_surface, (position[0], position[1]-25))

    # keyboard teleoperation        
    def update_keyboard(self):
        for event in pygame.event.get():
            ########### replace with your M1 codes ###########
            # drive forward
            if event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                self.command['motion'] = [6, 0]  # TODO: replace with your M1 code to make the robot drive forward
            # drive backward
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                self.command['motion'] = [-6, 0] # TODO: replace with your M1 code to make the robot drive backward
            # turn left
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                self.command['motion'] = [0, 4] # TODO: replace with your M1 code to make the robot turn left
            # drive right
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                self.command['motion'] = [0, -4] # TODO: replace with your M1 code to make the robot turn right
            if event.type == pygame.KEYUP and (event.key == pygame.K_UP or event.key == pygame.K_DOWN or event.key == pygame.K_LEFT or event.key == pygame.K_RIGHT):
                 self.command['motion'] = [0, 0]
            ####################################################
            # stop
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.command['motion'] = [0, 0]
            # save image
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_i:
                self.command['save_image'] = True
            # save SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.command['output'] = True
            # reset SLAM map
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                if self.double_reset_comfirm == 0:
                    self.notification = 'Press again to confirm CLEAR MAP'
                    self.double_reset_comfirm +=1
                elif self.double_reset_comfirm == 1:
                    self.notification = 'SLAM Map is cleared'
                    self.double_reset_comfirm = 0
                    self.ekf.reset()
            # run SLAM
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                n_observed_markers = len(self.ekf.taglist)
                if n_observed_markers == 0:
                    if not self.ekf_on:
                        self.notification = 'SLAM is running'
                        self.ekf_on = True
                    else:
                        self.notification = '> 2 landmarks is required for pausing'
                elif n_observed_markers < 3:
                    self.notification = '> 2 landmarks is required for pausing'
                else:
                    if not self.ekf_on:
                        self.request_recover_robot = True
                    self.ekf_on = not self.ekf_on
                    if self.ekf_on:
                        self.notification = 'SLAM is running'
                    else:
                        self.notification = 'SLAM is paused'
            # quit
            elif event.type == pygame.QUIT:
                self.quit = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit = True
        if self.quit:
            pygame.quit()
            sys.exit()


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

def find_lm(): #function to turn left and right on the spot at beginning to sort where land marks are
    pose = get_robot_pose()
    th_init = pose[2]
    th_des = np.pi/4
    th = th_init
    start = time.time()
    operate.command['motion'] = [0, 1]
    i = 0
    while th < (th_init + th_des):
        end = time.time()
        lv, rv = operate.pibot.set_velocity( operate.command['motion'], turning_tick = 10)
        start_prev = start
        start = time.time()
        time.sleep(0.1)
        dt = end - start_prev
        if i ==0:
            dt = 0.1
        drive_meas = measure.Drive(lv, rv, dt)
        operate.take_pic()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        th = get_robot_pose()[2]
        i = i + 1
    
    th_init = th
    operate.command['motion'] = [0,-1]
    th_des = -np.pi/2
    i = 0
    while th > (th_init + th_des):
        print("EFK")
        print(operate.ekf_on)
        end = time.time()
        lv, rv = operate.pibot.set_velocity( operate.command['motion'], turning_tick = 10)
        start_prev = start
        start = time.time()
        time.sleep(0.1)
        dt = end - start_prev
        if i ==0:
            dt = 0.1
        drive_meas = measure.Drive(lv, rv, dt)
        operate.take_pic()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        th = get_robot_pose()[2]
        i = i + 1

def find_lm360(): #function to turn left and right on the spot at beginning to sort where land marks are
    pose = get_robot_pose()
    th_init = pose[2]
    print("360!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1")
    th_des = 2*np.pi
    th = th_init
    start = time.time()
    operate.command['motion'] = [0, 1]
    i = 0
    while th < (th_init + th_des):
        end = time.time()
        lv, rv = operate.pibot.set_velocity( operate.command['motion'], turning_tick = 10)
        start_prev = start
        start = time.time()
        time.sleep(0.2)
        dt = end - start_prev
        if i ==0:
            dt = 0.2
        drive_meas = measure.Drive(lv, rv, dt)
        operate.take_pic()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        th = get_robot_pose()[2]
        i = i + 1
        lms, operate.aruco_img = operate.aruco_det.detect_marker_positions(operate.img)
        is_success = operate.ekf.recover_from_pause(lms)
        if  is_success: 
            print("BREAK!!!!!!!!!!!!!!!!!!!!!" )
            break #is there more than 2 land_marks. 



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

     #turn on the spot and look for land marks

    robot_pose = get_robot_pose()
    wheel_vel = 20 # tick
    x_des = waypoint[0]
    y_des = waypoint[1]
    
    x_robot = robot_pose[0]
    y_robot = robot_pose[1]
    theta_robot = robot_pose[2]
    
    x_dif = x_des - x_robot
    y_dif = y_des - y_robot
    
    j = 0
    while np.sqrt(x_dif**2 + y_dif**2)>0.1:
        # turn towards the waypoint
        alpha = np.arctan2(y_dif, x_dif) #bound between [-pi,pi]
        
        #NEW PART
        #first bound to [0,360]
        if theta_robot > np.pi:
            while theta_robot > np.pi:
                theta_robot = theta_robot - 2*np.pi
        if theta_robot < (-np.pi):
            while theta_robot < np.pi:
                theta_robot = 2*np.pi + theta_robot
        
        #covert to [-pi,pi]
        #if theta_robot > np.pi:
        #   theta_robot = theta_robot - 2*np.pi
         
        
        #alpha and theta_robot now in [-pi,pi]
        
        #CALC

        theta_des = alpha - theta_robot
            
        if theta_des > np.pi:
            theta_des = -(2*np.pi - theta_des)
        elif theta_des < -np.pi:
                theta_des = 2*np.pi + theta_des
        
        
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

            
        theta_des = theta_des * 180 / np.pi
        print("D_theta")
        print(theta_des)
        print("theta_robot")
        print( theta_robot* 180 / np.pi)

        abs_theta_des = np.abs(theta_des)
        
        turn_time = 2 * abs_theta_des * baseline / 10
        #print("Turning for {:.2f} seconds".format(turn_time))
        print("turn_time")
        print(turn_time)

        #TURNING CONTROL
        if theta_des>0: 
            operate.command['motion'] = [0,1]
        else: 
            operate.command['motion'] = [0,-1]


        i = 0
        start = time.time()
        total_time = 0
        th = get_robot_pose()[2]
        
        while total_time < turn_time:
            end = time.time()
            lv, rv = operate.pibot.set_velocity( operate.command['motion'], turning_tick = 10)
            start_prev = start
            start = time.time()
            time.sleep(0.2)
            dt = end - start_prev
            if i ==0:
                dt = 0.2
            drive_meas = measure.Drive(lv, rv, dt)
            operate.take_pic()
            operate.update_slam(drive_meas)
            operate.record_data()
            operate.save_image()
            # visualise
            operate.draw(canvas)
            pygame.display.update()
            total_time = total_time + dt
            step = i + 1
            th = get_robot_pose()[2]



        #operate.pibot.set_velocity([0,0])
        #time.sleep(0.2)
        # after turning, drive straight to the waypoint
        dist_des = np.sqrt(x_dif**2 + y_dif**2)
        drive_speed = wheel_vel * scale/2

        drive_time = dist_des / drive_speed # replace with your calculation
        #print("Driving for {:.2f} seconds".format(drive_time))
        
        total_time = 0
        operate.command['motion'] = [1,0]

        i = 0
        start = time.time()
        total_time = 0
        while total_time < drive_time/5: 
            end = time.time()
            lv, rv = operate.pibot.set_velocity( operate.command['motion'], tick = wheel_vel)
            start_prev = start
            start = time.time()
            time.sleep(0.2)
            dt = end - start_prev
            if i ==0:
                dt = 0.2
            drive_meas = measure.Drive(lv, rv, dt)
            operate.take_pic()
            operate.update_slam(drive_meas)
            operate.record_data()
            operate.save_image()
            # visualise
            operate.draw(canvas)
            pygame.display.update()
            total_time = total_time + dt
            i = i + 1
            

        robot_pose = get_robot_pose()
        x_des = waypoint[0]
        y_des = waypoint[1]
        
        x_robot = robot_pose[0]
        y_robot = robot_pose[1]
        theta_robot = robot_pose[2]
        
        x_dif = x_des - x_robot
        y_dif = y_des - y_robot




    print("Arrived at [{}, {}]".format(waypoint[0], waypoint[1]))
    operate.pibot.set_velocity([0,0])


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



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map", type=str, default='M4_true_map_5fruits.txt') #change this line depending on what map to read
    parser.add_argument("--ip", metavar='', type=str, default='localhost')
    parser.add_argument("--port", metavar='', type=int, default=40000)
    parser.add_argument("--calib_dir", type=str, default="calibration/param/")
    parser.add_argument("--save_data", action='store_true')
    parser.add_argument("--play_data", action='store_true')
    args, _ = parser.parse_known_args()
    
    pygame.font.init() 
    TITLE_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 35)
    TEXT_FONT = pygame.font.Font('pics/8-BitMadness.ttf', 40)
    
    width, height = 700, 660
    canvas = pygame.display.set_mode((width, height))
    pygame.display.set_caption('ECE4078 2021 Lab')
    pygame.display.set_icon(pygame.image.load('pics/8bit/pibot5.png'))
    canvas.fill((0, 0, 0))
    splash = pygame.image.load('pics/loading.png')
    pibot_animate = [pygame.image.load('pics/8bit/pibot1.png'),
                     pygame.image.load('pics/8bit/pibot2.png'),
                     pygame.image.load('pics/8bit/pibot3.png'),
                    pygame.image.load('pics/8bit/pibot4.png'),
                     pygame.image.load('pics/8bit/pibot5.png')]
    pygame.display.update()

    start = False

    counter = 40
    while not start:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                start = True
        canvas.blit(splash, (0, 0))
        x_ = min(counter, 600)
        if x_ < 600:
            canvas.blit(pibot_animate[counter%10//2], (x_, 565))
            pygame.display.update()
            counter += 2

    operate = Operate(args)

    operate.pibot.set_velocity([0,0])

    while start:
        operate.update_keyboard()
        operate.take_pic()
        drive_meas = operate.control()
        operate.update_slam(drive_meas)
        operate.record_data()
        operate.save_image()
        # visualise
        operate.draw(canvas)
        pygame.display.update()
        operate.pibot.set_velocity([0, 0])
            
        if operate.ekf_on: 
            operate.load_slam()
            while True: 
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
                waypoint = [x, y]
                
                i = 0
                #waypoints = [[0.0, -0.3], [0.3, -0.6], [0.3, -0.7], [0.4, -0.8], [1, -0.8], [0.9, -0.8]]
                #for waypoint in waypoints: 

                robot_pose = get_robot_pose()
                if i ==0: 
                    find_lm()
                else: 
                    ind_lm360()

                robot_poose = get_robot_pose()
                    # robot drives to the waypoint
                drive_to_point(waypoint,robot_pose)
                robot_pose = get_robot_pose()
                i = i + 1
                operate.pibot.set_velocity([0,0])
                #    print("Finished driving to waypoint: {}; New robot pose: {}".format(waypoint,robot_pose))

                    # exit
                operate.pibot.set_velocity([0, 0])
                uInput = input("Add a new waypoint? [Y/N]")
                if uInput == 'N':
                    operate.pibot.set_velocity([0, 0])
                    break
                

