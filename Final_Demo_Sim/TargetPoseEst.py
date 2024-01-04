# estimate the pose of a target object detected
import numpy as np
import json
import os
from pathlib import Path
import ast
# import cv2
import math
#from machinevisiontoolbox import Image

import matplotlib.pyplot as plt
import PIL

def extract_data(): #extracts data from output.txt file and converts it into an array
    lines = []
    with open('output.txt') as f: #reading text file
        lines = f.readlines()

    #lines = np.array(lines)
    pose = []
    detections = []
    new_detections = []
    for line in lines: #grouping detections by snapshot
        new_line = []
        detections.append(line)
        if "]]" in line: #end of snapshot
            
            for det in detections: #removing unwanted charaters
                    new_det = [charater for charater in det if charater != ('[') ]
                    new_det = [charater for charater in new_det if charater != (']') ]
                    new_det = [charater for charater in new_det if charater != ('\n') ]
                    new_detections.append(new_det)

            pose.append(new_detections)
            detections = []
            new_detections = []

    #joining strings
    pose_new = []
    detections_new = []
    for detections in pose: 
        for detection in detections:
            new = ''
            for char in detection:
                new += char
                
            #print(new)
            detections_new.append(new)
        pose_new.append(detections_new)
        detections_new = []

    #converting text into lists
    pose_array = []
    detections_array = []
    for detections in pose_new:
        for detection in detections:
            detection_array = np.array(list(map(float, detection.split())))
            detections_array.append(detection_array)
        pose_array.append(detections_array)
        detections_array = []

    return pose_array

# use the machinevision toolbox to get the bounding box of the detected target(s) in an image
def get_bounding_box(target_number, image_vals):
    #image = PIL.Image.open(image_path).resize((640,480), PIL.Image.NEAREST)
    #target = Image(image)==target_number
    #blobs = target.blobs()
    #[[u1,u2],[v1,v2]] = blobs[0].bbox # bounding box
    #width = abs(u1-u2)
    #height = abs(v1-v2)
    #center = np.array(blobs[0].centroid).reshape(2,)
    #box = [center[0], center[1], int(width), int(height)] # box=[x,y,width,height]
    # plt.imshow(fruit.image)
    # plt.annotate(str(fruit_number), np.array(blobs[0].centroid).reshape(2,))
    # plt.show()
    # assert len(blobs) == 1, "An image should contain only one object of each target type"

    u1 = image_vals[0]
    u2 = image_vals[2]
    v1 = image_vals[1]
    v2 = image_vals[3]
    width = abs(u1-u2)
    height = abs(v1-v2)
    center = [(u2 - u1)/2 + u1, (v2 - v1)/2 + v1]
    box = [int(center[0]), int(center[1]), int(width), int(height)]
    print(box)
    return box

# read in the list of detection results with bounding boxes and their matching robot pose info
def get_image_info(base_dir, file_path, image_poses, snapshot):
    # there are at most 5 types of targets in each image
    target_lst_box = [[], [], [],[],[]]
    target_lst_pose = [[], [], [],[],[]]
    completed_img_dict = {}
    
    box_array = extract_data()
    #GETTING TEXT FILE DATA AND USING THEM IN AN ARRAY
    #data -> poses -> detections -> x,y,x,y
    # add the bounding box info of each target in each image
    # target labels: 1 = apple, 2 = lemon, 3 = person, 0 = not_a_target
    #img_vals = set(Image(base_dir / file_path, grey=True).image.reshape(-1))
    img_vals = box_array[snapshot] #list of detections
    

    for i in range(len(img_vals)):
        target_num = int(img_vals[i][5])+1 #this is the class (0-4)
        if target_num > 0:
            try:
                box = get_bounding_box(target_num, img_vals[i]) # [x,y,width,height]
                pose = image_poses[file_path] # [x, y, theta]
                target_lst_box[target_num-1].append(box) # bouncing box of target
                target_lst_pose[target_num-1].append(np.array(pose).reshape(3,)) # robot pose
            except ZeroDivisionError:
                pass
    
    
    # if there are more than one objects of the same type, combine them
    for i in range(5):
        if len(target_lst_box[i])>0:
            box = np.stack(target_lst_box[i], axis=1)
            pose = np.stack(target_lst_pose[i], axis=1)
            completed_img_dict[i+1] = {'target': box, 'robot': pose}
        
    return completed_img_dict

# estimate the pose of a target based on size and location of its bounding box in the robot's camera view and the robot's pose
def estimate_pose(base_dir, camera_matrix, completed_img_dict):
    camera_matrix = camera_matrix
    focal_length = camera_matrix[0][0]
    print("focal length: ", focal_length)
    # actual sizes of targets [For the simulation models]
    # You need to replace these values for the real world objects
    target_dimensions = []
    apple_dimensions = [0.075448, 0.074871, 0.071889]
    target_dimensions.append(apple_dimensions)
    lemon_dimensions = [0.060588, 0.059299, 0.053017]
    target_dimensions.append(lemon_dimensions)
    orange_dimensions = [0.0721, 0.0771, 0.0739]
    target_dimensions.append(orange_dimensions)
    pear_dimensions = [0.0946, 0.0948, 0.135]
    target_dimensions.append(pear_dimensions)
    strawberry_dimensions = [0.052, 0.0346, 0.0376]
    target_dimensions.append(strawberry_dimensions)

    target_list = ['apple', 'lemon', 'orange', 'pear', 'strawberry'] #swapped orange and pear

    target_pose_dict = {}
    # for each target in each detection output, estimate its pose
    for target_num in completed_img_dict.keys():
        box = completed_img_dict[target_num]['target'] # [[x],[y],[width],[height]]
        robot_pose = completed_img_dict[target_num]['robot'] # [[x], [y], [theta]]
        true_height = target_dimensions[target_num-1][2]
        
        print("true height ref: ", target_num - 1)


        ######### Replace with your codes #########
        # TODO: compute pose of the target based on bounding box info and robot's pose
        
        box_height = box[3]
        print("box height: ", box_height)
        x_coord = box[0]
        print("box x: ", x_coord)
        y_coord = box[1]
        print("box y: ", y_coord)
        u_0 = camera_matrix[0][2]
        print("u_0: ", u_0)

        Z = true_height * focal_length / box_height
        print("Z: ", Z)
 
        robot_x = robot_pose[0]
        robot_y = robot_pose[1]
        robot_theta = robot_pose[2]
        #robot_x = 0.024757
        #robot_y = -0.040691
        #robot_theta = 1.191196
        
        print("robot x:", robot_x)
        print("robot y: ", robot_y)
        print("robot theta: ", robot_theta)

        #first bound to [0,360]
    
        if robot_theta > 2*np.pi:
            while robot_theta > 2*np.pi:           
                robot_theta = robot_theta - 2*np.pi
    
        if robot_theta < 0:      
            while robot_theta < 0:           
                robot_theta = 2*np.pi + robot_theta
    
        #covert to [-pi,pi]  
        if robot_theta > np.pi:                         
            robot_theta = robot_theta - 2*np.pi
        
        X = Z*((x_coord) - u_0)/focal_length

        print("theta is: ", robot_theta)
        #calc fruit x and y for in front of robot
        if robot_theta >= 0 and robot_theta <= np.pi/2: #[0,90]
            fruit_x_int = Z*np.cos(robot_theta) + robot_x 
            fruit_y_int = Z*np.sin(robot_theta) + robot_y 
            if (X >= 0):  #right                     
                x_add = np.cos(np.pi/2 - robot_theta)*X           
                y_add = - np.sin(np.pi/2 - robot_theta)*X
            else : #left                     
                x_add = - np.cos(np.pi/2 - robot_theta)*-X           
                y_add = np.sin(np.pi/2 - robot_theta)*-X
            
        if robot_theta > np.pi/2 and robot_theta <= np.pi: #[90,180]
            fruit_x_int = - Z*np.cos(np.pi - robot_theta) + robot_x 
            fruit_y_int = Z*np.sin(np.pi - robot_theta) + robot_y 
            if (X >= 0):  #right         
                x_add = np.cos(robot_theta - np.pi/2)*X  #check         
                y_add = np.sin(robot_theta - np.pi/2)*X 
            else : #left          
                x_add = - np.cos(robot_theta - np.pi/2)*-X         
                y_add = - np.sin(robot_theta - np.pi/2)*-X

        if robot_theta > -np.pi/2 and robot_theta < 0: #[-90,0] 
            fruit_x_int = Z*np.cos(robot_theta) + robot_x 
            fruit_y_int = - Z*np.sin(-robot_theta) + robot_y 
            if (X >= 0): #right                    
                x_add = - np.cos(np.pi/2 - np.abs(robot_theta))*X                    
                y_add = - np.sin(np.pi/2 - np.abs(robot_theta))*X 
            else :   #left        
                x_add = np.cos(np.pi/2 - np.abs(robot_theta))*-X         
                y_add = np.sin(np.pi/2 - np.abs(robot_theta))*-X

        if robot_theta >= -np.pi and robot_theta < -np.pi/2: #[-180,-90]
            fruit_x_int = - Z*np.cos(np.pi + robot_theta) + robot_x 
            fruit_y_int = - Z*np.sin(np.pi + robot_theta) + robot_y 
            if (X >= 0):           
                x_add = -np.cos(np.abs(robot_theta) - np.pi/2)*X         
                y_add = np.sin(np.abs(robot_theta) - np.pi/2)*X
            else : #left          
                x_add = np.cos(np.abs(robot_theta) - np.pi/2)*-X       
                y_add = - np.sin(np.abs(robot_theta) - np.pi/2)*-X 


        y_target_pose = fruit_y_int + y_add
        x_target_pose = fruit_x_int + x_add
        target_pose = {'y': y_target_pose, 'x': x_target_pose}
        
        target_pose_dict[target_list[target_num-1]] = target_pose
        ###########################################
    
    return target_pose_dict

# merge the estimations of the targets so that there are at most 3 estimations of each target type
def merge_estimations(target_pose_dict):
    target_pose_dict = target_pose_dict
    apple_est, lemon_est, pear_est, orange_est, strawberry_est = [], [], [], [], []
    target_est = {}
    
    # combine the estimations from multiple detector outputs
    for f in target_map:
        for key in target_map[f]:
            if key.startswith('apple'):
                apple_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('lemon'):
                lemon_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('pear'):
                pear_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('orange'):
                orange_est.append(np.array(list(target_map[f][key].values()), dtype=float))
            elif key.startswith('strawberry'):
                strawberry_est.append(np.array(list(target_map[f][key].values()), dtype=float))

    ######### Replace with your codes #########
    # TODO: the operation below takes the first three estimations of each target type, replace it with a better merge solution
    if len(apple_est) > 2:
        apple_est = apple_est[0:2]
    if len(lemon_est) > 2:
        lemon_est = lemon_est[0:2]
    if len(pear_est) > 2:
        pear_est = pear_est[0:2]
    if len(orange_est) > 2:
        orange_est = orange_est[0:2]
    if len(strawberry_est) > 2:
        strawberry_est = strawberry_est[0:2]

    for i in range(3):
        try:
            target_est['apple_'+str(i)] = {'y':apple_est[i][0][0], 'x':apple_est[i][1][0]}
        except:
            pass
        try:
            target_est['lemon_'+str(i)] = {'y':lemon_est[i][0][0], 'x':lemon_est[i][1][0]}
        except:
            pass
        try:
            target_est['pear_'+str(i)] = {'y':pear_est[i][0][0], 'x':pear_est[i][1][0]}
        except:
            pass
        try:
            target_est['orange_'+str(i)] = {'y':orange_est[i][0][0], 'x':orange_est[i][1][0]}
        except:
            pass
        try:
            target_est['strawberry_'+str(i)] = {'y':strawberry_est[i][0][0], 'x':strawberry_est[i][1][0]}
        except:
            pass
    ###########################################
        
    return target_est


if __name__ == "__main__":
    # camera_matrix = np.ones((3,3))/2
    fileK = "{}intrinsic.txt".format('./calibration/param/')
    camera_matrix = np.loadtxt(fileK, delimiter=',')
    base_dir = Path('./')
    
    
    # a dictionary of all the saved detector outputs
    image_poses = {}
    poses = []
    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            image_poses[pose_dict['imgfname']] = pose_dict['pose']
            
    
     

    with open(base_dir/'lab_output/images.txt') as fp:
        for line in fp.readlines():
            pose_dict = ast.literal_eval(line)
            pose = pose_dict['pose']
            poses.append(pose)


    

    # estimate pose of targets in each detector output
    target_map = {}     
    snapshot = 0;   

    for file_path in image_poses.keys():
        print("snapshot")
        print(snapshot)
        completed_img_dict = get_image_info(base_dir, file_path, image_poses, snapshot)
        target_map[file_path] = estimate_pose(base_dir, camera_matrix, completed_img_dict)
        snapshot = snapshot + 1
        print("")

    # merge the estimations of the targets so that there are at most 3 estimations of each target type
    target_est = merge_estimations(target_map)
    print(target_est)
                     
    # save target pose estimations
    with open(base_dir/'lab_output/targets.txt', 'w') as fo:
        json.dump(target_est, fo)
    
    print('Estimations saved!')
