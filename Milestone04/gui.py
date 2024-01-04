import numpy as np
import cv2 as cv
import argparse
import json

SCALE = 2
IMAGE_SIZE = 160*SCALE # Output Map Size
BOX_SIZE = 5*SCALE # Size of Aruco Marker box/warning size
FRUIT_RADIUS = 4*SCALE # Size of target fruit
FRUIT_GRAB = 25*SCALE # Size of area to grab fruit
ROBOT_RADIUS = 5*SCALE # Size of Robot
ANGLE_MARK_SIZE = 7*SCALE # Length of angle indicator of robot
DIMENSION = 3 # [-3,3] coordinate map max size

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
        for i in range(5):
            if fruit == fruit_list[i]:
                print('{}) {} at [{}, {}]'.format(n_fruit,
                                                  fruit,
                                                  np.round(fruit_true_pos[i][0], 1),
                                                  np.round(fruit_true_pos[i][1], 1)))
        
        n_fruit += 1

def translate_coord(coord):
    return round((coord+DIMENSION/2)*IMAGE_SIZE/DIMENSION)

def draw():
    # Create a white image
    img = np.ones((IMAGE_SIZE,IMAGE_SIZE,3), np.uint8)*255

    # Draw Fruit Radius
    for fruit in search_list:
        for i in range(5):
            if fruit == fruits_list[i]:
                centre = (translate_coord(fruits_true_pos[i][0]),translate_coord(fruits_true_pos[i][1]))            
                cv.circle(img,centre,FRUIT_GRAB, (150,255,150), -1)

    # Draw ARUCO Obstacles
    for marker in aruco_true_pos:
        corner1 = (translate_coord(marker[0])-BOX_SIZE,translate_coord(marker[1])+BOX_SIZE)
        corner2 = (translate_coord(marker[0])+BOX_SIZE,translate_coord(marker[1])-BOX_SIZE)
        cv.rectangle(img,corner1,corner2,(0,0,255),-1)

    # Draw Robot
    robot_pose = [0.0,0.0,0.0]
    centre = (translate_coord(robot_pose[0]),translate_coord(robot_pose[1]))
    cv.circle(img,centre,ROBOT_RADIUS, (0,0,0), -1)
    angle_marker = (centre[0]+round(ANGLE_MARK_SIZE*np.cos(robot_pose[2])),centre[1]+round(ANGLE_MARK_SIZE*np.sin(robot_pose[2])))
    cv.line(img,centre,angle_marker,(0,0,0),2)
    
    # Draw Fruits
    for fruit in search_list:
        for i in range(5):
            if fruit == fruits_list[i]:
                centre = (translate_coord(fruits_true_pos[i][0]),translate_coord(fruits_true_pos[i][1]))            
                cv.circle(img,centre,FRUIT_RADIUS, (120,120,0), -1)
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(img, fruit, (centre[0]+25,centre[1]), font, 0.4, (120, 120, 0), 1)

    cv.imshow("Map",img)
    


def click_event(event, x, y, flags, param):
    # Create a white image
    if event == cv.EVENT_LBUTTONDOWN:
        x = (x-IMAGE_SIZE/2)/(IMAGE_SIZE/(DIMENSION))
        y = (y-IMAGE_SIZE/2)/(IMAGE_SIZE/(DIMENSION))
        print(round(x,3), ' ', round(y,3))

    
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Fruit searching")
    parser.add_argument("--map", type=str, default='M4_true_map_5fruits.txt')
    args, _ = parser.parse_known_args()

    fruits_list, fruits_true_pos, aruco_true_pos = read_true_map(args.map)
    search_list = read_search_list()
    print_target_fruits_pos(search_list, fruits_list, fruits_true_pos)
    draw()
    
    cv.setMouseCallback('Map', click_event)
    k = cv.waitKey(0)
