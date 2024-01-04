import numpy as np
import json

# Load fruit positions from targets.txt
with open("lab_output/targets.txt", 'r') as fd:
        gt_dict = json.load(fd)
        fruit_list = []
        fruit_pos = []

        for key in gt_dict:
            x = np.round(gt_dict[key]['x'], 1)*10
            y = np.round(gt_dict[key]['y'], 1)*10
            fruit_list.append(key[:-2])
            if len(fruit_pos) == 0:
                fruit_pos = np.array([[x, y]])
            else:
                fruit_pos = np.append(fruit_pos, [[x, y]], axis=0)

        print("fruit_list =", fruit_list)
        print("fruit_pos =", fruit_pos)

# Load aruco marker positions from slam.txt
with open("lab_output/slam.txt", 'r') as fd:
        usr_dict = json.load(fd)
        taglist = []
        for (i, tag) in enumerate(usr_dict['taglist']):
            taglist.append(tag)

        aruco_pos = np.zeros((len(taglist),2))
        for i in range(len(taglist)):
            aruco_pos[i][0] = usr_dict['map'][0][i]
            aruco_pos[i][1] = usr_dict['map'][1][i]
        aruco_pos = np.round(10*aruco_pos)
print("aruco_pos =", aruco_pos)
