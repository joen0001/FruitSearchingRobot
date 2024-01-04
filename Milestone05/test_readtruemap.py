import numpy as np
import json


with open("M5_lab1_sim_map.txt", 'r') as fd:
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

        print("fruit_list =",fruit_list)
        print("fruit_true_pos =",fruit_true_pos)
        print("aruco_true_pos =",aruco_true_pos)