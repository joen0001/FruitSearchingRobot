import json
import numpy as np

search_list = ["strawberry","lemon","orange"]
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
        print("len =",len(fruit_pos))

# {"apple_0": {"y": -1.0278397122150909, "x": 1.2414522976964}, "lemon_0": {"y": 1.147201024708971, "x": 1.1202306603853311}, "pear_0": {"y": -0.8506232427032367, "x": 0.6596834885857927}, "orange_0": {"y": 0.5717372486782871, "x": -0.7558613981681271}, "strawberry_0": {"y": 0.699319913262176, "x": 1.105209620319834}}
fruits_in_order = []
for fruit in search_list:
    for i in range(len(fruit_list)):
        if fruit == fruit_list[i]:
            #print('{}) {} at [{}, {}]'.format(n_fruit, fruit, np.round(fruit_true_pos[i][0], 1), np.round(fruit_true_pos[i][1], 1)))
            fruits_in_order.append([fruit_pos[i][0],fruit_pos[i][1]])
print("fruits in order,",fruits_in_order)