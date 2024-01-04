import json
import numpy as np
with open("lab_output/slam.txt", 'r') as fd:
    usr_dict = json.load(fd)
    aruco_dict = {}
    taglist = []
    

    
    for (i, tag) in enumerate(usr_dict['taglist']):
        taglist.append(tag)

    markers = np.zeros((2,len(taglist)))
    x = []
    y = []
    for i in range(len(taglist)):
        x.append(usr_dict['map'][0][i])      
        y.append(usr_dict['map'][1][i])
    markers[0] = x
    markers[1] = y

    covariance = np.zeros((2*len(taglist),2*len(taglist)+3))
    for i in range(3,2*len(taglist)):
        for j in range(3,2*len(taglist)):
                covariance[i][j] = (usr_dict['covariance'][i][j])

        # markers[0].append(usr_dict['map'][0][i])
        # markers[1].append(usr_dict['map'][1][i])
        # print(usr_dict['map'][0][i])
        # for j in range(8):
        
        # aruco_dict[tag] = np.reshape([usr_dict['map'][0][i], usr_dict['map'][1][i]], (2, 1))
    # for i in enumerate(usr_dict['covariance']):
    #     covariance[0].append(usr_dict['covariance'][0][i])
    
print((markers))
print((taglist))
print(covariance)