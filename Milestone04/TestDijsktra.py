import math
import heapq
import matplotlib.pyplot as plt
import numpy as np
import json

class Env:
    def __init__(self, aruco_list, fruit_list):
        ### Size of the map and add one more to the width and height
        self.x_range = 31  
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

            for i in np.arange(Aruco_x, Aruco_x+1):
                obs.add((i, Aruco_y))

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
        #self.aruco_list = aruco_list

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

def generate_path(current_pose, current_fruit):
    s_start = current_pose
    s_goal = current_fruit 
    return 0


def main():

    #list = generate_path()

    fruit_list, fruit_true_pos, aruco_true_pos = read_true_map("M4_true_map_5fruits.txt")

    robot_pose = [0.0,0.0]
    target_pose = fruit_true_pos[-1]
    
    
    s_start = tuple(robot_pose) # this should be the robot's current pose
    s_goal = tuple(target_pose) # this should be the current fruit of interest 

    print(fruit_list)
    print(fruit_true_pos)
    print(aruco_true_pos)

    # Change to aruco_true_pos
    dijkstra = Dijkstra(s_start, s_goal, 'None', aruco_true_pos, fruit_true_pos[0:-2])
    plot = Plotting(s_start, s_goal, aruco_true_pos, fruit_true_pos[0:-2])

    path, visited = dijkstra.searching()
    print(path)# path store all of the waypoints generated by the algorithm 
    plot.animation(path, visited, "Dijkstra's")  # animation generate


if __name__ == '__main__':
    main()
