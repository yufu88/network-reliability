import numpy as np

# connection matrix of network
def network(from_node, to_node):
    node[from_node][to_node] = 1

# find all mp between s&t in network
def path_finder(start_node, end_node, visited, path, all_paths):
        visited[start_node] =True
        path.append(start_node)

        if(start_node == end_node):
            all_paths.append(path.copy())
        else:
            for i in range(node_num):
                if(node[start_node][i])==1:
                    if visited[i] == False:
                        path_finder(i, end_node, visited, path, all_paths)
        path.pop()
        visited[start_node] = False

def print_all_path(start_node, end_node):
    visited = [False]*(len(node))
    path = list()
    all_paths = list()
    path_finder(start_node, end_node, visited, path, all_paths)
    return(all_paths)

# find all mp
def minimal_path(start_node, end_node, node):
    pathset = print_all_path(start_node, end_node)
    arc_index = np.argwhere(node==1)
    arc_num = len(arc_index)

    P = np.zeros((len(pathset), arc_num), dtype=int)
    for p in range(len(pathset)):
        for i in range(len(pathset[p])-1):
            arc = [pathset[p][i], pathset[p][i+1]]
            arc_used = np.where((arc_index == arc).all(axis=1))
            P[p][arc_used[0]] = 1

    arc_max_capacity = (np.count_nonzero(arc_capacity, axis=1)-1).reshape(1,arc_num)
    P_lead_time = np.sum(P*lead_time, axis=1)
    P_max_capacity = [np.min(c[np.nonzero(c)]) for c in P*arc_max_capacity]

    mp = []
    max_cap = []
    for i in range(len(P)):
        if P_lead_time[i]+(demand/P_max_capacity[i]) <= time:
            mp.append(P[i])
            max_cap.append(P_max_capacity[i])
    return(np.array(mp), max_cap, arc_max_capacity)

def echeck(a1, a2):
    e_array = np.full(3, False, dtype=bool)
    for i in range(len(a1)):
        if a1[i]<=a2[i]:
            e_array[i] = True
    return(all(e_array))

# value = demand, v_num=mp_num
def find(pointer, value, v_num, mp_array, F=[], total=0):
    if pointer == (v_num-1):
        F.append(value-total)

        check = np.full(3, False, dtype=bool)
        #1
        if np.sum(F) == value:
            check[0]=True
        
        #2 flow <= mp_max_capacity
        check[1] = echeck(F,mp_array[1])
        
        #3 flow_sum <= arc_max_capacity
        flow_sum = np.array(np.sum(np.reshape(F,(len(F),1))*MP[0],axis=0))
        print(flow_sum)
        #check[2] = echeck(flow_sum, mp_array[2])
        
        Q.append(list(F))
        del F[-1]
        return Q
    else:
        for i in range(value-total): 
            if total + i <=value:
                F.append(i)
                find(pointer+1, value, v_num, mp_array, F, total+i)
                del F[-1]
            else:
                break

#network building
node_num = 5
demand = 8
time = 9

node = np.zeros((node_num, node_num), dtype=int)
network(0,1)
network(1,2)
network(0,2)
network(1,4)
network(1,3)
network(2,4)
network(2,3)
network(3,4)

arc_capacity = np.array([
[0.05,  0.05,  0.1,    0.8, 0, 0],
[0.05,  0.1,   0.85,   0,   0, 0],
[0.05,  0.05,  0.1,    0.8, 0, 0],
[0.1,   0.9,   0 ,     0,   0, 0], 
[0.1,   0.9,   0 ,     0,   0, 0], 
[0.05,  0.1,   0.1,    0.1, 0.1, 0.55], 
[0.05,  0.05,  0.1,    0.2, 0.6, 0],
[0.05,  0.05,  0.1,    0.8, 0, 0]
])

lead_time = np.array([2,3,1,1,3,2,2,1])

MP = minimal_path(0,4, node)

Q=[]
find(0,8,len(MP[0]),MP)
print(MP[0])

