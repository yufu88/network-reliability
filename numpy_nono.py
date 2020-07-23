import numpy as np
from itertools import product

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

def print_all_path( start_node, end_node):
    visited = [False]*(len(node))
    path = list()
    all_paths = list()

    path_finder(start_node, end_node, visited, path, all_paths)
    return(all_paths)

# find all mp
def minimal_path(start_node, end_node):
    pathset = print_all_path(start_node, end_node)
    mp = np.zeros((len(pathset), len(arc_index)), dtype=int)
    for p in range(len(pathset)):
        for i in range(len(pathset[p])-1):
            arc = [pathset[p][i], pathset[p][i+1]]
            arc_used = np.where((arc_index == arc).all(axis=1))
            mp[p][arc_used[0]] = 1
    return(mp)

def feasible_solutions_get(demand, mp_max_capacity, arc_max_capacity):
    feasible_solutions = []
    options = [list(range(c+1)) for c in mp_max_capacity]

    for flow in np.array(list(product(*options))):
        check = np.full(3, False, dtype=bool)

        # 1. flow_sum = d
        if(np.sum(flow) == demand):
            check[0] = True

        # 2. flow <= mp_max_capacity
        check[1] = (flow <= mp_max_capacity).all()

        # 3. flow_sum <= arc_max_capacity
        check[2] = (np.sum(flow.reshape(np.size(N,0),1)*N, axis=0) <= arc_max_capacity).all()


        if(check.all()==True):
            a=1
            feasible_solutions.append(flow)
    feasible_solutions = np.array(feasible_solutions)
    return(feasible_solutions)

def current_capacity_get(feasible_solutions, N):
    current_capacity = []
    for f in feasible_solutions:
        current_capacity.append(np.sum(f.reshape(np.size(N,0),1)*N, axis=0))
    current_capacity = np.array(current_capacity)
    return(current_capacity)

# num of nodes
node_num = 4
node = np.zeros((node_num, node_num), dtype=int)

network(0,1)
network(0,2)
network(1,3)
network(2,3)
network(1,2)
network(2,1)

# list of arcs
arc_index = np.argwhere(node==1)
arc_max_capacity = np.array([2,3,2,3,3,3])

N = minimal_path(0,3)

# find maximal capacity of each mp
cap = N*arc_max_capacity
mp_max_capacity = [np.min(c[np.nonzero(c)]) for c in cap]

# generate feasible solutions

feasible_solutions = feasible_solutions_get(5, mp_max_capacity, arc_max_capacity)

# current capacity
current_capacity = current_capacity_get(feasible_solutions, N)
print(feasible_solutions)
print(current_capacity)