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
            feasible_solutions.append(flow)
    feasible_solutions = np.array(feasible_solutions)
    return(feasible_solutions)

def current_capacity_get(feasible_solutions, N):
    current_capacity = []
    for f in feasible_solutions:
        current_capacity.append(np.sum(f.reshape(np.size(N,0),1)*N, axis=0))
    current_capacity = np.array(current_capacity)
    return(current_capacity)

def d_MP(current_capacity):
    I=[]
    for i in range(len(current_capacity)-1):
        for j in range(i+1,len(current_capacity)):
            if i not in I:
                if((current_capacity[i] <= current_capacity[j]).all()):
                    I.append(j)
    d_MP = np.delete(current_capacity,I,0)
    return(d_MP)

# num of nodes
node_num = 4
node = np.zeros((node_num, node_num), dtype=int)

network(0,1)
network(0,2)
network(1,3)
network(2,3)
network(1,2)
network(2,1)

arc_capacity = np.array([
[0.05, 0.1, 0.25,  0.6],
[0.1,  0.9,  0,    0],
[0.1,  0.9,  0,    0],
[0.1 , 0.3,  0.6,  0],
[0.1,  0.9,  0,    0],
[0.05, 0.25, 0.7,  0]
])

# list of arcs
arc_index = np.argwhere(node==1)
arc_num = len(arc_index)
arc_max_capacity = (np.count_nonzero(arc_capacity, axis=1)-1).reshape(1,len(arc_index))

N = minimal_path(0,3)

# find maximal capacity of each mp
cap = N*arc_max_capacity
mp_max_capacity = [np.min(c[np.nonzero(c)]) for c in cap]

# generate feasible solutions
demand = 3
feasible_solutions = feasible_solutions_get(demand, mp_max_capacity, arc_max_capacity)

# current capacity
current_capacity = current_capacity_get(feasible_solutions, N)

#d_MP
d_MP = d_MP(current_capacity)


prob_n = 0
prob_total = 1

#print(d_MP[0])
for i in range(arc_num):
    print('d_mp:',d_MP[2][i])
    for k in range(d_MP[2][i],arc_capacity.shape[1]):
        print(i,k)
        print('capacity', arc_capacity[i][k])


print(arc_capacity[0][3])
"""for k in range(arc_num):
    for i in range(d_MP[0][k], len((arc_capacity[0]))):
        prob_n += arc_capacity[k][i]
        print(arc_capacity[k][i])
    print("n")
    prob_total = prob_total * prob_n
    prob_n = 0
    
"""
