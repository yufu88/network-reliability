import numpy as np
from itertools import product
from math import factorial

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
def minimal_path(start_node, end_node):
    pathset = print_all_path(start_node, end_node)
    mp = np.zeros((len(pathset), len(arc_index)), dtype=int)
    for p in range(len(pathset)):
        for i in range(len(pathset[p])-1):
            arc = [pathset[p][i], pathset[p][i+1]]
            arc_used = np.where((arc_index == arc).all(axis=1))
            mp[p][arc_used[0]] = 1
    return(mp)

def solution_finder(pointer,  Q, c_array, P=[]):
    if pointer == len(c_array):
        
        check = np.full(3, False, dtype=bool)
        
        # 1. flow_sum = d
        P = np.array(P, dtype=int)
        
        s1= np.sum(P[:count_2])
        s2= np.sum(P[count_2:])
        if s1== demand_1 & s2== demand_2:
            check[0] = True
        # 2. flow <= mp_max_capacity
        check[1] = (P <= mp_max_capacity).all()
        # 3. flow_sum <= arc_max_capacity
        check[2] = (np.sum(P.reshape(np.size(N,0),1)*N[:pointer], axis=0) <= arc_max_capacity).all()

        if check.all():
            Q.append(P)
    else :
        for i in range(c_array[pointer]+1):
            R = np.array(P, dtype=int)
            # 3. flow_sum <= arc_max_capacity
            if pointer==0 or (np.sum(R.reshape(pointer,1)*N[:pointer], axis=0) <= arc_max_capacity).all():
                solution_finder(pointer+1, Q, c_array, P+[i])
            else:
                break

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

# caculate prob
def probability(cap_array):
    TM=1
    for i in range(arc_num):
        prob_k = 0
        for k in range(cap_array[i],arc_capacity.shape[1]):
            prob_k += arc_capacity[i][k]
        TM *= prob_k
    return round(TM,4)

# array comparison
def compare(array1, array2):
    temp = np.zeros(arc_num, dtype=int)
    for i in range(arc_num):
        temp[i] = max(array1[i],array2[i])
    return temp

def comparison(index, base_line, vector=[]):
    if len(vector) == 0:
        vector = base_line

    if index == 0:
        return probability(vector)
    else:
        vector = compare(base_line, vector)
        return comparison(index-1, base_line, vector)

def RSDP(d_MP):
    index = len(d_MP)
    prob = 0
    for i in range(len(d_MP)):
        if i==0:
            TM = probability(d_MP[i])
            prob += TM
        else:
            TM = probability(d_MP[i]) - comparison(i, d_MP[i])
            prob += TM
    return prob


def rate(bag_num, failure, success_rate):
    # C ( n+m-1 m )
    C = factorial(bag_num+failure-1)/(factorial(failure)*factorial(bag_num-1))
    rate = C*pow(success_rate,bag_num)*pow((1-success_rate),failure)
    return round(rate,4)

def transfer_num(expection, r):
    return round(expection/r,0)

# num of nodes
node_num = 8
node = np.zeros((node_num, node_num), dtype=int)

network(0,2)
network(0,4)
network(1,2)
network(1,4)
network(2,3)
network(2,4)
network(2,5)
network(3,2)
network(3,4)
network(3,5)
network(3,6)
network(3,7)
network(4,2)
network(4,3)
network(4,5)
network(5,2)
network(5,3)
network(5,4)
network(5,6)
network(5,7)

arc_capacity = np.array([
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.25,    0.25],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.25,    0.25],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.25,    0.25],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0],
[0.25,  0.25,  0.5,    0]
])

# list of arcs
arc_index = np.argwhere(node==1)
arc_num = len(arc_index)
arc_max_capacity = (np.count_nonzero(arc_capacity, axis=1)-1).reshape(1,len(arc_index))
"""
N = np.empty((0,arc_num),dtype=int)

count_2 = 0
count_4 = 0
for i in [3,4]:
    for j in range(2):
        N = np.append(N, minimal_path(j,i),axis=0)
        if i==3:
            count_2 += len(minimal_path(j,i))
        else: 
            count_4 += len(minimal_path(j,i))

# find maximal capacity of each mp
cap = N*arc_max_capacity
mp_max_capacity = [np.min(c[np.nonzero(c)]) for c in cap]

# generate feasible solutions
demand_1 = 3
demand_2 = 3

feasible_solutions = []

solution_finder(0, feasible_solutions, mp_max_capacity)

# current capacity
current_capacity = current_capacity_get(feasible_solutions, N)

#d_MP
d_MP = d_MP(current_capacity)

#caculate RSDP
success_rate = RSDP(d_MP)
print(success_rate)
"""