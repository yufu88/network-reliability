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
        #print(P)
        check = np.full(3, False, dtype=bool)
        
        # 1. flow_sum = d
        P = np.array(P, dtype=int)
        
        s1= np.sum(P[:count_2])
        s2= np.sum(P[count_2:count_3])
        s3= np.sum(P[count_3:count_4])
        s4= np.sum(P[count_4:])

        if (s1== demand_1) & (s2== demand_2) & (s3==demand_3) & (s4==demand_4):
            check[0] = True
        # 2. flow <= mp_max_capacity
        check[1] = (P <= mp_max_capacity).all()
        # 3. flow_sum <= arc_max_capacity
        check[2] = (P@N[:pointer] <= arc_max_capacity).all()
        if check.all():
            Q.append(P)
    else :
        for i in range(c_array[pointer]+1):
            R = np.array(P, dtype=int)
            # 3. flow_sum <= arc_max_capacity
            if pointer==0 or (R@N[:pointer] <= arc_max_capacity).all():
                solution_finder(pointer+1, Q, c_array, P+[i])
            else:
                break

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
node_num = 5
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

# list of arcs
arc_index = np.argwhere(node==1)
arc_num = len(arc_index)
arc_max_capacity = (np.count_nonzero(arc_capacity, axis=1)-1).reshape(1,len(arc_index))

N = np.empty((0,arc_num),dtype=int)

N = np.append(N, minimal_path(0,3),axis=0)
count_2 = len(N)
N = np.append(N, minimal_path(1,3),axis=0)
count_3 = len(N)
N = np.append(N, minimal_path(0,4),axis=0)
count_4 = len(N)
N = np.append(N, minimal_path(1,4),axis=0)

# find maximal capacity of each mp
cap = N*arc_max_capacity
mp_max_capacity = [np.min(c[np.nonzero(c)]) for c in cap]

demand_1=1
demand_2=2
demand_3=2
demand_4=2

# generate feasible solutions
feasible_solutions = []

solution_finder(0, feasible_solutions, mp_max_capacity)

# current capacity
current_capacity=[]
for f in feasible_solutions:
        current_capacity.append(f@N)
current_capacity = np.array(current_capacity)

#d_MP
d_MP = d_MP(current_capacity)

#caculate RSDP
success_rate = RSDP(d_MP)
print(success_rate)