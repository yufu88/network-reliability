import numpy as np
import math

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
def minimal_path(start_node, end_node, node, demand, time):
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
    flow = []
    for i in range(len(P)):
        ceiling = math.ceil(demand/(time-P_lead_time[i]))
        if P_max_capacity[i] >= ceiling:
            mp.append(P[i])
            max_cap.append(P_max_capacity[i])
            flow.append(ceiling)
    return(np.array(mp), max_cap, arc_max_capacity.flatten(),flow)


def echeck(a1, a2):
    e_array = np.full(len(a1), False, dtype=bool)
    for i in range(len(a1)):
        if a1[i]<=a2[i]:
            e_array[i] = True
    return(all(e_array))

def mp_compare(mp_array):
    c_cap = np.reshape(MP[3], (len(MP[3]),1))*MP[0]
    index = []
    for i in range(len(c_cap)-1):
        for j in range(i+1, len(c_cap)):
            if echeck(c_cap[i], c_cap[j]):
                index.append(j)
                c_cap = np.delete(c_cap, index, 0)    
    return(c_cap)

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

# prob of each iteration
def TM_caculator(d_MP, index):
    PR = probability(d_MP[index])
    Y = 0
    for i in range(index):
        temp = compare(d_MP[index], d_MP[i])
        Y = max(Y, probability(temp))
    TM = PR-Y
    return TM

# sum prob
def RSDP(d_MP):
    index = 0
    prob = 0
    for i in range(len(d_MP)):
        prob += TM_caculator(d_MP, index)
        print("T",TM_caculator(d_MP, index))
        index+=1
        
    return round(prob,4)

#network building
node_num = 5
d = 8
t = 12

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

MP = minimal_path(0,4, node, d, t)
arc_index = np.argwhere(node==1)
arc_num = len(arc_index)
d_MP = mp_compare(MP)
print(d_MP)

for i in range

"""
Q = []

def find(pointer, value, v_num, F=[], total=0):
      if pointer == v_num:
          F.append(value-total)
          print(F)
          Q.append(list(F))
          del F[-1]
          return Q
      else:
          for i in range(value-total): 
              if total + i <=value:
                  F.append(i)
                  find(pointer+1, value, v_num, F, total+i)
                  del F[-1]
              else:
                  break

find(1, d, len(MP[3]))

"""