def solution_finder(pointer,  Q, c_array, P=[]):
    if pointer == len(c_array):
        check = np.full(3, False, dtype=bool)
        
        # 1. flow_sum = d
        s1= np.sum(flow[:count_2])
        s2= np.sum(flow[count_2:])
        if s1== demand_1 & s2== demand_2:
            check[0] = True
        # 2. flow <= mp_max_capacity
        check[1] = (flow <= mp_max_capacity).all()
        # 3. flow_sum <= arc_max_capacity
        check[2] = (np.sum(flow.reshape(np.size(N,0),1)*N, axis=0) <= arc_max_capacity).all()

        if check.all():
            Q.append(P)
    
    else:
        for i in range(c_array[pointer]+1):
            
            solution_finder(pointer+1, Q, c_array, P+[i])