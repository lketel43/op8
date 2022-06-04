import numpy as np

def proj_kiwiel_positive(a, y, l, u):
    """ projects the vector "a" onto the intersection of H and C
        where C is a box defined by l <= x <= u (possibly infinites)
        and H is the hyperplane of equation <x,y>=0
        we assume here that y > 0 
        
        Kiwiel, K. C. (2007). On linear-time algorithms for the continuous quadratic knapsack problem. 
        Journal of Optimization Theory and Applications, 134(3), 549-554.
    """
    # definitions
    d = a.shape[0]
    x = np.zeros(d) # the projection
    candidate = lambda t : np.minimum(u, np.maximum(l, a-t*y))
    constraint = lambda t : y.T @ candidate(t)
    
    # we handle special cases
    if np.min(y) < 0:
        print("Error the vector *y* should be positive")
        return
    elif np.min(y) == 0: 
        idx = (y==0)
        idx_c = (y > 0)
        x[idx] = np.minimum(u[idx], np.maximum(l[idx], a[idx])) # for those idx we immediately get the solution
        x[idx_c] = proj_kiwiel(a[idx_c], y[idx_c]) # for the others we recursively call the projection
        return x
    elif np.min(y) > 0: # main case
        T = np.concatenate( ( (a-l)/y, (a-u)/y ) )
        t_U = - np.inf
        t_L = np.inf
        counter = 0
        while T.shape[0] > 0:
            counter = counter + 1
            t = np.median(T)
            if constraint(t) == 0:
                return candidate(t)
            elif constraint(t) > 0:
                t_L = t
                T = T[T > t]
            else: # constraint(t) < 0
                t_U = t
                T = T[T < t]
        if T.shape[0] == 0: # at this point should be the case but who knows
            t_sol = t_L - constraint(t_L)*(t_U - t_L)/(constraint(t_U)-constraint(t_L))
            return candidate(t_sol)            
        else:
            print("unknown problem")
            return
        
def proj_kiwiel(a, y, l, u):
    """ projects the vector "a" onto the intersection of H and C
        where C is a box defined by l <= x <= u (possibly infinites)
        and H is the hyperplane of equation <x,y>=0
        
        Kiwiel, K. C. (2007). On linear-time algorithms for the continuous quadratic knapsack problem. 
        Journal of Optimization Theory and Applications, 134(3), 549-554.
    """
    dim = a.shape[0]
    idx = (y < 0)
    # we change the signs wherever needed so that y>0
    a_pos = a.copy()
    y_pos = y.copy()
    l_pos = l.copy()
    u_pos = u.copy()
    a_pos[idx] = - a_pos[idx]
    y_pos[idx] = - y_pos[idx]
    tmp = l_pos[idx]
    l_pos[idx] = - u_pos[idx]
    u_pos[idx] = - tmp
    # we call the projection in the case y >0
    p = proj_kiwiel_positive(a_pos, y_pos, l_pos, u_pos)
    # we revert the signs
    p[idx] = -p[idx]
    return p    