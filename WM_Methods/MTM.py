import cvxpy as cp

def optimise(a,b,c,d,cons,weights):
    m = a.shape[0]
    n = a.shape[1]

    x = cp.Variable(n)

    cost = cp.sum_squares(a@x-b)

    constraints = [c@x==d, x>0]
    prob = cp.Problem(cp.Minimize(cost), constraints)

    # The optimal objective value is returned by prob.solve()`.
    # OSQP, ECOS, ECOS_BB, MOSEK, CBC, CVXOPT, NAG, GUROBI, and SCS
    result = prob.solve(verbose=True, solver=cp.ECOS)

    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % prob.value)
    for variable in prob.variables():
        print("Variable %s: value %s" % (variable.name(), variable.value))

    # The optimal value for x is stored in `x.value`.
    g_ij = x.value
    return g_ij

def mix_adj(trans,T,S,V):
    # This is the temperature and salinity the late water masses acheive by mixing the early water masses
    Tmixed = np.matmul(Tmatrix,xxx)/(flat_Vol_blc2/xnorming) #Changed from flat_vol_blc1
    Tmixed[~np.isfinite(Tmixed)]= np.nan
    Tmixed[Tmixed>100] = np.nan
    Smixed = np.matmul(Smatrix,xxx)/(flat_Vol_blc2/xnorming) #Changed from flat_vol_blc1
    Smixed[~np.isfinite(Smixed)]= np.nan
    Smixed[Smixed>10**4] = np.nan

    # Now the necessary heat and salt adjustment is simply the difference
    # between this and what we actually get
    dTV_adj = (flat_T_blc2-Tmixed)*flat_Vol_blc2
    dSV_adj = (flat_S_blc2-Smixed)*flat_Vol_blc2

    # Dividing by the volume we get the average temperature change.
    T_Av_adj = dTV_adj/(flat_Vol_blc2)
    T_Av_adj[np.isnan(T_Av_adj)] = 0
    S_Av_adj = dSV_adj/(flat_Vol_blc2)
    S_Av_adj[np.isnan(S_Av_adj)]= 0
    G = np.zeros((flat_Vol_blc1.shape[0],flat_Vol_blc2.shape[0]))
    ix=0
    for i in tqdm(range(flat_Vol_blc1.size)):
        for j in range(flat_Vol_blc2.size):
            if globe == False:
                if connected[i,j]==1:
                    G[i,j] = xxx[ix] #vertex ix brings temp of WM i to WM j
                    ix=ix+1
            else:
                if connected[i,j]==1:
                    G[i,j] = xxx[ix]
                    ix=ix+1    

    return g_i