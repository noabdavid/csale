from Dijkstar.dijkstar.algorithm import find_path
import Graphs
import numpy as np
import math

def clucb(delta, epsilon, graph, start, dest):
    # initialize
    S = Graphs.get_edges(graph)
    n = len(S)
    oracle_counter = 0
    global mu_t
    global mu_tilde_t
    mu_t = dict.fromkeys(S, 0)
    mu_tilde_t = dict.fromkeys(S, 0)
    N_t = dict.fromkeys(S, 1)
    rad_t = dict.fromkeys(S, 0)
    # pull each edge once
    for e in S:
        mu_t[e] = np.random.binomial(n=1, p=e[2])
    t = n
    while True:
        M = find_path(graph, start, dest, cost_func=cost_func)
        oracle_counter = oracle_counter + 1
        for e in S:
            rad_t[e] = compute_radius(delta, n, t, N_t[e])
            if e in M.edges:
                mu_tilde_t[e] = mu_t[e] + rad_t[e]
            else:
                mu_tilde_t[e] = mu_t[e] - rad_t[e]


        Mtilde = find_path(graph, start, dest, cost_func=cost_func_tilde)
        oracle_counter = oracle_counter + 1
        M_score = sum([mu_tilde_t[a] for a in M.edges])
        Mtilde_score = sum([mu_tilde_t[a] for a in Mtilde.edges])
        # this is a minimization problem
        if M_score - Mtilde_score <= epsilon:
            return M.edges, sum(N_t.values()), oracle_counter

        symDif = set(M.edges).symmetric_difference(set(Mtilde.edges))
        
        max_rad = - math.inf
        p = None
        for e in symDif:
            if rad_t[e] > max_rad:
                p = e
                max_rad = rad_t[e]

        mu_t[p] = (mu_t[p] * N_t[p] + np.random.binomial(n=1, p=p[2])) / (N_t[p] + 1)
        N_t[p] = N_t[p] + 1
        t = t + 1


def compute_radius(delta, n, t, T, R=1):
    return R * math.sqrt(2 * math.log(4 * n * (t ** 3) / delta) / T)

def cost_func(u, v, e, prev_e):
    return mu_t[e]

def cost_func_tilde(u, v, e, prev_e):
    return max(mu_tilde_t[e], 0)

