import networkx as nx
from networkx.algorithms.matching import max_weight_matching, maximal_matching
import Graphs
import numpy as np
import math

def clucb(delta, epsilon, edges_info):
    # initialize
    edges = edges_info[0]
    weights = edges_info[1]
    S = edges
    n = len(S)
    oracle_calls = 0
    global mu_t
    global mu_tilde_t
    mu_t = dict.fromkeys(S, 0)
    mu_tilde_t = dict.fromkeys(S, 0)
    N_t = dict.fromkeys(S, 1)
    rad_t = dict.fromkeys(S, 0)
    g_hat = nx.Graph()
    for e in S:
        mu_t[e] = np.random.binomial(n=1, p=weights[e])
        g_hat.add_edge(e[0], e[1], weight=mu_t[e])
    t = n
    while True:
        g_tilde = nx.Graph()

        M = max_weight_matching(g_hat)
        M = [Graphs.edge(e) for e in M]
        oracle_calls = oracle_calls + 1
        for e in S:
            rad_t[e] = compute_radius(delta, n, t, N_t[e])
            if e in M:
                mu_tilde_t[e] = mu_t[e] - rad_t[e]
            else:
                mu_tilde_t[e] = mu_t[e] + rad_t[e]

            g_tilde.add_edge(e[0], e[1], weight=mu_tilde_t[e])


        Mtilde = max_weight_matching(g_tilde)
        Mtilde = [Graphs.edge(e) for e in Mtilde]
        oracle_calls = oracle_calls + 1
        #print('Tilde shortest path in round ' + str(t - n + 1) + ': ' + str(Mtilde))
        M_score = sum([mu_tilde_t[a] for a in M])
        Mtilde_score = sum([mu_tilde_t[a] for a in Mtilde])

        if Mtilde_score - M_score <= epsilon:
            return M, sum(N_t.values()), oracle_calls

        symDif = set(M).symmetric_difference(set(Mtilde))
        #print('Symmetric difference in round ' + str(t - v + 1) + ': ' + str(symDif))
        max_rad = - math.inf
        p = None
        for e in symDif:
            if rad_t[e] > max_rad:
                p = e
                max_rad = rad_t[e]

        mu_t[p] = (mu_t[p] * N_t[p] + np.random.binomial(n=1, p=weights[p])) / (N_t[p] + 1)
        g_hat.remove_edge(p[0], p[1])
        g_hat.add_edge(p[0], p[1], weights=mu_t[p])
        #print('edge ' + str(p) + ' is pulled in round ' + str(t) + ' and its expectation is now ' + str(mu_t[p]))
        N_t[p] = N_t[p] + 1
        t = t + 1


def compute_radius(delta, n, t, T, R=1):
    return R * math.sqrt(2 * math.log(4 * n * (t ** 3) / delta) / T)



