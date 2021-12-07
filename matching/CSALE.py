import numpy as np
import math
import networkx as nx
from networkx.algorithms.matching import max_weight_matching, maximal_matching
import Graphs





def csale(epsilon, delta, edges_info):
    global accept
    accept = set()
    edges = edges_info[0]
    graph_copy = nx.Graph()
    graph_copy.add_edges_from(edges)
    v = graph_copy.number_of_nodes()
    global weights
    weights = edges_info[1]
    S = set(edges)
    global mu_t
    mu_t = dict.fromkeys(S, 0)
    N_t = dict.fromkeys(S, 0)
    epsilon_t = epsilon
    t = 1
    oracle_calls = 0
    d_t = len(maximal_matching(graph_copy))
    print('maximal cardinality matching in round ' + str(t) + ': ' + str(d_t))
    T = math.ceil(math.log2(2 * d_t))

    while epsilon_t > (epsilon / (d_t - len(accept))):

        theta = epsilon_t * (d_t - len(accept))
        print('Threshold in round ' + str(t) + ": " + str(theta))

        # sample every edge in S and update mu
        g_tmp = nx.Graph()
        for e in S:
            mu_t[e], N_t[e] = sample(e, epsilon_t / 2, delta / (T * len(S)), mu_t[e], N_t[e])
            g_tmp.add_edge(e[0], e[1], weight=mu_t[e])


        for e in accept:
            g_tmp.add_edge(e[0], e[1], weight=v)

        Mhat = max_weight_matching(g_tmp)
        oracle_calls = oracle_calls + 1
        print('candidtae edges: ' + str(Mhat))
        Mhat_score = sum([mu_t[Graphs.edge(a)] for a in Mhat])
        for e in S.intersection(Mhat):
            g_tmp.remove_edge(e[0], e[1])
            Mtilde = max_weight_matching(g_tmp)
            oracle_calls = oracle_calls + 1
            Mtilde_score = sum([mu_t[Graphs.edge(a)] for a in Mtilde])
            g_tmp.add_edge(e[0], e[1], weight=mu_t[e])
            gap_e = Mhat_score - Mtilde_score  # this is a maximization problem
            print('The gap of edge ' + str(e) + ': ' + str(gap_e))
            if gap_e > theta:
                print('accept edge ' + str(e))
                accept.update({e})
                S.remove(e)
                # e = (u,v), eliminate edges (u,*) and (*,v)
                to_remove = set()
                for a in S:
                    if Graphs.share_node(e, a):
                        to_remove.add(a)
                        graph_copy.remove_edge(a[0], a[1])
                S = S - to_remove
                #print('eliminate edges: ' + str(to_remove))
                d_t = len(maximal_matching(graph_copy))
                print('maximal cardinality matching in round ' + str(t) + ': ' + str(d_t))
                theta = epsilon_t * (d_t - len(accept))
                print('Threshold in round ' + str(t) + ": " + str(theta))
        if Graphs.is_maximal_matching(accept, graph_copy):
            print('Terminating before t=T')
            return accept, sum(N_t.values()), len(accept), oracle_calls
        t = t + 1
        epsilon_t = epsilon_t / 2

    epsilon_last = epsilon / (d_t - len(accept))
    # sample every a in S and update mu
    g_tmp = nx.Graph()
    for e in S:
        mu_t[e], N_t[e] = sample(e, epsilon_last / 2, delta / (T * len(S)), mu_t[e], N_t[e])
        g_tmp.add_edge(e[0], e[1], weight=mu_t[e])

    for e in accept:
        g_tmp.add_edge(e[0], e[1], weight=v)

    Mhat = max_weight_matching(g_tmp)
    oracle_calls = oracle_calls + 1
    return Mhat, sum(N_t.values()), len(accept), oracle_calls

def naive(epsilon, delta, edges_info):
    edges = edges_info[0]
    global weights
    weights = edges_info[1]
    S = edges
    graph = nx.Graph()
    graph.add_edges_from(edges)
    d = len(maximal_matching(graph))
    global mu_t
    mu_t = dict.fromkeys(S, 0)
    N_t = dict.fromkeys(S, 0)

    g = nx.Graph()
    for e in S:
        mu_t[e], N_t[e] = sample(e, epsilon / (2 * d), delta / len(S), 0, 0)
        g.add_edge(e[0], e[1], weight=mu_t[e])

    M = max_weight_matching(g)
    return M, sum(N_t.values())

def sample(edge, epsilon, delta, x, n):
    n_new = N(epsilon / 2, delta)
    samp_size = (n_new - n)
    if samp_size > 0:
        try:
            x = x + np.random.binomial(n=samp_size, p=weights[edge])
        except OverflowError:
            # if n is too large, use normal approximation
            x = x + approx_binomial(n=samp_size, p=weights[edge])
    n = n_new
    return float(x) / n, n

def approx_binomial(n, p, size=None):
    gaussian = np.random.normal(n*p, math.sqrt(n*p*(1-p)), size=size)
    # Add the continuity correction to sample at the midpoint of each integral bin.
    gaussian += 0.5
    if size is not None:
        binomial = gaussian.astype(np.int64)
    else:
        # scalar
        binomial = int(gaussian)
    return binomial

def N(epsilon, delta):
    return math.ceil(math.log(2 / delta) / (2 * epsilon ** 2))





