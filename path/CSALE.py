import numpy as np
import math
from Dijkstar.dijkstar.algorithm import find_path, NoPathError
import Graphs


def csale(epsilon, delta, graph, start, dest, d=None):
    global accept
    accept = set()
    S = Graphs.get_edges(graph)
    v = len(graph.get_data())
    graph_copy = Graphs.copy_graph(graph, v)
    global mu_t
    mu_t = dict.fromkeys(S, 0)
    N_t = dict.fromkeys(S, 0)
    epsilon_t = epsilon
    t = 1
    oracle_counter = 0
    if not d:
        d_t = Graphs.find_longest_path(graph_copy, v, start, dest)
    else:
        d_t = d
    print('Longest path in round ' + str(t) + ": " + str(d_t))
    T = math.ceil(math.log2(2 * d_t))
    while epsilon_t > (epsilon / (d_t - len(accept))):

        theta = epsilon_t * (d_t - len(accept))
        print('Threshold in round ' + str(t) + ": " + str(theta))

        # sample every edge in S and update mu
        for e in S:
            mu_t[e], N_t[e] = sample(e, epsilon_t / 2, delta / (T * len(S)), mu_t[e], N_t[e])
        Mhat = find_path(graph_copy, start, dest, cost_func=cost_func)
        oracle_counter = oracle_counter + 1
        print('candidtae edges: ' + str(Mhat.edges))
        Mhat_score = sum([mu_t[a] for a in Mhat.edges])
        for e in S.intersection(set(Mhat.edges)):
            accept_e = False
            graph_copy.remove_edge(e[0], e[1])
            try:
                Mtilde = find_path(graph_copy, start, dest, cost_func=cost_func)
                Mtilde_score = sum([mu_t[a] for a in Mtilde.edges])
                oracle_counter = oracle_counter + 1

            except NoPathError:
                accept_e = True
            graph_copy.add_edge(e[0], e[1], e)
            if not accept_e:
                gap_e = Mtilde_score - Mhat_score  # this is a minimization problem
                print('The gap of edge ' + str(e) + ': ' + str(gap_e))
                accept_e = gap_e > theta
            if accept_e:
                print('accept edge ' + str(e))
                accept.update({e})
                S.remove(e)
                to_remove = set()
                # e = (u,v), eliminate edges (u,*) and (*,v)
                for a in S:
                    if a[0] == e[0] or a[1] == e[1]:
                        to_remove.add(a)
                        graph_copy.remove_edge(a[0], a[1])
                print('eliminate edges: ' + str(to_remove))
                S = S - to_remove
                if not d:
                    d_t = Graphs.find_longest_path(graph_copy, v, start, dest)
                print('Longest path in round ' + str(t) + ": " + str(d_t))
                theta = epsilon_t * (d_t - len(accept))
                print('Threshold in round ' + str(t) + ": " + str(theta))
        if Graphs.isPath(accept, start, dest):
            print('Terminating before t=T')
            return accept, sum(N_t.values()), len(accept), oracle_counter
        t = t + 1
        epsilon_t = epsilon_t / 2

    epsilon_last = epsilon / (d_t - len(accept))
    # sample every a in S and update mu
    for e in S:
        mu_t[e], N_t[e] = sample(e, epsilon_last / 2, delta / (T * len(S)), mu_t[e], N_t[e])

    Mhat = find_path(graph_copy, start, dest, cost_func=cost_func)
    oracle_counter = oracle_counter + 1
    return Mhat.edges, sum(N_t.values()), len(accept), oracle_counter

def naive(epsilon, delta, graph, start, dest, d=None):
    S = Graphs.get_edges(graph)
    global mu_t
    mu_t = dict.fromkeys(S, 0)
    N_t = dict.fromkeys(S, 0)
    if not d:
        d = Graphs.find_longest_path(graph, len(graph.get_data()), start, dest)
    for e in S:
        mu_t[e], N_t[e] = sample(e, epsilon / (2 * d), delta / len(S), mu_t[e], N_t[e])


    M = find_path(graph, start, dest, cost_func=cost_func)
    return M.edges, sum(N_t.values())

def sample(edge, epsilon, delta, x, n):
    n_new = N(epsilon / 2, delta)
    samp_size = (n_new - n)
    if samp_size > 0:
        try:
            x = x + np.random.binomial(n=samp_size, p=edge[2])
        except OverflowError:
            # if n is too large, use normal approximation
            x = x + approx_binomial(n=samp_size, p=edge[2])
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

def cost_func(u, v, e, prev_e):
    if e in accept:
        return 0
    return mu_t[e]






