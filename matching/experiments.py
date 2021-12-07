import CSALE
import pandas as pd
import random
import Graphs
import time
import CLUCB
import networkx as nx
from networkx.algorithms.matching import max_weight_matching, maximal_matching

def large_networks_experiment(network, num_iterations=10):
    results = pd.DataFrame(columns=['epsilon','iteration',
                                    'naive solution', 'naive SC', 'naive epsilon', 'naive time',
                                    'active solution', 'active SC', 'active epsilon', 'active time',
                                    'active oracle calls', 'SC ratio',
                                    'accepted ratio'])
    delta = 0.05
    errors = [2, 1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.01, 0.005, 0.001]
    edges, weights = Graphs.generate_from_file(network + '.txt', True)
    row = 0
    opt_cost = optimalCost(edges, weights)
    for epsilon in errors:
        for i in range(1,num_iterations + 1):
            active_time = time.perf_counter()
            M1, N1, num_accepted, O1 = CSALE.csale(epsilon, delta, [edges, weights])
            active_time = time.perf_counter() - active_time
            active_eps = opt_cost - getCost(M1, weights)
            naive_time = time.perf_counter()
            M2, N2 = CSALE.naive(epsilon, delta, [edges, weights])
            naive_time = time.perf_counter() - naive_time
            naive_eps = opt_cost - getCost(M2, weights)
            accepted_ratio = float(num_accepted) * 100 / len(M1)

            results.loc[row] = [epsilon] + \
                                   [i, M2, N2, naive_eps, naive_time, M1, N1, active_eps, active_time, O1, float(N1) * 100 / N2, accepted_ratio]
            print(results.loc[row])
            row = row + 1

    results.to_csv('results_m_' + network + '.csv')

def clucb_synthetic_experiment():

    results = pd.DataFrame(columns=['epsilon',
                                    'clucb solution', 'clucb SC', 'clucb epsilon', 'clucb time',
                                    'clucb oracle calls',
                                    'active solution', 'active SC', 'active epsilon', 'active time',
                                    'active oracle calls', 'SC ratio',
                                    'time ratio', 'accepted ratio', 'graph epsilon',
                                    'naive solution', 'naive SC', 'naive epsilon', 'naive time'])
    delta = 0.05
    errors = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2]
    i = 1
    row = 0
    while i <= 100:
        print('iteration ' + str(i))
        edges, weights = Graphs.generate_toy_graph()
        while not is_unique(weights):
            edges, weights = Graphs.generate_toy_graph()
        edges_info = [edges, weights]
        opt_cost = optimalCost(edges, weights)
        graph_epsilon = true_gap(weights)


        for epsilon in errors:
            print('running CLUCB..')
            clucb_time = time.perf_counter()
            M2, N2, O3 = CLUCB.clucb(delta, epsilon, edges_info)
            clucb_time = time.perf_counter() - clucb_time
            print('CLUCB time: ' + str(clucb_time))
            clucb_eps = getCost(M2, weights) - opt_cost
            print('running CSALE..')
            active_time = time.perf_counter()
            M1, N1, num_accepted, O1 = CSALE.csale(epsilon, delta, edges_info)
            active_time = time.perf_counter() - active_time
            print('CSALE time: ' + str(active_time))
            active_eps = getCost(M1, weights) - opt_cost
            accepted_ratio = float(num_accepted) * 100 / len(M1)

            print('running Naive..')
            naive_time = time.perf_counter()
            M3, N3 = CSALE.naive(epsilon, delta, edges_info)
            naive_time = time.perf_counter() - naive_time
            print('Naive time: ' + str(naive_time))
            naive_eps = getCost(M3, weights) - opt_cost

            results.loc[row] = [epsilon] + [M2, N2, clucb_eps, clucb_time, O3,
                                            M1, N1, active_eps, active_time, O1,
                                            float(N1) * 100 / N2, active_time * 100 / clucb_time, accepted_ratio,
                                            graph_epsilon, M3, N3, naive_eps, naive_time]
            print(results.loc[row])
            row = row + 1
        i = i + 1

    results.to_csv('results_clucb_synthetic.csv')

def clucb_usa_experiment(num_iteration=10):

    results = pd.DataFrame(columns=['epsilon',
                                    'clucb solution', 'clucb SC', 'clucb epsilon', 'clucb time',
                                    'clucb oracle calls',
                                    'active solution', 'active SC', 'active epsilon', 'active time',
                                    'active oracle calls', 'SC ratio',
                                    'time ratio', 'accepted ratio',
                                    'naive solution', 'naive SC', 'naive epsilon', 'naive time'])
    edges, weights = Graphs.generate_from_file('USAir97.txt', True)
    n = 332
    found = False
    delta = 0.05
    errors = [0.03125, 0.0625, 0.125, 0.25, 0.5, 1, 2]
    i = 1
    row = 0
    while i <= num_iteration:
        print('iteration ' + str(i))
        while not found:
            nodes = random.sample(range(1, n+1), 4)
            new_edges = []
            for e in edges:
                if e[0] in nodes and e[1] in nodes:
                    new_edges.append(e)
            found = len(new_edges) > 4
        edges_info = [new_edges, weights]
        opt_cost = optimalCost(new_edges, weights)
        


        for epsilon in errors:
            print('running CLUCB..')
            clucb_time = time.perf_counter()
            M2, N2, O3 = CLUCB.clucb(delta, epsilon, edges_info)
            clucb_time = time.perf_counter() - clucb_time
            print('CLUCB time: ' + str(clucb_time))
            clucb_eps = getCost(M2, weights) - opt_cost
            epsilon1 = epsilon / 2
            print('running CSALE..')
            active_time = time.perf_counter()
            M1, N1, num_accepted, O1 = CSALE.csale(epsilon, delta, edges_info)
            active_time = time.perf_counter() - active_time
            print('CSALE time: ' + str(active_time))
            active_eps = getCost(M1, weights) - opt_cost
            accepted_ratio = float(num_accepted) * 100 / len(M1)

            print('running Naive..')
            naive_time = time.perf_counter()
            M3, N3 = CSALE.naive(epsilon, delta, edges_info)
            naive_time = time.perf_counter() - naive_time
            print('Naive time: ' + str(naive_time))
            naive_eps = getCost(M3, weights) - opt_cost

            results.loc[row] = [epsilon] + [M2, N2, clucb_eps, clucb_time, O3,
                                            M1, N1, active_eps, active_time, O1,
                                            float(N1) * 100 / N2, active_time * 100 / clucb_time, accepted_ratio,
                                            M3, N3, naive_eps, naive_time]
            print(results.loc[row])
            row = row + 1
        i = i + 1

    results.to_csv('results_clucb_usa.csv')


def dfs(graph, v, visited):

    stack = []
    stack.append(v)

    while stack:
        v = stack[-1]
        stack.pop()
        visited[v] = True
        for u in (graph.get_data()[v]).keys():
            if not visited[u]:
                stack.append(u)


def optimalCost(edges, W):
    graph = nx.Graph()
    for e in edges:
        graph.add_edge(e[0], e[1], weight=W[e])
    O = max_weight_matching(graph)
    return sum([W[Graphs.edge(e)] for e in O])

def getCost(sol, W):
    return sum([W[Graphs.edge(e)] for e in sol])

def is_unique(W):
    scores = []
    for p in setpartition(range(6), n=2):
        scores.append(sum([W[e] for e in p]))
    scores = ['%.3f' % elem for elem in scores]
    opt = max(scores)
    if scores.count(opt) > 1:
        return False
    return True



def setpartition(iterable, n=2):
    from itertools import combinations
    iterable = list(iterable)
    partitions = combinations(combinations(iterable, r=n), r=int(len(iterable) / n))
    for partition in partitions:
        seen = set()
        for group in partition:
            if seen.intersection(group):
                break
            seen.update(group)
        else:
            yield partition

def true_gap(W):
    scores = []
    for p in setpartition(range(6), n=2):
        scores.append(sum([W[e] for e in p]))
    scores = [round(elem, 3) % elem for elem in scores]
    opt_score = max(scores)
    scores.remove(opt_score)
    second_score = max(scores)
    return opt_score - second_score







