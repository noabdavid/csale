import random
import matplotlib.pyplot as plt

def generate_toy_graph():
    weights = [0.1, 0.5, 0.9]
    edges = []
    W = dict()
    for i in range(6):
        for j in range(i + 1, 6):
            e = (i, j)
            edges.append(e)
            W[e] = random.choice(weights)

    return edges, W

def generate_from_file(path, weighted=True):
    edges = []
    W = dict()
    f = open(path, 'r')
    lines = f.readlines()
    if weighted:
        max_weight = -1
    for l in lines:
        if l.find('#') == -1 and l.find('%') == -1:
            e = l.split()
            left = min([int(e[0]), int(e[1])])
            right = max([int(e[0]), int(e[1])])
            if not left == right:
                a = (left, right)
                edges.append(a)
                if weighted:
                    W[a] = float(e[2])
                    max_weight = max(max_weight, W[a])
                else:
                    W[a] = random.choice([0.1, 0.5, 0.9])

    if weighted:
        for e in W.keys():
            W[e] = W[e] / max_weight


    return edges, W

def edge(e):
    return (min(e[0], e[1]), max(e[0], e[1]))
    
def share_node(e1, e2):
    return e1[0] == e2[0] or e1[0] == e2[1] or e1[1] == e2[0] or e1[1] == e2[1]
    
def is_maximal_matching(edge_set, graph):
    l = []
    for e in edge_set:
        l.append(e[0])
        l.append(e[1])

    for e in graph.edges:
        if e[0] not in l and e[1] not in l:
            return False

    return True

