# CSALE
This project provides the implementations for the paper "A Fast Algorithm for PAC Combinatorial Pure Exploration", Noa Ben-David and Sivan Sabato, AAAI-22.

CSALE is a Combinatorial Pure Exploration (CPE) algorithm, which deals with finding a combinatorial set of arms with a high reward, when the rewards of individual arms are unknown in advance and must be estimated using arm pulls. CSALE is a CPE algorithm in the PAC setting, which is computationally light weight, and so can easily be applied to problems with tens of thousands of arms. This is achieved since CSALE requires a very small number of combinatorial oracle calls. The algorithm is based on successive acceptance of arms, along with elimination which is based on the combinatorial structure of the problem.

Environment
===========
The code is written in python3.
Required packages: numpy, pandas, networkx.



Graph file format
=================
The format of a file describing a graph is as follows:
For a weighted graph, each row has the format "x y z", which represents the edge (x,y) with weight z. x,y, z is a non-negative number. If the maximal weight in the file is larger than 1, then the weights are renormalized such that the maximal weight is 1. 

For an unweighted graph, each row has the format "x y", which represents the edge (x,y). x,y are integers. In this case, the weights will be sampled randomly out of {0.1,0.5,0.9} when reading the graph file.


====== Running one of the algorithms for shortest path or matching =============
The following steps run one of the three algorithms (naive, CLUCB-PAC, CSALE) on a given graph and simulate arm pulls to find PAC solution for one of the problems (shortest path / matching).
1. Load a graph file
2. Run the selected algorithm.

Below, we specify how to run steps 1 and 2 for each problem.



====== Shortest path ==========

To solve shortest path problems, run the code in the folder 'path'.
For step 1 (load a graph file), run one of the following commands, where the nodes in the graph file must be named using the integers 'first'...'n+first-1':
g = Graphs.read_weighted_graph_from_txt(<file_name>, n, first=0, undirected=False)
g = Graphs.read_graph_from_txt(<file_name>, n, first=0, undirected=False)


For step 2 (running one of the algorithms), use one of the following commands:
csale_results = CSALE.csale(epsilon, delta, graph, start, dest)
naive_results = CSALE.naive(epsilon, delta, graph, start, dest)
clucbpac_results = CLUCB.clucb(epsilon, delta, graph, start, dest)


Example for steps 1 and 2:
g = Graphs.read_weighted_graph_from_txt('USAir97_graph.txt', n=332, first=1, undirected=False)
csale_log = CSALE.csale(epsilon=0.01, delta=0.05, graph=g, start=269, dest=107)


====== Matching ==========

To solve shortest path problems, run the code in the folder 'matching'.


For step 1 (load a graph file), run the following command:
edges_info = Graphs.generate_from_file(<file_path>, weighted=True)

For step 2 (running one of the algorithms), use one of the following commands:
csale_results = CSALE.csale(epsilon, delta, edges_info)
naive_results = CSALE.naive(epsilon, delta, edges_info)
clucbpac_results = CLUCB.clucb(epsilon, delta, edges_info)


Example for step 2:
edges, weights = Graphs.generate_from_file('USAir97_graph.txt', weighted=True)
CSALE.csale(epsilon=0.01, delta=0.05, edges_info=[edges, weights])


====== Running the experiments =========
Running the experiments generates csv files with the results.
Follow the following steps:

1. Download all the graph files from the links given in the paper:

For the shortest path experiments, read each graph from its file and save it as follows:
g = read_weighted_graph_from_txt('Davis_southern_club_women-cooccurance.txt', 18, first=1, undirected=True)
g.dump('southern_women')
g = read_weighted_graph_from_txt('Freemans_EIES-3_n32.txt', 32, first=1)
g.dump('Freemans_EIES')
g = read_weighted_graph_from_txt('Cross_Parker-Consulting_info.txt', 46, first=1)
g.dump('Consulting_info')
g = read_weighted_graph_from_txt('Cross_Parker-Consulting_value.txt', 46, first=1)
g.dump('Consulting_value')
g = read_weighted_graph_from_txt('Cross_Parker-Manufacturing_info.txt', 77, first=1)
g.dump('Manufacturing_info')
g = read_weighted_graph_from_txt('Cross_Parker-Manufacturing_aware.txt', 77, first=1)
g.dump('Manufacturing_aware')
g = read_weighted_graph_from_txt('celegans_n306.txt', 306, first=1)
g.dump('celegans')
g = read_weighted_graph_from_txt('USAir97.txt', 332, first=1)
g.dump('USAir')
g = read_weighted_graph_from_txt('p2p-Gnutella05.txt', 8846, first=0, undirected=True)
g.dump('p2p_Gnutella05')
g = read_weighted_graph_from_txt('p2p-Gnutella06.txt', 8717, first=0, undirected=True)
g.dump('p2p_Gnutella06')
g = read_weighted_graph_from_txt('p2p-Gnutella08.txt', 6301, first=0, undirected=True)
g.dump('p2p_Gnutella08')
g = read_weighted_graph_from_txt('p2p-Gnutella09.txt', 8114, first=0, undirected=True)
g.dump('p2p_Gnutella09')

2. Run the following commands from the folder 'path/experiments':

clucb_synthetic_experiment()
clucb_usa_experiment()
large_networks_experiment('southern_women', 4)
large_networks_experiment('Freemans_EIES', 5)
large_networks_experiment('Consulting_info', 4)
large_networks_experiment('Consulting_value', 4)
large_networks_experiment('Manufacturing_info', 4)
large_networks_experiment('Manufacturing_aware', 4)
large_networks_experiment('celegans', 10)
large_networks_experiment('USAir')
large_networks_experiment('p2p_Gnutella05', 18)
large_networks_experiment('p2p_Gnutella06', 20)
large_networks_experiment('p2p_Gnutella08', 18)
large_networks_experiment('p2p_Gnutella09', 20)

3. Run the following commands from the folder 'matching/experiments':

clucb_synthetic_experiment()
clucb_usa_experiment()
large_networks_experiment('Davis_southern_club_women-cooccurance')
large_networks_experiment('Freemans_EIES-3_n32')
large_networks_experiment('Cross_Parker-Consulting_info')
large_networks_experiment('Cross_Parker-Consulting_value')
large_networks_experiment('Cross_Parker-Manufacturing_info')
large_networks_experiment('Cross_Parker-Manufacturing_aware')
large_networks_experiment('celegans_n306')
large_networks_experiment('USAir97')
