import networkx as nx
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph


def networkx_algorithm(synapse: MaximumCliqueOfLambdaGraph):
    num_nodes = synapse.number_of_nodes
    adjacency_list = synapse.adjacency_list
    dict_of_lists = {i: adjacency_list[i] for i in range(num_nodes)}
    graph = nx.from_dict_of_lists(dict_of_lists)
    return list(nx.approximation.max_clique(graph))
