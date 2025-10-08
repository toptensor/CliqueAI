from CliqueAI.protocol import MaximumCliqueOfLambdaGraph

from .gnn_models import SC_MODEL


def scattering_clique_algorithm(number_of_nodes: int, adjacency_list: list[list[int]]) -> list[int]:
    def extend_to_maximal_clique(
        adjacency_list: list[list[int]], clique: list[int]
    ) -> list[int]:
        clique_set = set(clique)
        n = len(adjacency_list)
        changed = True
        while changed:
            changed = False
            for v in range(n):
                if v in clique_set:
                    continue
                neighbors = set(adjacency_list[v])
                if clique_set.issubset(neighbors):
                    clique_set.add(v)
                    changed = True
                    break
        return list(clique_set)

    num_nodes = number_of_nodes
    adjacency_list = adjacency_list
    maximum_clique: list[int] = []
    for clique in SC_MODEL.predict_iter(num_nodes, adjacency_list):
        clique = extend_to_maximal_clique(adjacency_list, list(map(int, clique)))
        if len(clique) > len(maximum_clique):
            maximum_clique = clique
    return maximum_clique
