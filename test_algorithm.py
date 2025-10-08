import json

from CliqueAI.clique_algorithms import (networkx_algorithm,
                                        scattering_clique_algorithm)
from CliqueAI.protocol import MaximumCliqueOfLambdaGraph

data_paths = [
    "test_data/general_0.1.json",
    "test_data/general_0.2.json",
    "test_data/general_0.4.json",
]


def get_test_data(data_path: str) -> MaximumCliqueOfLambdaGraph:
    with open(data_path, "r") as f:
        data = json.load(f)
    synapse = MaximumCliqueOfLambdaGraph.model_validate(data)
    return synapse


def check_clique(adjacency_list: list[list[int]], clique: list[int]) -> bool:
    clique_set = set(clique)
    for i in range(len(clique)):
        node = clique[i]
        neighbors = set(adjacency_list[node])
        if not clique_set.issubset(neighbors.union({node})):
            return False
    for v in range(len(adjacency_list)):
        if v in clique_set:
            continue
        if all(v in adjacency_list[node] for node in clique):
            return False
    return True


def run(algorithm, synapse: MaximumCliqueOfLambdaGraph):
    maximum_clique = algorithm(synapse)
    clique_check = check_clique(synapse.adjacency_list, maximum_clique)
    if not clique_check:
        print("Invalid clique found by algorithm!")
    else:
        print(f"Clique size: {len(maximum_clique)}")


def main():
    for data_path in data_paths:
        synapse = get_test_data(data_path)
        print(f"Testing data from {data_path} with {synapse.number_of_nodes} nodes")
        # put your algorithm here
        run(scattering_clique_algorithm, synapse)


if __name__ == "__main__":
    main()
