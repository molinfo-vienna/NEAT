import torch
from torch_geometric.data import Data


class SourceTargetSplitter:
    """Class to create source-target splits for molecular graphs.

    Args:
        splitting_mode (str): The mode of splitting to use. Default is "neighborhood".
        target_set_max_size (int): Maximum size of the target set. Default is -1 (no limit).
    """

    def __init__(
        self,
        splitting_mode: str = "neighborhood",
        target_set_max_size: int = -1,
    ):
        self.splitting_mode = splitting_mode
        self.target_set_max_size = target_set_max_size

    def create_source_target_split(
        self,
        data: Data,
        device: torch.device = None,
    ):
        """Create a source-target split for a batch of data.

        Args:
            data (Data): The data to split.
            device (torch.device): The device to use for computations.

        Returns:
            A tuple containing the source-target split.
        """
        if self.splitting_mode == "neighborhood":
            return self.neighborhood_guided_source_target_split(data, device=device)
        else:
            raise ValueError(f"Unknown splitting mode: {self.splitting_mode}")

    def neighborhood_guided_source_target_split(
        self,
        data: Data,
        beta: float = 1.5,
        gamma: float = 0.45,
        device: torch.device = None,
    ):
        """
        Split atoms into source and target sets, using a neighborhood-guided approach.

        Args:
            data (Data): The data to split.
            device (torch.device): The device to use for computations.

        Returns:
            A tuple containing the source-target split.
        """
        atom_counts = torch.bincount(data.batch)

        # First we need to sample a random atom from each graph in the batch and mark it as a source set atom
        marked_nodes_idx = (
            torch.cat(
                [
                    torch.randint(0, atom_count, (1,), device=device)
                    for atom_count in atom_counts
                ]
            )
            + data.ptr[:-1]
        )
        marked_nodes_eccentricity = data.eccentricity[marked_nodes_idx]

        # The number of neighbourhood hops is determined by the graph diameter
        num_iterations_per_graph = (
            (marked_nodes_eccentricity * beta)
            * torch.rand(marked_nodes_eccentricity.shape[0], device=device)
            * 0.999
        ).long() + 1
        max_num_iterations = num_iterations_per_graph.max()

        for iteration in range(max_num_iterations):
            # For graphs that have not yet reached their iteration limit
            active_graphs_mask = iteration < num_iterations_per_graph
            if active_graphs_mask.sum() == 0:
                break
            active_nodes_mask = active_graphs_mask[data.batch]
            marked_modes_mask = torch.zeros_like(
                data.batch, device=device, dtype=torch.bool
            )
            marked_modes_mask[marked_nodes_idx] = True
            active_marked_nodes_mask = active_nodes_mask & marked_modes_mask
            active_marked_node_idx = torch.nonzero(
                active_marked_nodes_mask, as_tuple=False
            ).squeeze()

            # From the current random nodes, find all connected neighbours in the full graphs
            connected_edges_mask = torch.isin(
                data.edge_index[0], active_marked_node_idx
            )
            one_hop_neighbours_idx = data.edge_index[1][connected_edges_mask]
            one_hop_neighbours_idx = torch.unique(one_hop_neighbours_idx)

            # Randomly pick a subset of these neighbours
            permutation = torch.randperm(one_hop_neighbours_idx.size(0))
            one_hop_neighbours_idx = one_hop_neighbours_idx[permutation]
            one_hop_neighbours_idx = one_hop_neighbours_idx[
                : int(len(one_hop_neighbours_idx) * (1 - gamma))
            ]

            # Add these neighbours to the random nodes
            marked_nodes_idx = torch.cat((marked_nodes_idx, one_hop_neighbours_idx))
            marked_nodes_idx = torch.unique(marked_nodes_idx)

        # By excluding the target set atoms, we can already create the source set
        source_ptr = marked_nodes_idx
        source_set_mask = torch.zeros_like(data.batch, device=device, dtype=torch.bool)
        source_set_mask[source_ptr] = 1

        # Now we can find all one-hop neighbours of the source set in the full graph
        # The interaction of these neighbours with the above target set defines the actual target set
        target_set_mask = ~source_set_mask
        target_set_idx = torch.nonzero(target_set_mask, as_tuple=False).squeeze()
        source_set_one_hop_mask = torch.isin(data.edge_index[0], source_ptr)
        source_set_one_hop_idx = torch.unique(
            data.edge_index[1][source_set_one_hop_mask]
        )
        target_set_neighbours_mask = torch.isin(source_set_one_hop_idx, target_set_idx)
        target_ptr = torch.unique(source_set_one_hop_idx[target_set_neighbours_mask])

        return (
            source_ptr,
            target_ptr,
        )
