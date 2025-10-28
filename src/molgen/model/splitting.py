import torch
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_add_pool


import torch
from rdkit import Chem
from rdkit.Chem import Draw


class SourceTargetSplitter:
    def __init__(self, splitting_mode="random", target_set_max_size: int = -1):
        self.target_set_max_size = target_set_max_size
        self.splitting_mode = splitting_mode

    def create_source_target_split(self, data: Data, device=None):
        if self.splitting_mode == "random":
            return self.random_source_target_split(data, device=device)
        elif self.splitting_mode == "mst":
            return self.mst_source_target_split(data, device=device)
        elif self.splitting_mode == "cyclic":
            return self.cyclic_source_target_split(data, device=device)
        else:
            raise ValueError(f"Unknown splitting mode: {self.splitting_mode}")

    def random_source_target_split(self, data: Data, device=None):
        atom_counts = torch.bincount(data.batch)

        # Randomly select a subset of atoms per molecule
        uniform_distribution = torch.rand(atom_counts.shape, device=device) * 0.999
        deletion_limit = atom_counts - 1
        if self.target_set_max_size > 0:
            deletion_limit = torch.min(
                torch.ones_like(deletion_limit) * self.target_set_max_size,
                deletion_limit,
            )

        # This samples between 0 and up to N-1 atoms to delete per molecule
        atoms_to_delete = ((deletion_limit.float() + 1) * uniform_distribution).int()
        atoms_to_keep = atom_counts - atoms_to_delete
        random_indices = torch.cat(
            [
                (torch.randperm(i, device=device) + k)
                for i, k in zip(atom_counts, data.ptr[0:-1])
            ]
        )
        subset_idx = torch.cat(
            [random_indices[j : j + k] for j, k in zip(data.ptr[0:-1], atoms_to_keep)]
        )
        subset_mask = torch.zeros_like(data.batch, device=device, dtype=torch.bool)
        subset_mask[subset_idx] = 1

        x_source = data.x[subset_mask]
        pos_source = data.pos[subset_mask]
        batch_source = data.batch[subset_mask]
        x_target = data.x[~subset_mask]
        pos_target = data.pos[~subset_mask]
        batch_target = data.batch[~subset_mask]
        stop_tokens = atoms_to_delete == 0

        return (
            x_source,
            pos_source,
            batch_source,
            x_target,
            pos_target,
            batch_target,
            stop_tokens,
        )

    def cyclic_source_target_split(self, data: Data, device=None):
        atom_counts = torch.bincount(data.batch)

        # First we need to sample a random atom from each graph in the batch
        random_nodes = (
            torch.cat(
                [
                    torch.randint(0, atom_count, (1,), device=device)
                    for atom_count in atom_counts
                ]
            )
            + data.ptr[:-1]
        )
        num_iterations_per_graph = (
            data.diameter * torch.rand(data.diameter.size(0), device=device)
        ).long()
        max_num_iterations = num_iterations_per_graph.max()

        for iteration in range(max_num_iterations):
            # For graphs that have not yet reached their iteration limit
            mask = iteration < num_iterations_per_graph
            if mask.sum() == 0:
                break
            masked_atoms = mask[data.batch]
            random_atom_mask = torch.zeros_like(
                data.batch, device=device, dtype=torch.bool
            )
            random_atom_mask[random_nodes] = True
            random_nodes_mask = masked_atoms & random_atom_mask
            random_nodes_masked = torch.nonzero(
                random_nodes_mask, as_tuple=False
            ).squeeze()

            # From the current random nodes, find all connected neighbours in the full graphs
            edge_mask = torch.isin(data.edge_index[0], random_nodes_masked)
            neighbours = data.edge_index[1][edge_mask]

            # Randomly pick a subset of these neighbours
            permutation = torch.randperm(neighbours.size(0))
            neighbours = neighbours[permutation]
            neighbours = neighbours[: int(len(neighbours) * 0.75)]
            neighbours = torch.unique(neighbours)

            # Add these neighbours to the random nodes
            random_nodes = torch.cat((random_nodes, neighbours))
            random_nodes = torch.unique(random_nodes)

        # By excluding the target set atoms, we can already create the source set
        source_set_idx = random_nodes
        source_set_mask = torch.zeros_like(data.batch, device=device, dtype=torch.bool)
        source_set_mask[source_set_idx] = 1
        x_source = data.x[source_set_mask]
        pos_source = data.pos[source_set_mask]
        batch_source = data.batch[source_set_mask]
        target_set_mask = ~source_set_mask
        target_set_idx = torch.nonzero(target_set_mask, as_tuple=False).squeeze()

        # Now we can find all one-hop neighbours of the source set in the full graph
        # The interaction of these neighbours with the above target set defines the actual target set
        source_set_one_hop_mask = torch.isin(data.edge_index[0], source_set_idx)
        source_set_one_hop_idx = torch.unique(
            data.edge_index[1][source_set_one_hop_mask]
        )
        target_set_neighbours_mask = torch.isin(source_set_one_hop_idx, target_set_idx)
        target_set_neighbours_idx = torch.unique(
            source_set_one_hop_idx[target_set_neighbours_mask]
        )
        target_set_final_mask = torch.zeros_like(
            data.batch, device=device, dtype=torch.bool
        )
        target_set_final_mask[target_set_neighbours_idx] = 1
        x_target = data.x[target_set_final_mask]
        pos_target = data.pos[target_set_final_mask]
        batch_target = data.batch[target_set_final_mask]

        # Let's do some final checks
        assert source_set_mask.sum() + target_set_mask.sum() == data.x.size(
            0
        ), "Source and target set sizes do not add up to total atom count!"
        assert (
            target_set_final_mask + target_set_mask
        ).sum() == target_set_mask.sum(), (
            "Target set final mask needs to be a subset of target set mask!"
        )

        # Currently, I do not create stop tokens for MST splitting
        # This should be introduced with some small probability
        source_set_atom_count = global_add_pool(source_set_mask.int(), data.batch)
        stop_tokens = source_set_atom_count == atom_counts

        # data_point = data[0]
        # target_idx = target_set_neighbours_idx[
        #     target_set_neighbours_idx < data_point.num_nodes
        # ]
        # source_idx = source_set_idx[source_set_idx < data_point.num_nodes]
        # self.create_rdkit_molecule(data_point, source_idx, target_idx)

        return (
            x_source,
            pos_source,
            batch_source,
            x_target,
            pos_target,
            batch_target,
            stop_tokens,
        )

    def mst_source_target_split(self, data: Data, device=None):
        atom_counts = torch.bincount(data.batch)

        # First we need to sample a random edge from each Minimum Spanning Tree (MST)
        random_edges = self.sample_weighted_edge_from_mst(data)

        # Now we can start from the target nodes and mark all reachable nodes in the MST
        # print("Building adjacency list for MST...")
        source_nodes = random_edges[1]
        target_set_idx = source_nodes.clone()
        while True:
            mask = torch.isin(data.edge_index_mst[0], source_nodes)
            if mask.sum() == 0:
                break
            target_nodes = data.edge_index_mst[1][mask]
            target_set_idx = torch.cat((target_set_idx, target_nodes))
            source_nodes = target_nodes
        target_set_idx = torch.unique(target_set_idx)

        # By excluding the target set atoms, we can already create the source set
        target_set_mask = torch.zeros_like(data.batch, device=device, dtype=torch.bool)
        target_set_mask[target_set_idx] = 1
        source_set_mask = ~target_set_mask
        x_source = data.x[source_set_mask]
        pos_source = data.pos[source_set_mask]
        batch_source = data.batch[source_set_mask]
        source_set_idx = torch.nonzero(source_set_mask, as_tuple=False).squeeze()

        # Now we can find all one-hop neighbours of the source set in the full graph
        # The interaction of these neighbours with the above target set defines the actual target set
        source_set_one_hop_mask = torch.isin(data.edge_index[0], source_set_idx)
        source_set_one_hop_idx = torch.unique(
            data.edge_index[1][source_set_one_hop_mask]
        )
        target_set_neighbours_mask = torch.isin(source_set_one_hop_idx, target_set_idx)
        target_set_neighbours_idx = torch.unique(
            source_set_one_hop_idx[target_set_neighbours_mask]
        )
        target_set_final_mask = torch.zeros_like(
            data.batch, device=device, dtype=torch.bool
        )
        target_set_final_mask[target_set_neighbours_idx] = 1
        x_target = data.x[target_set_final_mask]
        pos_target = data.pos[target_set_final_mask]
        batch_target = data.batch[target_set_final_mask]

        # Let's do some final checks
        assert source_set_mask.sum() + target_set_mask.sum() == data.x.size(
            0
        ), "Source and target set sizes do not add up to total atom count!"
        assert (
            target_set_final_mask + target_set_mask
        ).sum() == target_set_mask.sum(), (
            "Target set final mask needs to be a subset of target set mask!"
        )

        # Currently, I do not create stop tokens for MST splitting
        # This should be introduced with some small probability
        stop_tokens = atom_counts == 0

        # data_point = data[0]
        # target_idx = target_set_neighbours_idx[
        #     target_set_neighbours_idx < data_point.num_nodes
        # ]
        # source_idx = source_set_idx[source_set_idx < data_point.num_nodes]
        # self.create_rdkit_molecule(data_point, source_idx, target_idx)

        return (
            x_source,
            pos_source,
            batch_source,
            x_target,
            pos_target,
            batch_target,
            stop_tokens,
        )

    def sample_weighted_edge_from_mst(self, batch):
        """
        Samples a single weighted random edge from the MST edge set (edge_index_mst) for each graph in a batch.
        The sampling weight is halved for each step into a lower hierarchy.

        Args:
            batch (torch_geometric.data.Batch): A batch of PyG Data objects.

        Returns:
            torch.Tensor: A tensor containing one weighted random edge (2 nodes) for each graph in the batch.
        """
        # Get the batch-wise edge_index_mst, edge_attributes_mst, and batch information
        device = batch.x.device
        edge_index_mst = batch.edge_index_mst  # Shape: [2, num_edges]
        edge_attributes_mst = batch.edge_attr_mst  # Shape: [num_edges]
        batch_indices = batch.batch[edge_index_mst[0]]  # Batch indices for each edge

        # Get the number of graphs in the batch
        num_graphs = batch.num_graphs

        # Find the number of edges per graph in the MST
        edge_counts = torch.bincount(batch_indices)

        # Compute weights for edges based on their hierarchical level
        weights = 2 ** (-edge_attributes_mst.float())  # Shape: [num_edges]

        # Normalize weights within each graph
        normalized_weights = torch.zeros_like(weights, device=device)
        for i in range(num_graphs):
            graph_mask = batch_indices == i  # Mask for edges belonging to graph i
            graph_weights = weights[graph_mask]
            normalized_weights[graph_mask] = graph_weights / graph_weights.sum()

        # Sample one edge per graph using the normalized weights
        random_indices = torch.cat(
            [
                torch.multinomial(
                    normalized_weights[batch_indices == i], 1, replacement=False
                )
                for i in range(num_graphs)
            ]
        )
        offsets = torch.cat(
            (torch.tensor([0], device=device), torch.cumsum(edge_counts, 0))
        )[:-1]
        random_indices += offsets

        # Extract the random edges
        random_edges = edge_index_mst[:, random_indices]

        return random_edges

    # def create_rdkit_molecule(
    #     self, data, source_index, target_index, output_file="molecule.png"
    # ):
    #     """
    #     Create an RDKit molecule object from PyTorch Geometric data and visualize it.

    #     Parameters:
    #         data: PyTorch Geometric data object containing atom and bond information.
    #         source_index: List or tensor of indices pointing to source atoms (colored blue).
    #         target_index: List or tensor of indices pointing to target atoms (colored red).
    #         output_file: File name for saving the molecule visualization as a PNG.
    #     """
    #     # Create an empty RDKit molecule
    #     mol = Chem.RWMol()

    #     # Add atoms to the molecule
    #     atom_mapping = {}  # Map PyTorch Geometric atom indices to RDKit atom indices
    #     for i, atomic_num in enumerate(data.x.tolist()):
    #         atom = Chem.Atom(atomic_num)
    #         atom_idx = mol.AddAtom(atom)
    #         atom_mapping[i] = atom_idx

    #     # Add bonds to the molecule
    #     bond_types = [
    #         Chem.BondType.SINGLE,
    #         Chem.BondType.DOUBLE,
    #         Chem.BondType.TRIPLE,
    #         Chem.BondType.AROMATIC,
    #     ]
    #     for edge, edge_attr in zip(data.edge_index.T.tolist(), data.edge_attr.tolist()):
    #         start, end = edge
    #         bond_type_idx = edge_attr.index(
    #             1
    #         )  # Find the index of the one-hot encoded bond type
    #         bond_type = bond_types[bond_type_idx]
    #         try:
    #             mol.AddBond(atom_mapping[start], atom_mapping[end], bond_type)
    #         except Exception as e:
    #             print(f"Error adding bond {start}-{end}: {e}")
    #             continue

    #     # Finalize the molecule
    #     mol = mol.GetMol()

    #     # Prepare atom coloring
    #     atom_colors = {}
    #     for idx in source_index:
    #         atom_colors[atom_mapping[idx.detach().item()]] = (
    #             0.0,
    #             0.0,
    #             1.0,
    #         )  # Blue for source atoms
    #     for idx in target_index:
    #         atom_colors[atom_mapping[idx.detach().item()]] = (
    #             1.0,
    #             0.0,
    #             0.0,
    #         )  # Red for target atoms

    #     # Visualize the molecule
    #     drawer = Draw.MolDraw2DCairo(500, 500)  # Create a 500x500 PNG canvas
    #     drawer.DrawMolecule(
    #         mol,
    #         highlightAtoms=list(atom_colors.keys()),
    #         highlightAtomColors=atom_colors,
    #     )
    #     drawer.FinishDrawing()

    #     # Save the image to a file
    #     with open(output_file, "wb") as f:
    #         f.write(drawer.GetDrawingText())

    #     print(f"Molecule visualization saved to {output_file}")
    #     return mol
