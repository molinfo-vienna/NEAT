import matplotlib.pyplot as plt
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data
from torch_geometric.nn.pool import global_add_pool


class SourceTargetSplitter:
    def __init__(self, splitting_mode: str = "cyclic", target_set_max_size: int = -1):
        self.splitting_mode = splitting_mode
        self.target_set_max_size = target_set_max_size

    def create_source_target_split(self, data: Data, device: torch.device = None):
        """
        Creates a source-target split for a batch of data.

        Args:
            data (Data): The data to split.
            device (torch.device): The device to use for computations.

        Returns:
            A tuple containing the source-target split.
        """
        if self.splitting_mode == "random":
            return self.random_source_target_split(data, device=device)
        elif self.splitting_mode == "cyclic":
            return self.cyclic_source_target_split(data, device=device)
        elif self.splitting_mode == "hydrogen_random":
            return self.hydrogen_random_source_target_split(data, device=device)
        else:
            raise ValueError(f"Unknown splitting mode: {self.splitting_mode}")

    def random_source_target_split(self, data: Data, device: torch.device = None):
        """
        Randomly splits atoms into a source and target set.

        Args:
            data (Data): The data to split.
            device (torch.device): The device to use for computations.

        Returns:
            A tuple containing the source-target split.
        """
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

    def hydrogen_random_source_target_split(self, data: Data, device=None):
        """
        Splits atoms into source and target sets, putting a random proportion of hydrogen atoms in the target set
        and the remaining hydrogens and all heavy atoms in the source set.

        Args:
            data (Data): The data to split.
            device (torch.device): The device to use for computations.

        Returns:
            A tuple containing the source-target split.
        """
        batch_size = len(data)
        atom_counts = torch.bincount(data.batch, minlength=batch_size)

        # Identify hydrogen atoms (assuming atomic number 1 represents hydrogen)
        hydrogen_mask = data.x == 1  # Adjust indexing based on how x is structured
        hydrogen_counts = torch.bincount(
            data.batch[hydrogen_mask], minlength=batch_size
        )

        # Randomly select a subset of hydrogen atoms per molecule
        uniform_distribution = torch.rand(atom_counts.shape, device=device) * 0.999

        # This samples between 0 and up to N hydrogen atoms to delete per molecule
        hydrogens_to_delete = (
            (hydrogen_counts.float() + 1) * uniform_distribution
        ).long()
        hydrogens_to_keep = hydrogen_counts - hydrogens_to_delete
        hydrogen_indices = torch.nonzero(hydrogen_mask, as_tuple=False).squeeze()
        hydrogen_batch = data.batch[hydrogen_indices]

        # Initialize a list to store sampled nodes
        sampled_nodes = []

        # Iterate over each graph
        for graph_id in range(batch_size):
            # Get nodes belonging to the current graph
            graph_mask = hydrogen_batch == graph_id
            graph_nodes = hydrogen_indices[graph_mask]

            # Get the number of nodes to sample for this graph
            num_samples = hydrogens_to_keep[graph_id]

            # Randomly sample nodes (without replacement)
            sampled = graph_nodes[torch.randperm(len(graph_nodes))[:num_samples]]
            sampled_nodes.append(sampled)
        subset_hydrogen_idx = torch.cat(sampled_nodes)

        subset_mask = torch.zeros_like(data.batch, device=device, dtype=torch.bool)
        subset_mask[subset_hydrogen_idx] = 1
        subset_mask |= ~hydrogen_mask

        x_source = data.x[subset_mask]
        pos_source = data.pos[subset_mask]
        batch_source = data.batch[subset_mask]
        x_target = data.x[~subset_mask]
        pos_target = data.pos[~subset_mask]
        batch_target = data.batch[~subset_mask]
        stop_tokens = hydrogens_to_delete == 0

        # This is for debugging purposes
        # source_set_idx = torch.nonzero(subset_mask, as_tuple=False).squeeze()
        # target_set_idx = torch.nonzero(~subset_mask, as_tuple=False).squeeze()
        # data_point = data[0]
        # source_idx = source_set_idx[source_set_idx < data_point.num_nodes]
        # target_idx = target_set_idx[target_set_idx < data_point.num_nodes]
        # create_rdkit_molecule(data_point, source_idx, target_idx)
        # source_set_histogram(data, source_set_idx)

        return (
            x_source,
            pos_source,
            batch_source,
            x_target,
            pos_target,
            batch_target,
            stop_tokens,
        )

    def cyclic_source_target_split(self, data: Data, device: torch.device = None):
        """
        Splits atoms into source and target sets, using a cyclic approach.

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
            (marked_nodes_eccentricity * 1.5)
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
            one_hop_neighbours_idx = torch.unique(
                one_hop_neighbours_idx
            )  # TODO: This should be done before sampling, no?

            # Randomly pick a subset of these neighbours
            permutation = torch.randperm(one_hop_neighbours_idx.size(0))
            one_hop_neighbours_idx = one_hop_neighbours_idx[permutation]
            one_hop_neighbours_idx = one_hop_neighbours_idx[
                : int(len(one_hop_neighbours_idx) * 0.55)
            ]

            # Add these neighbours to the random nodes
            marked_nodes_idx = torch.cat((marked_nodes_idx, one_hop_neighbours_idx))
            marked_nodes_idx = torch.unique(marked_nodes_idx)

        # By excluding the target set atoms, we can already create the source set
        source_set_idx = marked_nodes_idx
        source_set_mask = torch.zeros_like(data.batch, device=device, dtype=torch.bool)
        source_set_mask[source_set_idx] = 1
        x_source = data.x[source_set_mask]
        pos_source = data.pos[source_set_mask]
        batch_source = data.batch[source_set_mask]

        # Now we can find all one-hop neighbours of the source set in the full graph
        # The interaction of these neighbours with the above target set defines the actual target set
        target_set_mask = ~source_set_mask
        target_set_idx = torch.nonzero(target_set_mask, as_tuple=False).squeeze()
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

        atom_count_source = global_add_pool(source_set_mask.int(), data.batch)
        stop_tokens = atom_count_source == atom_counts

        # This is for debugging purposes
        # data_point = data[0]
        # target_idx = target_set_neighbours_idx[
        #     target_set_neighbours_idx < data_point.num_nodes
        # ]
        # source_idx = source_set_idx[source_set_idx < data_point.num_nodes]
        # create_rdkit_molecule(data_point, source_idx, target_idx)
        # source_set_histogram(data, source_set_idx)

        return (
            x_source,
            pos_source,
            batch_source,
            x_target,
            pos_target,
            batch_target,
            stop_tokens,
        )

    @staticmethod
    def create_rdkit_molecule(
        data: Data,
        source_index: torch.Tensor,
        target_index: torch.Tensor,
        output_file: str = "molecule.png",
    ):
        """
        Creates an RDKit molecule object from PyTorch Geometric data and visualizes it.
        This is used for debugging purposes.

        Parameters:
            data (Data): PyTorch Geometric data object containing atom and bond information.
            source_index (torch.Tensor): List or tensor of indices pointing to source atoms (colored blue).
            target_index (torch.Tensor): List or tensor of indices pointing to target atoms (colored red).
            output_file (str): File name for saving the molecule visualization as a PNG.
        """
        # Create an empty RDKit molecule
        mol = Chem.RWMol()

        # Add atoms to the molecule
        atom_mapping = {}  # Map PyTorch Geometric atom indices to RDKit atom indices
        for i, atomic_num in enumerate(data.x.tolist()):
            atom = Chem.Atom(atomic_num)
            atom_idx = mol.AddAtom(atom)
            atom_mapping[i] = atom_idx

        # Add bonds to the molecule
        bond_types = [
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        ]
        for edge, edge_attr in zip(data.edge_index.T.tolist(), data.edge_attr.tolist()):
            start, end = edge
            bond_type_idx = edge_attr.index(
                1
            )  # Find the index of the one-hot encoded bond type
            bond_type = bond_types[bond_type_idx]
            try:
                mol.AddBond(atom_mapping[start], atom_mapping[end], bond_type)
            except Exception as e:
                print(f"Error adding bond {start}-{end}: {e}")
                continue

        # Finalize the molecule
        mol = mol.GetMol()

        # Prepare atom coloring
        atom_colors = {}
        for idx in source_index:
            atom_colors[atom_mapping[idx.detach().item()]] = (
                0.0,
                0.0,
                1.0,
            )  # Blue for source atoms
        for idx in target_index:
            atom_colors[atom_mapping[idx.detach().item()]] = (
                1.0,
                0.0,
                0.0,
            )  # Red for target atoms

        # Visualize the molecule
        drawer = Draw.MolDraw2DCairo(500, 500)  # Create a 500x500 PNG canvas
        drawer.DrawMolecule(
            mol,
            highlightAtoms=list(atom_colors.keys()),
            highlightAtomColors=atom_colors,
            highlightBonds=[],
        )
        drawer.FinishDrawing()

        # Save the image to a file
        with open(output_file, "wb") as f:
            f.write(drawer.GetDrawingText())

        print(f"Molecule visualization saved to {output_file}")
        return mol

    @staticmethod
    def source_set_histogram(data: Data, source_set_idx: torch.Tensor):
        """
        Creates a histogram of the source set size ratios.
        This is used for debugging purposes.

        Args:
            data (Data): The data to create the histogram from.
            source_set_idx (torch.Tensor): The indices of the source set.

        Returns:
            None
        """
        batch_source = data.batch[source_set_idx]
        source_count = global_add_pool(torch.ones_like(batch_source), batch_source)
        total_count = global_add_pool(torch.ones_like(data.batch), data.batch)
        ratio = source_count / total_count
        plt.figure()
        plt.hist(ratio.cpu().numpy(), bins=10, range=(0, 1))
        plt.xlabel("Source set size ratio")
        plt.ylabel("Number of molecules")
        plt.title("Histogram of source set size ratios")
        plt.savefig("source_set_histogram.png")
