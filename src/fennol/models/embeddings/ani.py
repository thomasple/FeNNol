import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict
import numpy as np
from ...utils.periodic_table import PERIODIC_TABLE


class ANIAEV(nn.Module):
    """
    Computes the Atomic Environment Vector (AEV) for a given molecular system using the ANI model.

    Args:
        species_order (Sequence[str]): The atomic species which are considered by the model.
        graph_angle_key (str): The key in the input dictionary that corresponds to the angular graph.
        cutoff (float, optional): The radial cutoff distance in Angstroms. Default is 5.2.
        angular_cutoff (float, optional): The angular cutoff distance in Angstroms. Default is 3.5.
        radial_eta (float, optional): The radial eta hyperparameter. Default is 16.0.
        angular_eta (float, optional): The angular eta hyperparameter. Default is 8.0.
        radial_dist_divisions (int, optional): The number of radial distance divisions. Default is 16.
        angular_dist_divisions (int, optional): The number of angular distance divisions. Default is 4.
        zeta (float, optional): The zeta hyperparameter. Default is 32.0.
        angle_sections (int, optional): The number of angular sections. Default is 4.
        radial_start (float, optional): The starting radial distance. Default is 0.8.
        angular_start (float, optional): The starting angular distance. Default is 0.8.
        embedding_key (str, optional): The key to use for the output embedding in the returned dictionary. Default is "embedding".
        graph_key (str, optional): The key in the input dictionary that corresponds to the radial graph. Default is "graph".
    """

    _graphs_properties: Dict
    species_order: Sequence[str]
    graph_angle_key: str
    radial_eta: float = 16.0
    angular_eta: float = 8.0
    radial_dist_divisions: int = 16
    angular_dist_divisions: int = 4
    zeta: float = 32.0
    angle_sections: int = 4
    radial_start: float = 0.8
    angular_start: float = 0.8
    embedding_key: str = "embedding"
    graph_key: str = "graph"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        rev_idx = {s: k for k, s in enumerate(PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())

        #convert species to internal indices
        conv_tensor = [0] * (maxidx + 2)
        for i, s in enumerate(self.species_order):
            conv_tensor[rev_idx[s]] = i
        indices = jnp.asarray(conv_tensor, dtype=jnp.int32)[species]
        num_species = len(self.species_order)
        num_species_pair = (num_species * (num_species + 1)) // 2

        # Radial graph
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        switch = graph["switch"]
        edge_src = graph["edge_src"]
        edge_dst = graph["edge_dst"]

        # Radial AEV
        cutoff = self._graphs_properties[self.graph_key]["cutoff"]
        shiftR = jnp.asarray(
            np.linspace(self.radial_start, cutoff, self.radial_dist_divisions + 1)[
                None, :-1
            ],
            dtype=distances.dtype,
        )
        x2 = self.radial_eta * (distances[:, None] - shiftR) ** 2
        radial_terms = 0.25 * jnp.exp(-x2) * switch[:, None]
        # aggregate radial AEV
        radial_index = edge_src * num_species + indices[edge_dst]

        radial_aev = jax.ops.segment_sum(
            radial_terms, radial_index, num_species * species.shape[0]
        ).reshape(species.shape[0], num_species * radial_terms.shape[-1])

        # Angular graph
        graph = inputs[self.graph_angle_key]
        angles = graph["angles"]
        distances = graph["distances"]
        central_atom = graph["central_atom"]
        angle_src, angle_dst = graph["angle_src"], graph["angle_dst"]
        switch = graph["switch"]
        d12 = 0.5 * (distances[angle_src] + distances[angle_dst])[:, None]

        # Angular AEV parameters
        angular_cutoff = self._graphs_properties[self.graph_angle_key]["cutoff"]
        angle_start = np.pi / (2 * self.angle_sections)
        shiftZ = jnp.asarray(
            (np.linspace(0, np.pi, self.angle_sections + 1) + angle_start)[None, :-1],
            dtype=distances.dtype,
        )
        shiftA = jnp.asarray(
            np.linspace(
                self.angular_start, angular_cutoff, self.angular_dist_divisions + 1
            )[None, :-1],
            dtype=distances.dtype,
        )

        # Angular AEV
        factor1 = (0.5 + 0.5 * jnp.cos(angles[:, None] - shiftZ)) ** self.zeta
        factor2 = jnp.exp(-self.angular_eta * (d12 - shiftA) ** 2)
        angular_terms = (
            (factor1[:, None, :] * factor2[:, :, None]).reshape(
                -1, self.angle_sections * self.angular_dist_divisions
            )
            * 2
            * (switch[angle_src] * switch[angle_dst])[:, None]
        )

        # aggregate angular AEV
        index_dest = indices[graph["edge_dst"]]
        species1, species2 = np.triu_indices(num_species, 0)
        pair_index = np.arange(species1.shape[0], dtype=np.int32)
        triu_index = np.zeros((num_species, num_species), dtype=np.int32)
        triu_index[species1, species2] = pair_index
        triu_index[species2, species1] = pair_index
        triu_index = jnp.asarray(triu_index, dtype=jnp.int32)
        angular_index = (
            central_atom * num_species_pair
            + triu_index[index_dest[angle_src], index_dest[angle_dst]]
        )

        angular_aev = jax.ops.segment_sum(
            angular_terms, angular_index, num_species_pair * species.shape[0]
        ).reshape(species.shape[0], num_species_pair * angular_terms.shape[-1])

        embedding = jnp.concatenate((radial_aev, angular_aev), axis=-1)
        if self.embedding_key is None:
            return embedding
        return {**inputs, self.embedding_key: embedding}
