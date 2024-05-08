import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict, Union, ClassVar
import numpy as np
from ...utils.periodic_table import PERIODIC_TABLE


class ANIAEV(nn.Module):
    """Computes the Atomic Environment Vector (AEV) for a given molecular system using the ANI model.

    FID : ANI_AEV

    ### Reference
    J. S. Smith, O. Isayev and A. E. Roitberg, ANI-1: an extensible neural network potential with DFT accuracy at force field computational cost, Chem. Sci., 2017, 8, 3192
    """

    _graphs_properties: Dict
    species_order: Union[str, Sequence[str]]
    """ The chemical species which are considered by the model."""
    graph_angle_key: str
    """ The key in the input dictionary that corresponds to the angular graph."""
    radial_eta: float = 16.0
    """ Controls the width of the gaussian sensitivity functions in radial AEV."""
    angular_eta: float = 8.0
    """ Controls the width of the gaussian sensitivity functions in angular AEV."""
    radial_dist_divisions: int = 16
    """ Number of basis function to encode distance in radial AEV."""
    angular_dist_divisions: int = 4
    """ Number of basis function to encode distance in angular AEV."""
    zeta: float = 32.0
    """ The power parameter in angle embedding."""
    angle_sections: int = 4
    """ The number of angle sections."""
    radial_start: float = 0.8
    """ The starting distance in radial AEV."""
    angular_start: float = 0.8
    """ The starting distance in angular AEV."""
    embedding_key: str = "embedding"
    """ The key to use for the output embedding in the returned dictionary."""
    graph_key: str = "graph"
    """ The key in the input dictionary that corresponds to the radial graph."""

    FID: ClassVar[str] = "ANI_AEV"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        rev_idx = {s: k for k, s in enumerate(PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())

        # convert species to internal indices
        conv_tensor = [0] * (maxidx + 2)
        if isinstance(self.species_order, str):
            species_order = [el.strip() for el in self.species_order.split(",")]
        else:
            species_order = [el for el in self.species_order]
        for i, s in enumerate(species_order):
            conv_tensor[rev_idx[s]] = i
        indices = jnp.asarray(conv_tensor, dtype=jnp.int32)[species]
        num_species = len(species_order)
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
            - np.linspace(self.radial_start, cutoff, self.radial_dist_divisions + 1)[
                None, :-1
            ],
            dtype=distances.dtype,
        )
        x2 = self.radial_eta * (distances[:, None] + shiftR) ** 2
        radial_terms = jnp.exp(-x2) * (0.25*switch)[:, None]
        # aggregate radial AEV
        radial_index = edge_src * num_species + indices[edge_dst]

        radial_aev = jax.ops.segment_sum(
            radial_terms, radial_index, num_species * species.shape[0]
        ).reshape(species.shape[0], num_species * radial_terms.shape[-1])

        # Angular graph
        # eta2 = self.angular_eta ** 0.5
        graph = inputs[self.graph_angle_key]
        angles = graph["angles"]
        distances = (0.5*self.angular_eta**0.5)*graph["distances"]
        central_atom = graph["central_atom"]
        angle_src, angle_dst = graph["angle_src"], graph["angle_dst"]
        d12 = (distances[angle_src] + distances[angle_dst])[:, None]

        # Angular AEV parameters
        angular_cutoff = self._graphs_properties[self.graph_angle_key]["cutoff"]
        angle_start = np.pi / (2 * self.angle_sections)
        shiftZ = jnp.asarray(
            -(np.linspace(0, np.pi, self.angle_sections + 1) + angle_start)[None, :-1],
            dtype=distances.dtype,
        )
        shiftA = jnp.asarray(
            (-self.angular_eta**0.5)*np.linspace(
                self.angular_start, angular_cutoff, self.angular_dist_divisions + 1
            )[None, :-1],
            dtype=distances.dtype,
        )

        # Angular AEV
        switch = graph["switch"] *(2.0 *(0.5**self.zeta))**0.5
        factor1 = switch[angle_src] * switch[angle_dst]
        factor1 = (
            factor1[:, None] * (1 + jnp.cos(angles[:, None] + shiftZ)) ** self.zeta
        )
        factor2 = jnp.exp(-(d12 + shiftA) ** 2)
        angular_terms = (factor1[:, None, :] * factor2[:, :, None]).reshape(
            -1, self.angle_sections * self.angular_dist_divisions
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
