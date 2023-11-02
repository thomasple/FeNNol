import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict
import numpy as np
import dataclasses

from ...utils.periodic_table import PERIODIC_TABLE, VALENCE_STRUCTURE
from ..encodings import SpeciesEncoding


class EEACSF(nn.Module):
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
    species_encoding: dict = dataclasses.field(default_factory=dict)

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        rev_idx = {s: k for k, s in enumerate(PERIODIC_TABLE)}
        maxidx = max(rev_idx.values())

        # Radial graph
        graph = inputs[self.graph_key]
        distances = graph["distances"]
        switch = graph["switch"]
        edge_src = graph["edge_src"]
        edge_dst = graph["edge_dst"]

        # convert species to valence structure
        valence_structure = np.array(VALENCE_STRUCTURE,dtype=np.float64)
        valence_structure[:, 0] /= 2.0
        valence_structure[:, 1] /= 6.0
        valence_structure[:, 2] /= 10.0
        valence_structure[:, 3] /= 14.0
        valence_structure[valence_structure[:, 2] > 0][:, [0, 1]] = 0
        valence_structure[valence_structure[:, 3] > 0][:, [0, 1]] = 0
        valence = jnp.asarray(valence_structure,dtype=distances.dtype)[species]


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
        radial_aev = jax.ops.segment_sum(
            radial_terms[:, :, None] * valence[edge_dst, None, :],
            edge_src,
            species.shape[0],
        ).reshape(species.shape[0], -1)

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

        valence_dst = valence[graph["edge_dst"]]
        valence_ang_p = valence_dst[angle_src] + valence_dst[angle_dst]
        valence_ang_m = valence_dst[angle_src] * valence_dst[angle_dst]
        valence_ang = (valence_ang_p[:, :, None] * valence_ang_m[:, None, :]).reshape(
            -1, valence_ang_p.shape[1] * valence_ang_m.shape[1]
        )

        angular_aev = jax.ops.segment_sum(
            angular_terms[:, :, None] * valence_ang[:, None, :],
            central_atom,
            species.shape[0],
        ).reshape(species.shape[0], -1)

        onehot = SpeciesEncoding(**self.species_encoding, name="SpeciesEncoding")(
            species
        )
        embedding = jnp.concatenate((onehot, radial_aev, angular_aev), axis=-1)
        if self.embedding_key is None:
            return embedding
        return {**inputs, self.embedding_key: embedding}
