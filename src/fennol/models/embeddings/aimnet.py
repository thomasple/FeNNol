import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, Dict, Callable, ClassVar,Union
import numpy as np
from ...utils.periodic_table import PERIODIC_TABLE
from ..misc.encodings import SpeciesEncoding
from ..misc.nets import FullyConnectedNet
import dataclasses


class AIMNet(nn.Module):
    """Atom-In-Molecule Network message-passing embedding

    FID : AIMNET

    ### Reference
    Roman Zubatyuk et al. ,Accurate and transferable multitask prediction of chemical properties with an atoms-in-molecules neural network.Sci. Adv.5,eaav6490(2019).DOI:10.1126/sciadv.aav6490
    """

    _graphs_properties: Dict
    graph_angle_key: str
    """ The key in the input dictionary that corresponds to the angular graph. """
    nlayers: int = 3
    """ The number of message-passing layers."""
    zmax: int = 86
    """ The maximum atomic number to allocate AFV."""
    radial_eta: float = 16.0
    """ Controls the width of the gaussian sensity functions in radial AEV."""
    angular_eta: float = 8.0
    """ Controls the width of the gaussian sensity functions in angular AEV."""
    radial_dist_divisions: int = 16
    """ Number of basis function to encode ditance in radial AEV."""
    angular_dist_divisions: int = 4
    """ Number of basis function to encode ditance in angular AEV."""
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
    keep_all_layers: bool = False
    """ If True, the output will contain the embeddings from all layers."""

    activation: Union[Callable, str] = "swish"
    """ The activation function to use."""
    combination_neurons: Sequence[int] = dataclasses.field(
        default_factory=lambda: [256, 128, 16]
    )
    """ The number of neurons in the AFV combination network."""

    embedding_neurons: Sequence[int] = dataclasses.field(
        default_factory=lambda: [512, 256, 256]
    )
    """ The number of neurons in the embedding network."""
    interaction_neurons: Sequence[int] = dataclasses.field(
        default_factory=lambda: [256, 256, 128]
    )
    """ The number of neurons in the interaction network."""
    afv_neurons: Sequence[int] = dataclasses.field(
        default_factory=lambda: [256, 256, 16]
    )
    """ The number of neurons in the AFV update network. The last number of neurons defines the size of AFV."""

    FID: ClassVar[str] = "AIMNET"

    @nn.compact
    def __call__(self, inputs):
        """Forward pass of the AIMNet model."""
        species = inputs["species"]

        # species encoding (AFV)
        afv_dim = self.afv_neurons[-1]
        afv = SpeciesEncoding(dim=afv_dim, zmax=self.zmax, encoding="random")(species)

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

        # Angular graph
        graph = inputs[self.graph_angle_key]
        edge_dst_a = graph["edge_dst"]
        angles = graph["angles"]
        distances = graph["distances"]
        central_atom = graph["central_atom"]
        angle_src, angle_dst = graph["angle_src"], graph["angle_dst"]
        angle_atom_1 = edge_dst_a[angle_src]
        angle_atom_2 = edge_dst_a[angle_dst]
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

        if self.keep_all_layers:
            mis = []
        for layer in range(self.nlayers):
            # combine pair info
            Gij = (radial_terms[:, None, :] * afv[edge_dst, :, None]).reshape(
                radial_terms.shape[0], -1
            )
            Gri = jax.ops.segment_sum(Gij, edge_src, species.shape[0])

            # combine triplet info
            afv1 = afv[angle_atom_1]
            afv2 = afv[angle_atom_2]
            afv12 = jnp.concatenate((afv1 * afv2, afv1 + afv2), axis=-1)
            afv_ang = FullyConnectedNet(
                self.combination_neurons,
                activation=self.activation,
                name=f"combination_net_{layer}",
            )(afv12)
            Gijk = (angular_terms[:, None, :] * afv_ang[:, :, None]).reshape(
                angular_terms.shape[0], -1
            )
            Gai = jax.ops.segment_sum(Gijk, central_atom, species.shape[0])

            # environment field
            fi = FullyConnectedNet(
                self.embedding_neurons,
                activation=self.activation,
                name=f"embedding_net_{layer}",
            )(jnp.concatenate((Gri, Gai), axis=-1))

            # update AFV
            dafv = FullyConnectedNet(
                self.afv_neurons,
                activation=self.activation,
                name=f"afv_update_net_{layer}",
            )(fi)
            afv = afv + dafv

            if self.keep_all_layers or layer == self.nlayers - 1:
                # embedding
                mi = FullyConnectedNet(
                    self.interaction_neurons,
                    activation=self.activation,
                    name=f"interaction_net_{layer}",
                )(fi)
                if self.keep_all_layers:
                    mis.append(mi)

        embedding_key = (
            self.embedding_key if self.embedding_key is not None else self.name
        )

        output = {**inputs, embedding_key: mi, embedding_key + "_afv": afv}
        if self.keep_all_layers:
            output[embedding_key + "_layers"] = jnp.stack(mis, axis=1)

        return output
