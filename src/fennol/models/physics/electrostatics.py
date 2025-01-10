import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np

# from jaxopt.linear_solve import solve_cg, solve_iterative_refinement, solve_gmres
from typing import Any, Dict, Union, Callable, Sequence, Optional, ClassVar
from ...utils import AtomicUnits as au
import dataclasses
from ...utils.periodic_table import (
    D3_ELECTRONEGATIVITIES,
    D3_HARDNESSES,
    D3_VDW_RADII,
    D3_COV_RADII,
    D3_KAPPA,
    VDW_RADII,
    POLARIZABILITIES,
    VALENCE_ELECTRONS,
)
import math


def prepare_reciprocal_space(
    cells, reciprocal_cells, coordinates, batch_index, k_points, bewald
):
    """Prepare variables for Ewald summation in reciprocal space"""
    A = reciprocal_cells
    if A.shape[0] == 1:
        s = coordinates @ A[0]
        ks = 2j * jnp.pi * jnp.einsum("ai,ki-> ak", s, k_points[0])  # nat x nk
    else:
        s = jnp.einsum("aj,aji->ai", coordinates,A[batch_index])
        ks = (
            2j * jnp.pi * jnp.einsum("ai,aki-> ak", s, k_points[batch_index])
        )  # nat x nk

    m2 = jnp.sum(
        jnp.einsum("ski,sji->skj", k_points, A) ** 2,
        axis=-1,
    )  # nsys x nk
    a2 = (jnp.pi / bewald) ** 2
    expfac = jnp.exp(-a2 * m2) / m2  # nsys x nk

    volume = jnp.abs(jnp.linalg.det(cells))  # nsys
    phiscale = (au.BOHR / jnp.pi) / volume
    selfscale = bewald * (2 * au.BOHR / jnp.pi**0.5)
    return batch_index, k_points, phiscale, selfscale, expfac, ks


def ewald_reciprocal(q, batch_index, k_points, phiscale, selfscale, expfac, ks):
    """Compute Coulomb interactions in reciprocal space using Ewald summation"""
    if phiscale.shape[0] == 1:
        Sm = (q[:, None] * jnp.exp(ks)).sum(axis=0)[None, :]  # nys x nk
    else:
        Sm = jax.ops.segment_sum(
            q[:, None] * jnp.exp(ks), batch_index, k_points.shape[0]
        )  # nsys x nk

    ### compute reciprocal Coulomb potential (https://arxiv.org/abs/1805.10363)
    phi = (
        jnp.real(((Sm * expfac)[batch_index] * jnp.exp(-ks)).sum(axis=-1))
        * phiscale[batch_index]
        - q * selfscale
    )

    return 0.5 * q * phi, phi


# def ewald_reciprocal(
#     q, cells, reciprocal_cells, coordinates, batch_index, k_points, bewald
# ):
#     A = reciprocal_cells
#     ### Ewald reciprocal space

#     if A.shape[0] == 1:
#         s = jnp.einsum("ij,aj->ai", A[0], coordinates)
#         ks = 2j * jnp.pi * jnp.einsum("ai,ki-> ak", s, k_points[0])  # nat x nk
#         Sm = (q[:, None] * jnp.exp(ks)).sum(axis=0)[None, :]  # nys x nk
#     else:
#         s = jnp.einsum("aij,aj->ai", A[batch_index], coordinates)
#         ks = (
#             2j * jnp.pi * jnp.einsum("ai,aki-> ak", s, k_points[batch_index])
#         )  # nat x nk
#         Sm = jax.ops.segment_sum(
#             q[:, None] * jnp.exp(ks), batch_index, k_points.shape[0]
#         )  # nsys x nk

#     m2 = jnp.sum(
#         jnp.einsum("sij,ski->skj", A, k_points) ** 2,
#         axis=-1,
#     )  # nsys x nk
#     a2 = (jnp.pi / bewald) ** 2
#     expfac = Sm * jnp.exp(-a2 * m2) / m2  # nsys x nk
#     volume = jnp.linalg.det(cells)  # nsys

#     ### compute reciprocal Coulomb potential (https://arxiv.org/abs/1805.10363)
#     phi = jnp.real((expfac[batch_index] * jnp.exp(-ks)).sum(axis=-1)) * (
#         (au.BOHR / jnp.pi) / volume[batch_index]
#     ) - q * (bewald * (2 * au.BOHR / jnp.pi**0.5))

#     return 0.5 * q * phi, phi


class Coulomb(nn.Module):
    """Coulomb interaction between distributed point charges

    FID: COULOMB   
    
    """
    _graphs_properties: Dict
    graph_key: str = "graph"
    """Key for the graph in the inputs"""
    charges_key: str = "charges"
    """Key for the charges in the inputs"""
    energy_key: Optional[str] = None
    """Key for the energy in the outputs"""
    # switch_fraction: float = 0.9
    scale: Optional[float] = None
    """Scaling factor for the energy"""
    charge_scale: Optional[float] = None
    """Scaling factor for the charges"""
    damp_style: Optional[str] = None
    """Damping style. Available options are: None, 'TS', 'OQDO', 'D3', 'SPOOKY', 'CP', 'KEY'"""
    damp_params: Dict = dataclasses.field(default_factory=dict)
    """Damping parameters"""
    bscreen: float = -1.0
    """Screening parameter. If >0, the Coulomb potential becomes a Yukawa potential and the reciprocal space is not computed"""
    trainable: bool = True
    """Whether the parameters are trainable"""
    _energy_unit: str = "Ha"
    """The energy unit of the model. **Automatically set by FENNIX**"""

    FID: ClassVar[str] = "COULOMB"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        distances = graph["distances"]
        switch = graph["switch"]

        rij = distances / au.BOHR
        q = inputs[self.charges_key]
        if q.shape[-1] == 1:
            q = jnp.squeeze(q, axis=-1)
        if self.charge_scale is not None:
            if self.trainable:
                charge_scale = jnp.abs(
                    self.param(
                        "charge_scale", lambda key: jnp.asarray(self.charge_scale)
                    )
                )
            else:
                charge_scale = self.charge_scale
            q = q * charge_scale

        damp_style = self.damp_style.upper() if self.damp_style is not None else None

        do_recip = self.bscreen <= 0.0 and "k_points" in graph

        if self.bscreen > 0.0:
            # dirfact = jax.scipy.special.erfc(self.bscreen * distances)
            dirfact = jnp.exp(-self.bscreen * distances)
        elif do_recip:
            k_points = graph["k_points"]
            bewald = graph["b_ewald"]
            cells = inputs["cells"]
            reciprocal_cells = inputs["reciprocal_cells"]
            batch_index = inputs["batch_index"]
            erec, _ = ewald_reciprocal(
                q,
                *prepare_reciprocal_space(
                    cells,
                    reciprocal_cells,
                    inputs["coordinates"],
                    batch_index,
                    k_points,
                    bewald,
                ),
            )
            dirfact = jax.scipy.special.erfc(bewald * distances)
        else:
            dirfact = 1.0

        if damp_style is None:
            Aij = switch * dirfact / rij
            eat = (
                0.5
                * q
                * jax.ops.segment_sum(Aij * q[edge_dst], edge_src, species.shape[0])
            )

        elif damp_style == "TS":
            cpB = self.damp_params.get("cpB", 3.5)
            s = self.damp_params.get("s", 2.4)

            if self.trainable:
                cpB = jnp.abs(self.param("cpB", lambda key: jnp.asarray(cpB)))
                s = jnp.abs(self.param("s", lambda key: jnp.asarray(s)))

            ratiovol_key = self.damp_params.get("ratiovol_key", None)
            if ratiovol_key is not None:
                ratiovol = inputs[ratiovol_key]
                if ratiovol.shape[-1] == 1:
                    ratiovol = jnp.squeeze(ratiovol, axis=-1)
                rvdw = jnp.asarray(VDW_RADII)[species] * ratiovol ** (1.0 / 3.0)
            else:
                rvdw = jnp.asarray(VDW_RADII)[species]
            Rij = rvdw[edge_src] + rvdw[edge_dst]
            Bij = cpB * (rij / Rij) ** s

            eBij = jnp.where(Bij < 20.0, jnp.exp(-Bij), 0.0)

            Aij = (dirfact - eBij) / rij * switch
            eat = (
                0.5
                * q
                * jax.ops.segment_sum(Aij * q[edge_dst], edge_src, species.shape[0])
            )

        elif damp_style == "OQDO":
            ratiovol_key = self.damp_params.get("ratiovol_key", None)
            alpha = jnp.asarray(POLARIZABILITIES)[species]
            if ratiovol_key is not None:
                ratiovol = inputs[ratiovol_key]
                if ratiovol.shape[-1] == 1:
                    ratiovol = jnp.squeeze(ratiovol, axis=-1)
                alpha = alpha * ratiovol

            alphaij = 0.5 * (alpha[edge_src] + alpha[edge_dst])
            Re = (alphaij * (128.0 / au.FSC ** (4.0 / 3.0))) ** (1.0 / 7.0)
            Re2 = Re**2
            Re4 = Re**4
            muw = (
                3.66316787e01
                - 5.79579187 * Re
                + 3.02674813e-01 * Re2
                - 3.65461255e-04 * Re4
            ) / (-1.46169102e01 + 7.32461225 * Re)
            # muw = (
            #     4.83053463e-01
            #     - 3.76191669e-02 * Re
            #     + 1.27066988e-03 * Re2
            #     - 7.21940151e-07 * Re4
            # ) / (3.84212120e-02 - 3.16915319e-02 * Re + 2.37410890e-02 * Re2)
            Bij = 0.5 * muw * rij**2

            eBij = jnp.where(Bij < 20.0, jnp.exp(-Bij), 0.0)

            Aij = (dirfact - eBij) / rij * switch
            eat = (
                0.5
                * q
                * jax.ops.segment_sum(Aij * q[edge_dst], edge_src, species.shape[0])
            )

        elif damp_style == "D3":
            ratiovol_key = self.damp_params.get("ratiovol_key", None)
            if ratiovol_key is not None:
                ratiovol = inputs[ratiovol_key]
                if ratiovol.shape[-1] == 1:
                    ratiovol = jnp.squeeze(ratiovol, axis=-1)
            else:
                ratiovol = 1.0

            gamma_scheme = self.damp_params.get("gamma_scheme", "D3")
            if gamma_scheme == "D3":
                if self.trainable:
                    rvdw = jnp.abs(
                        self.param("rvdw", lambda key: jnp.asarray(VDW_RADII))
                    )[species]
                else:
                    rvdw = jnp.asarray(VDW_RADII)[species]
                rvdw = rvdw * ratiovol ** (1.0 / 3.0)
                ai2 = rvdw**2
                gamma_ij = (ai2[edge_src] + ai2[edge_dst] + 1.0e-3) ** (-0.5)

            elif gamma_scheme == "QDO":
                gscale = self.damp_params.get("gamma_scale", 2.0)
                if self.trainable:
                    gscale = jnp.abs(
                        self.param("gamma_scale", lambda key: jnp.asarray(gscale))
                    )
                    alpha = jnp.abs(
                        self.param("alpha", lambda key: jnp.asarray(POLARIZABILITIES))
                    )[species]
                else:
                    alpha = jnp.asarray(POLARIZABILITIES)[species]
                alpha = alpha * ratiovol
                alphaij = 0.5 * (alpha[edge_src] + alpha[edge_dst])
                rvdwij = (alphaij * (128.0 / au.FSC ** (4.0 / 3.0))) ** (1.0 / 7.0)
                gamma_ij = gscale / rvdwij
            else:
                raise NotImplementedError(
                    f"gamma_scheme {gamma_scheme} not implemented"
                )

            Aij = (dirfact - jax.scipy.special.erfc(gamma_ij * rij)) / rij * switch

            eat = (
                0.5
                * q
                * jax.ops.segment_sum(Aij * q[edge_dst], edge_src, species.shape[0])
            )

        elif damp_style == "SPOOKY":
            shortrange_cutoff = self.damp_params.get("shortrange_cutoff", 5.0)
            r_on = self.damp_params.get("r_on", 0.25) * shortrange_cutoff
            r_off = self.damp_params.get("r_off", 0.75) * shortrange_cutoff
            x1 = (distances - r_on) / (r_off - r_on)
            x2 = 1.0 - x1
            mask1 = x1 <= 1.0e-6
            mask2 = x2 <= 1.0e-6
            x1 = jnp.where(mask1, 1.0, x1)
            x2 = jnp.where(mask2, 1.0, x2)
            s1 = jnp.where(mask1, 0.0, jnp.exp(-1.0 / x1))
            s2 = jnp.where(mask2, 0.0, jnp.exp(-1.0 / x2))
            Bij = s2 / (s1 + s2)

            Aij = Bij / (rij**2 + 1) ** 0.5 + (dirfact - Bij) / rij * switch
            eat = (
                0.5
                * q
                * jax.ops.segment_sum(Aij * q[edge_dst], edge_src, species.shape[0])
            )

        elif damp_style == "CP":
            cpA = self.damp_params.get("cpA", 4.42)
            cpB = self.damp_params.get("cpB", 4.12)
            gamma = self.damp_params.get("gamma", 0.5)

            if self.trainable:
                cpA = jnp.abs(self.param("cpA", lambda key: jnp.asarray(cpA)))
                cpB = jnp.abs(self.param("cpB", lambda key: jnp.asarray(cpB)))
                gamma = jnp.abs(self.param("gamma", lambda key: jnp.asarray(gamma)))

            rvdw = jnp.asarray(VDW_RADII)[species]
            ratiovol_key = self.damp_params.get("ratiovol_key", None)
            if ratiovol_key is not None:
                ratiovol = inputs[ratiovol_key]
                if ratiovol.shape[-1] == 1:
                    ratiovol = jnp.squeeze(ratiovol, axis=-1)
                rvdw = rvdw * ratiovol ** (1.0 / 3.0)

            Zv = jnp.asarray(VALENCE_ELECTRONS)[species]
            Zi, Zj = Zv[edge_src], Zv[edge_dst]
            qi, qj = q[edge_src], q[edge_dst]
            rvdwi, rvdwj = rvdw[edge_src], rvdw[edge_dst]

            eAi = jnp.exp(-cpA * rij / rvdwi)
            eAj = jnp.exp(-cpA * rij / rvdwj)
            eBi = jnp.exp(-cpB * rij / rvdwi)
            eBj = jnp.exp(-cpB * rij / rvdwj)
            eBij = eBi * eBj - eBi - eBj

            Bshort = jnp.exp(-gamma * distances**4)
            Dshort = 1.0 - Bshort
            ecp = Dshort * (
                Zi * Zj * (eAi + eAj + eBij)
                - qi * Zj * (eAi + eBij)
                - qj * Zi * (eAj + eBij)
            )

            # eq = qi * qj * (1 + eBij) * (1 - Bshort)
            eqq = qi * qj * (dirfact - Bshort + eBij * Dshort)

            epair = (ecp + eqq) * switch / rij

            # epair = (
            #     (1 - Bshort)
            #     * (
            #         Zi * Zj * (eAi + eAj + eBij)
            #         - qi * Zj * (eAi + eBij)
            #         - qj * Zi * (eAj + eBij)
            #         + qi * qj * (1 + eBij)
            #     )
            #     * switch
            #     / rij
            # )
            eat = 0.5 * jax.ops.segment_sum(epair, edge_src, species.shape[0])

        elif damp_style == "KEY":
            damp_key = self.damp_params["key"]
            damp = inputs[damp_key]
            epair = (dirfact - damp) * switch / rij
            eat = 0.5 * jax.ops.segment_sum(epair, edge_src, species.shape[0])
        else:
            raise NotImplementedError(f"damp_style {self.damp_style} not implemented")

        if do_recip:
            eat = eat + erec

        if self.scale is not None:
            if self.trainable:
                scale = jnp.abs(
                    self.param("scale", lambda key: jnp.asarray(self.scale))
                )
            else:
                scale = self.scale
            eat = eat * scale

        energy_key = self.energy_key if self.energy_key is not None else self.name
        energy_unit = au.get_multiplier(self._energy_unit)
        out = {**inputs, energy_key: eat*energy_unit}
        if do_recip:
            out[energy_key + "_reciprocal"] = erec*energy_unit
        return out


class QeqD4(nn.Module):
    """ QEq-D4 charge equilibration scheme

    FID: QEQ_D4
    
    ### Reference
    E. Caldeweyher et al.,A generally applicable atomic-charge dependent London dispersion correction,
    J Chem Phys. 2019 Apr 21;150(15):154122. (https://doi.org/10.1063/1.5090222)
    """
    graph_key: str = "graph"
    """Key for the graph in the inputs"""
    trainable: bool = False
    """Whether the parameters are trainable"""
    charges_key: str = "charges"
    """Key for the charges in the outputs. 
        If charges are provided in the inputs, they are not re-optimized and we only compute the energy"""
    energy_key: Optional[str] = None
    """Key for the energy in the outputs"""
    chi_key: Optional[str] = None
    """Key for additional electronegativity in the inputs"""
    c3_key: Optional[str] = None
    """Key for additional c3 in the inputs. Only used if charges are provided in the inputs"""
    c4_key: Optional[str] = None
    """Key for additional c4 in the inputs. Only used if charges are provided in the inputs"""
    total_charge_key: str = "total_charge"
    """Key for the total charge in the inputs"""
    non_interacting_guess: bool = False
    """Whether to use the non-interacting limit as an initial guess."""
    solver: str = "gmres"
    _energy_unit: str = "Ha"
    """The energy unit of the model. **Automatically set by FENNIX**"""

    FID: ClassVar[str] = "QEQ_D4"

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        switch = graph["switch"]

        rij = graph["distances"] / au.BOHR

        do_recip = "k_points" in graph
        if do_recip:
            k_points = graph["k_points"]
            bewald = graph["b_ewald"]
            cells = inputs["cells"]
            reciprocal_cells = inputs["reciprocal_cells"]
            batch_index = inputs["batch_index"]
            dirfact = jax.scipy.special.erfc(bewald * graph["distances"])
            ewald_params = prepare_reciprocal_space(
                cells,
                reciprocal_cells,
                inputs["coordinates"],
                batch_index,
                k_points,
                bewald,
            )
        else:
            dirfact = 1.0

        Jii = D3_HARDNESSES
        ai = D3_VDW_RADII
        ETA = [jii + (2.0 / np.pi) ** 0.5 / aii for jii, aii in zip(Jii, ai)]

        # D3 parameters
        if self.trainable:
            ENi = self.param("EN", lambda key: jnp.asarray(D3_ELECTRONEGATIVITIES))[
                species
            ]
            # Jii = self.param("J", lambda key: jnp.asarray(D3_HARDNESSES))[species]
            eta = jnp.abs(self.param("eta", lambda key: jnp.asarray(ETA)))[species]
            ai = jnp.abs(self.param("a", lambda key: jnp.asarray(D3_VDW_RADII)))[
                species
            ]
            rci = jnp.abs(self.param("rc", lambda key: jnp.asarray(D3_COV_RADII)))[
                species
            ]
            c3 = self.param("c3", lambda key: jnp.zeros(len(D3_ELECTRONEGATIVITIES)))[
                species
            ]
            c4 = jnp.abs(
                self.param("c4", lambda key: jnp.zeros(len(D3_ELECTRONEGATIVITIES)))[
                    species
                ]
            )
            kappai = self.param("kappa", lambda key: jnp.asarray(D3_KAPPA))[species]
            k1 = jnp.abs(self.param("k1", lambda key: jnp.asarray(7.5)))
            training = "training" in inputs.get("flags", {})
            if training:
                regularization = (
                    (ENi - jnp.asarray(D3_ELECTRONEGATIVITIES)[species]) ** 2
                    + (eta - jnp.asarray(ETA)[species]) ** 2
                    + (ai - jnp.asarray(D3_VDW_RADII)[species]) ** 2
                    + (rci - jnp.asarray(D3_COV_RADII)[species]) ** 2
                    + (kappai - jnp.asarray(D3_KAPPA)[species]) ** 2
                    + (k1 - 7.5) ** 2
                )
        else:
            c3 = jnp.zeros_like(species, dtype=jnp.float32)
            c4 = jnp.zeros_like(species, dtype=jnp.float32)
            ENi = jnp.asarray(D3_ELECTRONEGATIVITIES)[species]
            # Jii = jnp.asarray(D3_HARDNESSES)[species]
            eta = jnp.asarray(ETA)[species]
            ai = jnp.asarray(D3_VDW_RADII)[species]
            rci = jnp.asarray(D3_COV_RADII)[species]
            kappai = jnp.asarray(D3_KAPPA)[species]
            k1 = 7.5

        ai2 = ai**2
        rcij = (
            rci.at[edge_src].get(mode="fill", fill_value=1.0)
            + rci.at[edge_dst].get(mode="fill", fill_value=1.0)
            + 1.0e-3
        )
        mCNij = 1.0 + jax.scipy.special.erf(-k1 * (rij / rcij - 1))
        mCNi = 0.5 * jax.ops.segment_sum(mCNij * switch, edge_src, species.shape[0])
        chi = ENi - kappai * (mCNi + 1.0e-3) ** 0.5
        if self.chi_key is not None:
            chi = chi + inputs[self.chi_key]

        gamma_ij = (
            ai2.at[edge_src].get(mode="fill", fill_value=1.0)
            + ai2.at[edge_dst].get(mode="fill", fill_value=1.0)
            + 1.0e-3
        ) ** (-0.5)

        Aii = eta  # Jii + ((2.0 / np.pi) ** 0.5) / ai
        # Aij = jax.scipy.special.erf(gamma_ij * rij) / rij * switch
        Aij = (dirfact - jax.scipy.special.erfc(gamma_ij * rij)) / rij * switch

        if self.charges_key in inputs:
            q = inputs[self.charges_key]
            q_ = q
        else:
            nsys = inputs["natoms"].shape[0]
            batch_index = inputs["batch_index"]

            def matvec(x):
                l, q = jnp.split(x, (nsys,))
                Aq_self = Aii * q
                qdest = q.at[edge_dst].get(mode="fill", fill_value=0.0)
                Aq_pair = jax.ops.segment_sum(Aij * qdest, edge_src, species.shape[0])
                Aq = (
                    Aq_self
                    + Aq_pair
                    + l.at[batch_index].get(mode="fill", fill_value=0.0)
                )
                if do_recip:
                    _, phirec = ewald_reciprocal(q, *ewald_params)
                    Aq = Aq + phirec
                Al = jax.ops.segment_sum(q, batch_index, nsys)
                return jnp.concatenate((Al, Aq))

            Qtot = (
                inputs[self.total_charge_key].astype(chi.dtype).reshape(nsys)
                if self.total_charge_key in inputs
                else jnp.zeros(nsys, dtype=chi.dtype)
            )
            b = jnp.concatenate([Qtot, -chi])

            if self.non_interacting_guess:
                # build initial guess
                si = 1./Aii
                q0 = -chi*si
                qtot = jax.ops.segment_sum(q0,batch_index,nsys)
                sisum = jax.ops.segment_sum(si,batch_index,nsys)
                l0 = sisum*(qtot - Qtot)
                q0 = q0 - si*l0[batch_index]
                x0 = jnp.concatenate((l0,q0))
            else:
                x0 = None

            solver = self.solver.lower()
            if solver == "bicg":
                x = jax.scipy.sparse.linalg.bicgstab(matvec, b,x0=x0)[0]
            elif solver == "gmres":
                x = jax.scipy.sparse.linalg.gmres(matvec, b,x0=x0)[0]
            elif solver == "cg":
                print("Warning: Use of cg solver for Qeq is not recommended")
                x = jax.scipy.sparse.linalg.cg(matvec, b,x0=x0)[0]
            else:
                raise NotImplementedError(f"solver '{solver}' is not implemented. Choose one of [bicg, gmres]")


            q = x[nsys:]
            q_ = jax.lax.stop_gradient(q)

        eself = 0.5 * Aii * q_**2 + chi * q_

        phi = jax.ops.segment_sum(Aij * q_[edge_dst], edge_src, species.shape[0])
        if do_recip:
            erec, _ = ewald_reciprocal(q_, *ewald_params)
        epair = 0.5 * q_ * phi

        if self.charges_key in inputs:
            if self.c3_key is not None:
                c3 = c3 + inputs[self.c3_key]
            if self.c4_key is not None:
                c4 = c4 + inputs[self.c4_key]
            eself = eself + c3 * q_**3 + c4 * q_**4
            training = "training" in inputs.get("flags", {})
            if self.trainable and training:
                Aii_ = jax.lax.stop_gradient(Aii)
                chi_ = jax.lax.stop_gradient(chi)
                phi_ = jax.lax.stop_gradient(phi)
                c3_ = jax.lax.stop_gradient(c3)
                c4_ = jax.lax.stop_gradient(c4)
                switch_ = jax.lax.stop_gradient(switch)
                Aij_ = jax.lax.stop_gradient(Aij)
                phi_ = jax.ops.segment_sum(
                    Aij_ * q[edge_dst], edge_src, species.shape[0]
                )

                dedq = Aii_ * q + chi_ + phi_ + 3 * c3_ * q**2 + 4 * c4_ * q**3
                dedq = jax.ops.segment_sum(
                    switch_ * (dedq[edge_src] - dedq[edge_dst]) ** 2,
                    edge_src,
                    species.shape[0],
                )
                etrain = (
                    0.5 * Aii_ * q**2
                    + chi_ * q
                    + 0.5 * q * phi_
                    + c3_ * q**3
                    + c4_ * q**4
                )
                if do_recip:
                    etrain = etrain + erec

        energy = eself + epair
        if do_recip:
            energy = energy + erec

        energy_key = self.energy_key if self.energy_key is not None else self.name
        energy_unit = au.get_multiplier(self._energy_unit)
        output = {
            **inputs,
            self.charges_key: q,
            energy_key: energy*energy_unit,
        }
        if do_recip:
            output[energy_key + "_reciprocal"] = erec*energy_unit

        training = "training" in inputs.get("flags", {})
        if self.charges_key in inputs and self.trainable and training:
            output[energy_key + "_regularization"] = regularization
            output[energy_key + "_dedq"] = dedq*energy_unit
            output[energy_key + "_etrain"] = etrain*energy_unit
        return output


class ChargeCorrection(nn.Module):
    """Charge correction scheme
    
    FID: CHARGE_CORRECTION

    Used to correct the provided charges to sum to the total charge of the system.
    """
    key: str = "charges"
    """Key for the charges in the inputs"""
    output_key: str = None
    """Key for the corrected charges in the outputs. If None, it is the same as the input key"""
    dq_key: str = "delta_qtot"
    """Key for the deviation of the raw charge sum in the outputs"""
    ratioeta_key: str = None
    """Key for the ratio of hardness between AIM and free atom in the inputs. Used to adjust charge redistribution."""
    trainable: bool = False
    """Whether the parameters are trainable"""
    cn_key: str = None
    """Key for the coordination number in the inputs. Used to adjust charge redistribution."""
    total_charge_key: str = "total_charge"
    """Key for the total charge in the inputs"""
    _energy_unit: str = "Ha"
    """The energy unit of the model. **Automatically set by FENNIX**"""

    FID: ClassVar[str] = "CHARGE_CORRECTION"

    @nn.compact
    def __call__(self, inputs) -> Any:
        species = inputs["species"]
        batch_index = inputs["batch_index"]
        nsys = inputs["natoms"].shape[0]
        q = inputs[self.key]
        if q.shape[-1] == 1:
            q = jnp.squeeze(q, axis=-1)
        qtot = jax.ops.segment_sum(q, batch_index, nsys)
        Qtot = (
            inputs[self.total_charge_key].astype(q.dtype)
            if self.total_charge_key in inputs
            else jnp.zeros(qtot.shape[0], dtype=q.dtype)
        )
        dq = Qtot - qtot

        Jii = D3_HARDNESSES
        ai = D3_VDW_RADII
        eta = [jii + (2.0 / np.pi) ** 0.5 / aii for jii, aii in zip(Jii, ai)]
        if self.trainable:
            eta = jnp.abs(self.param("eta", lambda key: jnp.asarray(eta)))[species]
        else:
            eta = jnp.asarray(eta)[species]

        if self.ratioeta_key is not None:
            ratioeta = inputs[self.ratioeta_key]
            if ratioeta.shape[-1] == 1:
                ratioeta = jnp.squeeze(ratioeta, axis=-1)
            eta = eta * ratioeta

        s = (1.0e-6 + 2 * jnp.abs(eta)) ** (-1)
        if self.cn_key is not None:
            cn = inputs[self.cn_key]
            if cn.shape[-1] == 1:
                cn = jnp.squeeze(cn, axis=-1)
            s = s * cn

        f = dq / jax.ops.segment_sum(s, batch_index, nsys)

        qf = q + s * f[batch_index]

        energy_unit = au.get_multiplier(self._energy_unit)
        ecorr = (0.5*energy_unit) * eta * (qf - q) ** 2
        output_key = self.output_key if self.output_key is not None else self.key
        return {
            **inputs,
            output_key: qf,
            self.dq_key: dq,
            "charge_correction_energy": ecorr,
        }

class DistributeElectrons(nn.Module):
    """Distribute valence electrons between the atoms

    FID: DISTRIBUTE_ELECTRONS

    Used to predict charges that sum to the total charge of the system.
    """
    embedding_key: str
    """Key for the embedding in the inputs that is used to predict an 'electron affinity' weight"""
    output_key: Union[str,None] = None
    """Key for the charges in the outputs"""
    total_charge_key: str = "total_charge"
    """Key for the total charge in the inputs"""

    FID: ClassVar[str] = "DISTRIBUTE_ELECTRONS"

    @nn.compact
    def __call__(self, inputs) -> Any:
        species = inputs["species"]
        Nel = jnp.asarray(VALENCE_ELECTRONS)[species]

        ei = nn.Dense(1, use_bias=True, name="wi")(inputs[self.embedding_key]).squeeze(-1)
        wi = jax.nn.softplus(ei)

        batch_index = inputs["batch_index"]
        nsys = inputs["natoms"].shape[0]
        wtot = jax.ops.segment_sum(wi, inputs["batch_index"], inputs["natoms"].shape[0])

        Qtot = (
            inputs[self.total_charge_key].astype(ei.dtype)
            if self.total_charge_key in inputs
            else jnp.zeros(nsys, dtype=ei.dtype)
        )
        Neltot = jax.ops.segment_sum(Nel, batch_index, nsys) - Qtot

        f = Neltot / wtot
        Ni = wi* f[batch_index]
        q = Nel-Ni
        

        output_key = self.output_key if self.output_key is not None else self.name
        return {
            **inputs,
            output_key: q,
        }

