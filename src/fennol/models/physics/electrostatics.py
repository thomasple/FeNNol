import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from jaxopt.linear_solve import solve_cg, solve_iterative_refinement, solve_gmres
from typing import Any, Dict, Union, Callable, Sequence, Optional
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

class EwaldReciprocal(nn.Module):
    cutoff: float
    kmax: int = 30
    graph_key: str = "graph"

    @nn.compact
    def __call__(self,inputs):

        ### Ewald reciprocal space
        A = pbc["cell_inv"]
        s = jnp.einsum("ij,abj->abi", A, coords)

        ks = 2j * jnp.pi * (pbc["k"][None, None, :, :] * s[:, :, None, :]).sum(dim=-1)
        Sm = (q[:, :, None] * jnp.exp(ks)).sum(dim=1)
        # Sm=(q[:,:,None]*jnp.exp(-ks)).sum(dim=1)

        ### compute reciprocal Coulomb potential (https://arxiv.org/abs/1805.10363)
        phi = jnp.real((pbc["expfac"][None, None, :] * Sm * jnp.exp(-ks)).sum(dim=-1)) * (
            au.BOHR / (jnp.pi * pbc["volume"])
        ) - q * (2 * pbc["bewald"] * au.BOHR / jnp.pi**0.5)

        # erec=jnp.real((pbc.expfac*Sp*Sm).sum(dim=-1))*(au.BOHR/(2*torch.pi*pbc.volume))
        # eself=q.pow(2).sum(dim=1)*(pbc.bewald*au.BOHR/torch.pi**0.5)

        return 0.5 * q * phi, phi


class Coulomb(nn.Module):
    _graphs_properties: Dict
    graph_key: str = "graph"
    charges_key: str = "charges"
    energy_key: Optional[str] = None
    # switch_fraction: float = 0.9
    scale: Optional[float] = None
    charge_scale: Optional[float] = None
    damp_style: Optional[str] = None
    damp_params: Dict = dataclasses.field(default_factory=dict)
    trainable: bool = True

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

        if damp_style is None:
            Aij = switch / rij
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

            Aij = (1.0 - eBij) / rij * switch
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

            Aij = (1.0 - eBij) / rij * switch
            eat = (
                0.5
                * q
                * jax.ops.segment_sum(Aij * q[edge_dst], edge_src, species.shape[0])
            )

        elif damp_style == "D3":
            if self.trainable:
                rvdw = jnp.abs(
                    self.param("rvdw", lambda key: jnp.asarray(D3_VDW_RADII))
                )[species]
            else:
                rvdw = jnp.asarray(D3_VDW_RADII)[species]

            ai2 = rvdw**2
            gamma_ij = (ai2[edge_src] + ai2[edge_dst] + 1.0e-3) ** (-0.5)

            Aij = jax.scipy.special.erf(gamma_ij * rij) / rij * switch
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
            s1 = jnp.where(x1 < 0.0, 0.0, jnp.exp(-1.0 / x1))
            s2 = jnp.where(x2 < 0.0, 0.0, jnp.exp(-1.0 / x2))
            Bij = s2 / (s1 + s2)

            Aij = Bij / (rij**2 + 1) ** 0.5 + (1 - Bij) / rij * switch
            eat = (
                0.5
                * q
                * jax.ops.segment_sum(Aij * q[edge_dst], edge_src, species.shape[0])
            )

        elif damp_style == "CP":
            cpA = self.damp_params.get("cpA", 4.42)
            cpB = self.damp_params.get("cpB", 4.12)
            gamma = self.damp_params.get("gamma", 0.1)

            ai2 = jnp.asarray(D3_VDW_RADII)[species] ** 2
            gamma_ij = (ai2[edge_src] + ai2[edge_dst] + 1.0e-3) ** (-0.5)

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

            epair = (
                # (1-jnp.exp(-gamma * distances**4))*
                (
                    Zi * Zj * (eAi + eAj + eBij)
                    - qi * Zj * (eAi + eBij)
                    - qj * Zi * (eAj + eBij)
                    + qi * qj * (1 + eBij)
                )
                * switch
                / rij
            )
            eat = 0.5 * jax.ops.segment_sum(epair, edge_src, species.shape[0])

        elif damp_style == "KEY":
            damp_key = self.damp_params["key"]
            damp = inputs[damp_key]
            epair = (1 - damp) * switch / rij
            eat = 0.5 * jax.ops.segment_sum(epair, edge_src, species.shape[0])
        else:
            raise NotImplementedError(f"damp_style {self.damp_style} not implemented")

        if self.scale is not None:
            if self.trainable:
                scale = jnp.abs(
                    self.param("scale", lambda key: jnp.asarray(self.scale))
                )
            else:
                scale = self.scale
            eat = eat * scale

        energy_key = self.energy_key if self.energy_key is not None else self.name
        return {**inputs, energy_key: eat}


class QeqD4(nn.Module):
    graph_key: str = "graph"
    trainable: bool = False
    charges_key: str = "charges"
    energy_key: Optional[str] = None
    ridge: float = 1.0e-6
    chi_key: str = None

    @nn.compact
    def __call__(self, inputs):
        species = inputs["species"]
        graph = inputs[self.graph_key]
        edge_src, edge_dst = graph["edge_src"], graph["edge_dst"]
        switch = graph["switch"]

        rij = graph["distances"] / au.BOHR

        if "cells" in inputs:
            raise NotImplementedError(
                "QeqD3 does not support periodic boundary conditions"
            )

        # D3 parameters
        if self.trainable:
            ENi = self.param("EN", lambda key: jnp.asarray(D3_ELECTRONEGATIVITIES))[
                species
            ]
            Jii = self.param("J", lambda key: jnp.asarray(D3_HARDNESSES))[species]
            ai = jnp.abs(self.param("a", lambda key: jnp.asarray(D3_VDW_RADII)))[
                species
            ]
            rci = jnp.abs(self.param("rc", lambda key: jnp.asarray(D3_COV_RADII)))[
                species
            ]
            kappai = self.param("kappa", lambda key: jnp.asarray(D3_KAPPA))[species]
            k1 = jnp.abs(self.param("k1", lambda key: jnp.asarray(7.5)))
        else:
            ENi = jnp.asarray(D3_ELECTRONEGATIVITIES)[species]
            Jii = jnp.asarray(D3_HARDNESSES)[species]
            ai = jnp.asarray(D3_VDW_RADII)[species]
            rci = jnp.asarray(D3_COV_RADII)[species]
            kappai = jnp.asarray(D3_KAPPA)[species]
            k1 = 7.5

        ai2 = ai**2
        rcij = rci[edge_src] + rci[edge_dst] + 1.e-3
        mCNij = (0.5 + 0.5 * jax.scipy.special.erf(-k1 * (rij / rcij - 1))) * switch
        mCNi = jax.ops.segment_sum(mCNij, edge_src, species.shape[0])
        chi = ENi - kappai * (mCNi + 1.0e-3) ** 0.5
        if self.chi_key is not None:
            chi = chi + inputs[self.chi_key]

        gamma_ij = (ai2[edge_src] + ai2[edge_dst] + 1.0e-3) ** (-0.5)

        Aii = Jii + ((2.0 / np.pi) ** 0.5) / ai
        Aij = jax.scipy.special.erf(gamma_ij * rij) / rij * switch

        nsys = inputs["natoms"].shape[0]
        isys = inputs["isys"]

        def matvec(x):
            l = x[:nsys]
            q = x[nsys:]
            Aq_self = Aii * q
            Aq_pair = jax.ops.segment_sum(Aij * q[edge_dst], edge_src, species.shape[0])
            Aq = Aq_self + Aq_pair + l[isys]
            Al = jax.ops.segment_sum(q, isys, nsys)
            return jnp.concatenate([Al, Aq])

        Qtot = inputs["total_charge"] if "total_charge" in inputs else jnp.zeros(nsys)
        b = jnp.concatenate([Qtot, -chi])
        x = solve_cg(matvec, b, ridge=self.ridge)

        q = x[nsys:]
        eself = 0.5 * Aii * q**2 + chi * q
        epair = (
            0.5 * q * jax.ops.segment_sum(Aij * q[edge_dst], edge_src, species.shape[0])
        )

        energy = eself + epair

        energy_key = self.energy_key if self.energy_key is not None else self.name

        return {
            **inputs,
            self.charges_key: q,
            energy_key: energy,
            "chi": chi,
            "coordination": 2 * mCNi,
        }


class ChargeCorrection(nn.Module):
    key: str = "charges"
    key_out: str = None
    dq_key: str = "delta_qtot"
    ratioeta_key: str = None
    trainable: bool = False

    @nn.compact
    def __call__(self, inputs) -> Any:
        species = inputs["species"]
        isys = inputs["isys"]
        nsys = inputs["natoms"].shape[0]
        q = inputs[self.key]
        if q.shape[-1] == 1:
            q = jnp.squeeze(q, axis=-1)
        qtot = jax.ops.segment_sum(q, isys, nsys)
        Qtot = (
            inputs["total_charge"]
            if "total_charge" in inputs
            else jnp.zeros(qtot.shape[0])
        )
        dq = Qtot - qtot

        if self.trainable:
            eta = self.param("eta", lambda key: jnp.asarray(D3_HARDNESSES))[species]
        else:
            eta = jnp.asarray(D3_HARDNESSES)[species]

        if self.ratioeta_key is not None:
            ratioeta = inputs[self.ratioeta_key]
            if ratioeta.shape[-1] == 1:
                ratioeta = jnp.squeeze(ratioeta, axis=-1)
            eta = eta * ratioeta

        s = (1.0e-6 + 2 * eta) ** (-1)

        f = dq / jax.ops.segment_sum(s, isys, nsys)

        q = q + s * f[isys]
        key_out = self.key_out if self.key_out is not None else self.key
        return {**inputs, key_out: q, self.dq_key: dq}
