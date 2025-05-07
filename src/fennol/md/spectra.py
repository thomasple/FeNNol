import numpy as np
import flax.linen as nn
import jax
import jax.numpy as jnp
import math
import optax
import os
from pathlib import Path
from flax.core import freeze, unfreeze

from ..models.fennix import FENNIX
from ..utils.atomic_units import AtomicUnits as au  # CM1,THZ,BOHR,MPROT
from ..utils import Counter
from ..utils.deconvolution import (
    deconvolute_spectrum,
    kernel_lorentz_pot,
    kernel_lorentz,
)

def initialize_ir_spectrum(simulation_parameters,system_data,fprec,dt,apply_kubo_fact=False):
    state = {}

    parameters = simulation_parameters.get("ir_parameters", {})
    dipole_model = parameters["dipole_model"]
    dipole_model = Path(str(dipole_model).strip())
    if not dipole_model.exists():
        raise FileNotFoundError(f"Dipole model file {dipole_model} not found")
    else:
        print(f"# Using '{dipole_model}' as dipole model.")
        dipole_model = FENNIX.load(dipole_model)

        nblist_skin = simulation_parameters.get("nblist_skin", -1.0)
        pbc_data = system_data.get("pbc", None)

        ### CONFIGURE PREPROCESSING
        preproc_state = unfreeze(dipole_model.preproc_state)
        layer_state = []
        for st in preproc_state["layers_state"]:
            stnew = unfreeze(st)
            if pbc_data is not None:
                stnew["minimum_image"] = pbc_data["minimum_image"]
            if nblist_skin > 0:
                stnew["nblist_skin"] = nblist_skin
            if "nblist_mult_size" in simulation_parameters:
                stnew["nblist_mult_size"] = simulation_parameters["nblist_mult_size"]
            if "nblist_add_neigh" in simulation_parameters:
                stnew["add_neigh"] = simulation_parameters["nblist_add_neigh"]
            layer_state.append(freeze(stnew))
        preproc_state["layers_state"] = layer_state
        dipole_model.preproc_state = freeze(preproc_state)


    Tseg = parameters.get("tseg", 1.0 / au.PS) * au.FS
    nseg = int(Tseg / dt)
    Tseg = nseg * dt
    dom = 2 * np.pi / (3 * Tseg)
    omegacut = parameters.get("omegacut", 15000.0 / au.CM1) / au.FS
    nom = int(omegacut / dom)
    omega = dom * np.arange((3 * nseg) // 2 + 1)

    assert (
        omegacut < omega[-1]
    ), f"omegacut must be smaller than {omega[-1]*au.CM1} CM-1"

    startsave = parameters.get("startsave", 1)
    counter = Counter(nseg, startsave=startsave)
    state["istep"] = 0
    state["nsample"] = 0
    state["nadapt"] = 0

    use_qvel = parameters.get("use_qvel", False)
    if use_qvel:
        state["qvel"] = jnp.zeros((3,), dtype=fprec)
    else:
        nat = system_data["nat"]
        state["musave-2"] = jnp.zeros((3,), dtype=fprec)
        state["musave-1"] = jnp.zeros((3,), dtype=fprec)
        state["qsave-2"] = jnp.zeros((nat,1), dtype=fprec)
        state["qsave-1"] = jnp.zeros((nat,1), dtype=fprec)
        state["pos_save-2"] = jnp.zeros((nat,3), dtype=fprec)
        state["pos_save-1"] = jnp.zeros((nat,3), dtype=fprec)
    state["mudot"] = jnp.zeros((nseg, 3), dtype=fprec)
    state["Cmumu"] = jnp.zeros((nom,), dtype=fprec)
    state["first"] = True

    kT = system_data["kT"]
    kubo_fact = np.ones_like(omega)
    if apply_kubo_fact:
        uu = 0.5*omega[1:]*au.FS/kT
        kubo_fact[1:] = np.tanh(uu)/uu

    do_deconvolution = parameters.get("deconvolution", False)
    if do_deconvolution:
        gamma = simulation_parameters.get("gamma", 1.0 / au.THZ) / au.FS
        niter_deconv = parameters.get("niter_deconv", 20)
        print("# Deconvolution of IR spectra with gamma=", gamma*1000,"ps-1 and niter=",niter_deconv)
    

    kelvin = system_data["temperature"]
    c=2.99792458e-2 # speed of light in cm/ps
    # mufact = 1000*4*np.pi**2/(3*kT*c*au.BOHR**2)
    mufact = 1000*418.40*332.063714*2*np.pi**2/(0.831446215*kelvin*3.*c)
    pbc_data = system_data.get("pbc", None)
    if pbc_data is not None:
        cell = pbc_data["cell"]
        volume = np.abs(np.linalg.det(cell)) #/au.BOHR**3
        mufact = mufact/volume


    @jax.jit
    def compute_spectra(state):
        mudot = state["mudot"]
        smu = jnp.fft.rfft(mudot, 3 * nseg, axis=0, norm="ortho")
        Cmumu = dt * jnp.sum(jnp.abs(smu[:nom]) ** 2, axis=-1)

        nsinv = 1.0 / state["nsample"]
        b1 = 1.0 - nsinv
        Cmumu = state["Cmumu"] * b1 + Cmumu * nsinv

        return {
            **state,
            "Cmumu": Cmumu,
        }

    def write_spectra_to_file(state):
        Cmumu_avg = np.array(state["Cmumu"])*mufact*kubo_fact[:nom]

        if do_deconvolution:
            s_out, s_rec, K_D = deconvolute_spectrum(
                Cmumu_avg,
                omega[:nom],
                gamma,
                niter_deconv,
                kernel=kernel_lorentz,
                trans=False,
                symmetrize=True,
                verbose=False,
                K_D=state.get("K_D", None),
            )
            state = {**state, "K_D": K_D}
            spectra=(s_out,Cmumu_avg,s_rec)
        else:
            spectra=(Cmumu_avg,)

        columns = np.column_stack(
            (
                omega[:nom] * (au.FS * au.CM1),
                *spectra,
            )
        )
        np.savetxt(
            f"IR_spectrum.out",
            columns,
            fmt="%12.6f",
            header="#omega Cmudot",
        )
        print("# IR spectrum written.")

        return state
    
    @jax.jit
    def save_dipole(q,vel,pos,dip,cell_reciprocal, state):

        q=q.reshape(-1,1)
        if use_qvel:
            qvel = (q*vel).sum(axis=0)
        if "first" in state:
            state = {
                **state,
                "musave-2":dip,
                "musave-1":dip,
            }
            if use_qvel:
                state["qvel"] = qvel
            else:
                state["pos_save-2"] = pos
                state["pos_save-1"] = pos
                state["qsave-2"] = q
                state["qsave-1"] = q
            del state["first"]

        istep = state["istep"]
        new_state = {**state, "istep": istep + 1}
        new_state["musave-2"] = state["musave-1"]
        new_state["musave-1"] = dip
        
        dipdot = (dip - state["musave-2"])/(2*dt)
        if use_qvel:
            qrdot = state["qvel"]
            new_state["qvel"] = qvel
        else:
            new_state["pos_save-2"] = state["pos_save-1"]
            new_state["pos_save-1"] = pos
            new_state["qsave-2"] = state["qsave-1"]
            new_state["qsave-1"] = q

            if cell_reciprocal is not None:
                cell,reciprocal_cell = cell_reciprocal
                vec = pos - state["pos_save-2"]
                shift = -jnp.round(jnp.dot(vec, reciprocal_cell))
                pos = pos + jnp.dot(shift, cell)

            qrdot = (q*pos - state["qsave-2"]*state["pos_save-2"]).sum(axis=0)/(2*dt)

        new_state["mudot"] = state["mudot"].at[istep].set(qrdot+dipdot)

        return new_state

    # @jax.jit
    # def save_dipole(mudot, state):

    #     istep = state["istep"]
    #     new_state = {**state, "istep": istep + 1}
    #     new_state["mudot"] = state["mudot"].at[istep].set(mudot)

    #     return new_state
    
    def postprocess(state):
        counter.increment()
        if not counter.is_reset_step:
            return state
        state["nadapt"] += 1
        state["nsample"] = max(state["nadapt"] - startsave + 1, 1)
        state = compute_spectra(state)
        state["istep"] = 0
        state = write_spectra_to_file(state)
        return state
    
    return dipole_model,state,save_dipole, postprocess
        
        